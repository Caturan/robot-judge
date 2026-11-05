"""Train a Bradley-Terry judge on frozen V-JEPA 2 features and the training votes.

score(clip) = w . phi(clip);  P(A better) = sigmoid(score_A - score_B)
Fit on the vote share label_a of every training pair (soft labels).

Settings (feature set, per-task normalisation, regularisation) are chosen by
leave-one-task-out CV over the 7 training tasks, so model selection always
scores on a task it has not seen. cup_stacking (test) is scored once, at the end.

  .venv-vjepa/bin/python src/train_vjepa.py
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
from metrics import judge_scores  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw")


def clip_features(feats, kind):
    f = feats.astype(np.float32)  # (8 time steps, 1024)
    parts = {"mean": f.mean(0), "final": f[-1], "delta": f[-1] - f[0]}
    return np.concatenate([parts[k] for k in kind.split("+")])


def build(pairs, feats, kind, task_norm):
    keys = sorted(set(pairs.video_a_key) | set(pairs.video_b_key))
    emb = pd.DataFrame({k: clip_features(feats[k], kind) for k in keys}).T
    if task_norm:  # standardise within each task; uses clips only, never labels
        task = emb.index.str.split("/").str[1]
        emb = emb.groupby(task).transform(lambda c: (c - c.mean()) / (c.std() + 1e-6))
    return emb.loc[pairs.video_a_key].values - emb.loc[pairs.video_b_key].values


def fit(X, y, C):
    # soft labels -> two weighted rows per pair; symmetric (no intercept)
    clf = LogisticRegression(C=C, fit_intercept=False, max_iter=2000)
    clf.fit(np.vstack([X, X]), np.r_[np.ones(len(X)), np.zeros(len(X))], sample_weight=np.r_[y, 1 - y])
    return clf


def predict(clf, X, pairs):
    return pd.Series(clf.predict_proba(X)[:, 1], index=pairs.pair_key.values)


def main():
    feats = dict(np.load(os.path.join(ROOT, "data", "vjepa_features.npz")))
    train = pd.read_parquet(f"{RAW}/pairs/train.parquet")
    test = pd.read_parquet(f"{RAW}/pairs/test.parquet")
    have = lambda p: p[p.video_a_key.isin(feats) & p.video_b_key.isin(feats)]
    train, test = have(train), have(test)
    print(f"train pairs {len(train)}, test pairs {len(test)}")

    grid = list(itertools.product(["mean", "final", "mean+final", "mean+final+delta"], [False, True], [0.001, 0.01, 0.1]))
    cv = []
    for kind, norm, C in grid:
        X = build(train, feats, kind, norm)
        accs = []
        for task in sorted(train.task_id.unique()):
            tr, va = (train.task_id != task).values, (train.task_id == task).values
            clf = fit(X[tr], train.label_a.values[tr], C)
            accs.append(judge_scores(train[va], predict(clf, X[va], train[va]))["majority_accuracy"])
        cv.append({"features": kind, "task_norm": norm, "C": C, "loto_acc": np.mean(accs), "per_task": accs})
        print(f"{kind:18s} norm={norm!s:5s} C={C:<6} LOTO acc={np.mean(accs):.3f}", flush=True)

    best = max(cv, key=lambda r: r["loto_acc"])
    print("best:", {k: v for k, v in best.items() if k != "per_task"})
    X_tr = build(train, feats, best["features"], best["task_norm"])
    X_te = build(test, feats, best["features"], best["task_norm"])
    clf = fit(X_tr, train.label_a.values, best["C"])
    p_te = predict(clf, X_te, test)
    scores = judge_scores(test, p_te)
    print("TEST cup_stacking:", json.dumps(scores, indent=2))

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    p_te.rename("p_a").to_csv(os.path.join(ROOT, "results", "test_vjepa.csv"))
    json.dump({"best": best, "test": scores, "cv": cv}, open(os.path.join(ROOT, "results", "vjepa_report.json"), "w"),
              indent=2, default=float)


if __name__ == "__main__":
    main()
