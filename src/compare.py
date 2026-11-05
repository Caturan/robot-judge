"""Summarise all runs into results/summary.md and the README results block.

  python src/compare.py
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from metrics import bt_elo, human_ceiling, judge_scores  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw")
RUNS = {"Muse Spark 1.3": "muse-spark-1.3", "Muse Wonder 1.0": "muse-wonder-1.0"}
pairs = pd.read_parquet(f"{RAW}/pairs/test.parquet")
responses = pd.read_parquet(f"{RAW}/responses/test.parquet")
names = dict(pd.read_parquet(f"{RAW}/models/models.parquet")[["model_id", "display_name"]].values)


def load(model):
    recs = [json.loads(l) for l in open(os.path.join(ROOT, "results", f"test_{model}.jsonl"))]
    return pd.Series({r["pair_key"]: r["p_a"] for r in recs if r["p_a"] is not None}), recs


def boot_ci(sub, p_a, key, n=1000):
    rng = np.random.default_rng(0)
    keys = sub.pair_key.values
    vals = []
    for _ in range(n):
        s = sub.set_index("pair_key").loc[rng.choice(keys, len(keys))].reset_index()
        s["pair_key"] = s.pair_key + "_" + s.groupby("pair_key").cumcount().astype(str)
        pp = p_a.reindex(s.pair_key.str.rsplit("_", n=1).str[0]).set_axis(s.pair_key)
        vals.append(judge_scores(s, pp)[key])
    return np.percentile(vals, [2.5, 97.5])


def human_ci(sub, n=1000):
    rng = np.random.default_rng(0)
    r = responses[responses.pair_key.isin(sub.pair_key)]
    v = r.merge(sub[["pair_key", "votes_a"]], on="pair_key")
    pick = (v.chosen == "a").astype(int)
    ok = (pick == ((v.votes_a - pick) >= 5)).groupby(v.pair_key).mean()
    return np.percentile([ok.sample(len(ok), replace=True, random_state=int(rng.integers(1e9))).mean()
                          for _ in range(n)], [2.5, 97.5])


def pct(x):
    return f"{100 * x:.1f}%"


lines = []
loaded = {label: load(m) for label, m in RUNS.items()}
common = set.intersection(*(set(p.index) for p, _ in loaded.values()))

for scope, keyset in [("All test pairs", None), (f"Head-to-head on the same {len(common)} pairs", common)]:
    lines += [f"**{scope}**", "",
              "| | Pairs | Agrees with human majority (95% CI) | Clear-cut pairs (≥8–2) | Pearson r vs vote share | Order consistency |",
              "|---|---:|---|---:|---:|---:|"]
    hum_sub = None
    for label, (p_a, recs) in loaded.items():
        keys = set(p_a.index) if keyset is None else keyset
        sub = pairs[pairs.pair_key.isin(keys)]
        if keyset is None and label != "Muse Spark 1.3":
            continue  # full-set row only for the model that ran on everything
        s = judge_scores(sub, p_a)
        lo, hi = boot_ci(sub, p_a, "majority_accuracy")
        both = [r for r in recs if r["pair_key"] in keys and "winner" in r["fwd"] and "winner" in r["rev"]]
        cons = np.mean([r["fwd"]["winner"] == r["rev"]["winner"] for r in both])
        lines.append(f"| {label} | {s['n_pairs']} | **{pct(s['majority_accuracy'])}** ({pct(lo)}–{pct(hi)}) | "
                     f"{pct(s['majority_accuracy_strong(>=8)'])} | {s['pearson_r']:.2f} | {pct(cons)} |")
        hum_sub = sub
    h = human_ceiling(hum_sub, responses[responses.pair_key.isin(hum_sub.pair_key)])
    lo, hi = human_ci(hum_sub)
    lines += [f"| Single human vs. other 9 (ceiling) | {len(hum_sub)} | {pct(h['human_vs_majority_of_other9'])} "
              f"({pct(lo)}–{pct(hi)}) | – | – | – |", ""]

spark, _ = loaded["Muse Spark 1.3"]
full = pairs[pairs.pair_key.isin(spark.index)]
eh, ej = bt_elo(full), bt_elo(full, spark)
lines += ["**Model ranking on `cup_stacking`** (Bradley-Terry Elo from human votes vs. from the judge's verdicts)", "",
          "| Rank | Humans | Elo | Muse Spark judge | Elo |", "|---:|---|---:|---|---:|"]
for i, ((hm, hv), (jm, jv)) in enumerate(zip(eh.items(), ej.items()), 1):
    lines.append(f"| {i} | {names[hm]} | {hv:.0f} | {names[jm]} | {jv:.0f} |")
gap = (ej - eh.reindex(ej.index)).abs()
top4 = eh.drop(eh.idxmin())
lines += ["", f"Judge Elo is within **{gap.mean():.0f} points** of human Elo on average (max {gap.max():.0f}). "
          f"Both put {names[eh.idxmin()]} clearly last (~{top4.min() - eh.min():.0f} Elo below the rest). "
          f"The other four sit within {top4.max() - top4.min():.0f} Elo of each other even on human votes, "
          "so their order is within noise for humans and judge alike.", ""]

by_cond = []
for cond, g in full.groupby("condition"):
    by_cond.append(f"{cond.replace('_', '-')}: {pct(judge_scores(g, spark)['majority_accuracy'])} ({len(g)} pairs)")
lines.append("Muse Spark agreement by layout: " + "; ".join(by_cond) + ".")

train_path = os.path.join(ROOT, "results", "train_muse-spark-1.3.jsonl")
if os.path.exists(train_path):
    tr_pairs = pd.read_parquet(f"{RAW}/pairs/train.parquet")
    tr_p = pd.Series({r["pair_key"]: r["p_a"] for r in map(json.loads, open(train_path)) if r["p_a"] is not None})
    tr = tr_pairs[tr_pairs.pair_key.isin(tr_p.index)]
    tr_resp = pd.read_parquet(f"{RAW}/responses/train.parquet")

    lines += ["", "**Every task.** Muse Spark on a random sample of pairs from the 7 training tasks "
              "(these tasks were never used to tune the prompt either).", "",
              "| Task | Pairs | Judge agrees with majority | Single human vs. other 9 |", "|---|---:|---:|---:|"]
    rows = [(t, g) for t, g in tr.groupby("task_id")] + [("cup_stacking (test)", full)]
    for t, g in rows:
        pp = spark if t.startswith("cup_stacking") else tr_p
        resp = responses if t.startswith("cup_stacking") else tr_resp
        h = human_ceiling(g, resp[resp.pair_key.isin(g.pair_key)])
        lines.append(f"| `{t}` | {len(g)} | {pct(judge_scores(g, pp)['majority_accuracy'])} | "
                     f"{pct(h['human_vs_majority_of_other9'])} |")
    allp = pd.concat([tr, full])
    allpp = pd.concat([tr_p, spark])
    h = human_ceiling(allp, pd.concat([tr_resp, responses]).pipe(lambda r: r[r.pair_key.isin(allp.pair_key)]))
    lines.append(f"| **All 8 tasks** | {len(allp)} | **{pct(judge_scores(allp, allpp)['majority_accuracy'])}** | "
                 f"{pct(h['human_vs_majority_of_other9'])} |")

    # Calibration: fit p_cal = sigmoid(a*logit(p) + b) to the train vote shares, apply to test
    def logit(p):
        p = np.clip(p, 1e-3, 1 - 1e-3)
        return np.log(p / (1 - p))

    x, y = logit(tr_p.reindex(tr.pair_key).values), tr.label_a.values
    a, b = 1.0, 0.0
    for _ in range(500):  # gradient descent on soft-label log-loss
        q = 1 / (1 + np.exp(-(a * x + b)))
        a -= 0.5 * np.mean((q - y) * x)
        b -= 0.5 * np.mean(q - y)
    cal = pd.Series(1 / (1 + np.exp(-(a * logit(spark.values) + b))), index=spark.index)
    before, after = judge_scores(full, spark), judge_scores(full, cal)
    lines += ["", f"**Calibrated with training votes.** We fit `p = sigmoid({a:.2f}·logit(p_judge) {b:+.2f})` to the "
              f"human vote shares of the {len(tr)} training pairs, then applied it unchanged to the test task. "
              f"Test log-loss goes from {before['log_loss']:.3f} to **{after['log_loss']:.3f}**, and agreement with the "
              f"majority stays at {pct(after['majority_accuracy'])}. The confidence now reads like a vote share:", "",
              "| Judge confidence (calibrated) | Pairs | Share of humans who agreed with the judge |", "|---|---:|---:|"]
    conf = np.maximum(cal, 1 - cal)
    agree = np.where(cal.values >= 0.5, full.set_index("pair_key").label_a.reindex(cal.index).values,
                     1 - full.set_index("pair_key").label_a.reindex(cal.index).values)
    for lo, hi in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]:
        m = (conf.values >= lo) & (conf.values < hi)
        if m.sum():
            lines.append(f"| {int(lo * 100)}–{min(int(hi * 100), 100)}% | {m.sum()} | {pct(agree[m].mean())} |")

vj_path = os.path.join(ROOT, "results", "test_vjepa.csv")
if os.path.exists(vj_path):
    vj = pd.read_csv(vj_path, index_col=0).p_a
    vj_rep = json.load(open(os.path.join(ROOT, "results", "vjepa_report.json")))
    tr_pairs = pd.read_parquet(f"{RAW}/pairs/train.parquet")
    elo_tr = bt_elo(tr_pairs)  # policy strength from training tasks only
    policy = pd.Series(1 / (1 + 10 ** ((elo_tr[full.model_b].values - elo_tr[full.model_a].values) / 400)),
                       index=full.pair_key.values)
    no_x = full[(full.model_a != "xvla") & (full.model_b != "xvla")]
    d = full[full.votes_a != 5]
    prior_wrong = d[(policy.reindex(d.pair_key).values > 0.5) != (d.votes_a > 5).values]

    def acc(sub, p):
        return pct(judge_scores(sub, p)["majority_accuracy"])

    b = vj_rep["best"]
    lines += ["", "**A trained judge on the 49k training votes.** Frozen V-JEPA 2 (ViT-L) features for each clip "
              "(16 frames), and a linear Bradley-Terry scorer trained on the vote shares of all "
              f"{len(tr_pairs):,} training pairs. Settings were chosen by leave-one-task-out CV on the 7 training tasks "
              f"(best: `{b['features']}` features, C={b['C']}, LOTO accuracy {pct(b['loto_acc'])}); "
              "the test task was scored once.", "",
              "| Judge | All 607 test pairs | Without X-VLA | Pairs where the policy prior is wrong | Cost / speed per pair |",
              "|---|---:|---:|---:|---|",
              f"| Policy identity only (no video) | {acc(full, policy)} | {acc(no_x, policy)} | 0% | – |",
              f"| V-JEPA 2 + linear head (trained) | **{acc(full, vj)}** | {acc(no_x, vj)} | "
              f"{acc(prior_wrong, vj)} | free, ~1 s on a laptop |",
              f"| Muse Spark 1.3 (zero-shot) | **{acc(full, spark)}** | {acc(no_x, spark)} | "
              f"{acc(prior_wrong, spark)} | ~$0.07, ~40 s |",
              f"| Single human vs. other 9 | {pct(human_ceiling(full, responses)['human_vs_majority_of_other9'])} | – | – | – |",
              "",
              "The *policy identity* row predicts every pair from each policy's training-task Elo without watching "
              "anything. Both judges beat it by a wide margin, and stay well above chance on the "
              f"{len(prior_wrong)} pairs where the usually-weaker policy actually won. So they are judging the run, "
              "not recognising the robot.", "",
              "Muse stays ahead overall. On the 7 training tasks, V-JEPA averages "
              f"{pct(b['loto_acc'])} (each task scored by a model trained on the other six), against "
              f"{pct(np.mean([judge_scores(g, tr_p)['majority_accuracy'] for _, g in tr.groupby('task_id')]))} for Muse "
              "on its sample, so the small gap on the test task partly reflects that `cup_stacking` suits V-JEPA. "
              "The trained judge is the cheap first-pass filter, and Muse is the accurate judge that explains itself."]

md = "\n".join(lines)
open(os.path.join(ROOT, "results", "summary.md"), "w").write(md + "\n")
readme = os.path.join(ROOT, "README.md")
text = open(readme).read()
text = re.sub(r"<!-- RESULTS -->.*<!-- /RESULTS -->", "<!-- RESULTS -->\n" + md + "\n<!-- /RESULTS -->", text, flags=re.S)
open(readme, "w").write(text)
print(md)
