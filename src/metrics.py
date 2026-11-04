"""Agreement metrics for a robot-video judge against 10 human votes per pair.

A judge outputs p_a = P(run A is better). We compare it to humans three ways:
  - majority accuracy: does argmax(p_a) match the 10-vote majority (5-5 ties skipped)
  - vote agreement: expected agreement with one randomly drawn human vote
  - log-loss / Pearson r against the vote fraction label_a
The human ceiling scores each annotator the same way against the other 9.
"""
import numpy as np
import pandas as pd


def human_ceiling(pairs: pd.DataFrame, responses: pd.DataFrame) -> dict:
    v = responses.merge(pairs[["pair_key", "votes_a"]], on="pair_key")
    picked_a = (v.chosen == "a").astype(int)
    others_a = v.votes_a - picked_a  # A-votes among the other 9
    agree_others = np.where(picked_a == 1, others_a, 9 - others_a) / 9
    majority_others = np.where(others_a >= 5, 1, 0)  # 9 voters, so never tied
    return {
        "human_vs_majority_of_other9": float((picked_a == majority_others).mean()),
        "human_vote_agreement": float(agree_others.mean()),
    }


def judge_scores(pairs: pd.DataFrame, p_a: pd.Series) -> dict:
    df = pairs.assign(p_a=p_a.reindex(pairs.pair_key).values).dropna(subset=["p_a"])
    decisive = df[df.votes_a != 5]
    pred_a = decisive.p_a > 0.5
    hard = (decisive.p_a == 0.5).mean()
    p = df.p_a.clip(1e-3, 1 - 1e-3)
    return {
        "n_pairs": len(df),
        "majority_accuracy": float((pred_a == (decisive.votes_a > 5)).mean()),
        "majority_accuracy_strong(>=8)": float(
            (pred_a == (decisive.votes_a > 5))[(decisive.votes_a >= 8) | (decisive.votes_a <= 2)].mean()
        ),
        "abstain_rate": float(hard),
        "vote_agreement": float((df.p_a * df.label_a + (1 - df.p_a) * (1 - df.label_a)).mean()),
        "log_loss": float(-(df.label_a * np.log(p) + (1 - df.label_a) * np.log(1 - p)).mean()),
        "pearson_r": float(np.corrcoef(df.p_a, df.label_a)[0, 1]),
    }


def bt_elo(pairs: pd.DataFrame, p_a: pd.Series = None, iters: int = 200) -> pd.Series:
    """Bradley-Terry Elo from soft wins. Uses human label_a unless p_a is given."""
    w = pairs.label_a.values if p_a is None else p_a.reindex(pairs.pair_key).values
    a, b = pairs.model_a.values, pairs.model_b.values
    models = sorted(set(a) | set(b))
    s = {m: 1.0 for m in models}
    for _ in range(iters):
        new = {}
        for m in models:
            wins = np.sum(w[a == m]) + np.sum(1 - w[b == m])
            denom = sum(1 / (s[x] + s[y]) for x, y in zip(a, b) if m in (x, y))
            new[m] = wins / denom
        g = np.exp(np.mean(np.log(list(new.values()))))
        s = {m: v / g for m, v in new.items()}
    return pd.Series({m: 1000 + 400 * np.log10(v) for m, v in s.items()}).sort_values(ascending=False)
