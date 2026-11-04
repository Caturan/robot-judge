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

md = "\n".join(lines)
open(os.path.join(ROOT, "results", "summary.md"), "w").write(md + "\n")
readme = os.path.join(ROOT, "README.md")
text = open(readme).read()
text = re.sub(r"<!-- RESULTS -->.*<!-- /RESULTS -->", "<!-- RESULTS -->\n" + md + "\n<!-- /RESULTS -->", text, flags=re.S)
open(readme, "w").write(text)
print(md)
