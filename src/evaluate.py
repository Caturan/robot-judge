"""Run the judge on the test split and score it against humans.

  python src/evaluate.py --model muse-spark-1.3 --limit 40
  python src/evaluate.py --model muse-spark-1.3 --score-only
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import judge  # noqa: E402
from judge import run, usage  # noqa: E402
from metrics import bt_elo, human_ceiling, judge_scores  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="muse-spark-1.3")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()

    judge.MAX_TOKENS = args.max_tokens
    raw = os.path.join(ROOT, "data", "raw")
    pairs = pd.read_parquet(f"{raw}/pairs/{args.split}.parquet")
    responses = pd.read_parquet(f"{raw}/responses/{args.split}.parquet")
    tasks = pd.read_parquet(f"{raw}/tasks/{args.split}.parquet")
    task_text = dict(zip(tasks.task_id, tasks.instruction_shown))

    subset = pairs.sample(frac=1, random_state=0)
    if args.limit:
        subset = subset.head(args.limit)

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out = os.path.join(ROOT, "results", f"{args.split}_{args.model}.jsonl")
    if args.retry_failed and os.path.exists(out):
        keep = [l for l in open(out) if json.loads(l)["p_a"] is not None]
        open(out, "w").writelines(keep)
    if not args.score_only:
        run(subset, task_text, args.model, out, workers=args.workers)
        print("usage:", usage)

    recs = [json.loads(l) for l in open(out)]
    p_a = pd.Series({r["pair_key"]: r["p_a"] for r in recs if r["p_a"] is not None})
    errors = sum(r["p_a"] is None for r in recs)
    scored = pairs[pairs.pair_key.isin(p_a.index)]

    flips = [r for r in recs if "winner" in r["fwd"] and "winner" in r["rev"]]
    consistent = sum(r["fwd"]["winner"] == r["rev"]["winner"] for r in flips) / max(1, len(flips))

    report = {
        "model": args.model,
        "failed_pairs": errors,
        "order_consistency": consistent,
        "judge": judge_scores(scored, p_a),
        "human_ceiling": human_ceiling(scored, responses[responses.pair_key.isin(p_a.index)]),
        "elo_human": bt_elo(scored).round(1).to_dict(),
        "elo_judge": bt_elo(scored, p_a).round(1).to_dict(),
    }
    print(json.dumps(report, indent=2))
    json.dump(report, open(out.replace(".jsonl", "_report.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
