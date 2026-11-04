#!/usr/bin/env bash
# Download the robot-policy-human-preferences-55k tables and test-split videos.
# Usage: scripts/download.sh [--train-videos]
set -euo pipefail
cd "$(dirname "$0")/.."

T=$(cat ~/.cache/huggingface/token)
B=https://huggingface.co/datasets/datapointai/robot-policy-human-preferences-55k/resolve/main/data
OUT=data/raw

files=(models/models pairs/train pairs/test responses/train responses/test tasks/train tasks/test)
videos=(videos/test/test-0000)
if [[ "${1:-}" == "--train-videos" ]]; then
  videos+=(videos/train/train-000{0..4})
fi

for f in "${files[@]}" "${videos[@]}"; do
  mkdir -p "$OUT/$(dirname "$f")"
  [[ -s "$OUT/$f.parquet" ]] && continue
  echo "-> $f"
  curl -fL --progress-bar -H "Authorization: Bearer $T" "$B/$f.parquet" -o "$OUT/$f.parquet"
done
