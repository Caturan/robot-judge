"""Write each clip in the videos parquet shards to data/clips/<video_key>.mp4."""
import glob
import os

import pyarrow.parquet as pq

ROOT = os.path.join(os.path.dirname(__file__), "..")
out = os.path.join(ROOT, "data", "clips")
os.makedirs(out, exist_ok=True)
for shard in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "videos", "*", "*.parquet"))):
    t = pq.read_table(shard, columns=["video", "video_key"]).to_pylist()
    for row in t:
        with open(os.path.join(out, row["video_key"].replace("/", "__") + ".mp4"), "wb") as f:
            f.write(row["video"]["bytes"])
    print(shard, len(t))
