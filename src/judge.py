"""Zero-shot robot-run judge on the Meta Model API (Muse).

Each clip becomes a storyboard: N evenly spaced frames plus the final frame,
tiled into one labelled image. The judge sees both storyboards and the task,
and answers in JSON. Every pair is judged in both orders (A/B and B/A) to
cancel position bias; p_a is the average.
"""
import base64
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
API = "https://api.meta.ai/v1/chat/completions"
CLIPS = os.path.join(os.path.dirname(__file__), "..", "data", "clips")

PROMPT = """You are judging two real bimanual robot runs from a fixed top camera.
Task given to both robots: "{task}"

Each image is a storyboard of one run: frames in time order, left to right then top to bottom, timestamps in the corner. The last tile is the final state.
Image 1 = Run {first}. Image 2 = Run {second}.

Decide which robot did the task better, the way a careful human viewer would: credit task completion first, then partial progress toward the goal (e.g. how many objects end up correctly placed), then fewer mistakes (drops, knocking things over, collisions, flailing or stalling).

Reply with only JSON:
{{"run_1_summary": "<one sentence>", "run_2_summary": "<one sentence>", "reason": "<one or two sentences comparing them>", "winner": "{first}" or "{second}", "confidence": <0.5 to 1.0>}}"""


def storyboard(video_key: str, n: int = 8, cols: int = 3, width: int = 320) -> str:
    cap = cv2.VideoCapture(os.path.join(CLIPS, video_key.replace("/", "__") + ".mp4"))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    # final 1.5s is a freeze-frame hold; sample the run, then the last frame
    run_end = max(1, total - int(1.5 * fps) - 1)
    idx = list(np.linspace(0, run_end, n, dtype=int)) + [total - 1]
    tiles = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            f = tiles[-1].copy() if tiles else np.zeros((180, width, 3), np.uint8)
        else:
            f = cv2.resize(f, (width, int(f.shape[0] * width / f.shape[1])))
        label = "FINAL" if i == idx[-1] else f"t={i / fps * 2:.0f}s"  # clips play at 2x
        cv2.putText(f, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(f, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tiles.append(f)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
    ok, buf = cv2.imencode(".jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


_usage_lock = threading.Lock()
usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}


MAX_TOKENS = 4000  # reasoning models: muse-wonder needs ~16000 on hard pairs


def ask(model: str, task: str, img1: str, img2: str, first="A", second="B", retries=3) -> dict:
    content = [
        {"type": "text", "text": PROMPT.format(task=task, first=first, second=second)},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img1}},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img2}},
    ]
    for attempt in range(retries):
        try:
            r = requests.post(
                API,
                headers={"Authorization": "Bearer " + os.environ["META_API_KEY"]},
                json={"model": model, "max_tokens": MAX_TOKENS, "messages": [{"role": "user", "content": content}]},
                timeout=600,
            )
            r.raise_for_status()
            body = r.json()
            with _usage_lock:
                usage["calls"] += 1
                for k in ("prompt_tokens", "completion_tokens"):
                    usage[k] += body["usage"][k]
            text = body["choices"][0]["message"]["content"] or ""
            out = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
            if out.get("winner") not in (first, second):
                raise ValueError(f"bad winner: {out.get('winner')}")
            out["confidence"] = float(np.clip(float(out.get("confidence", 0.5)), 0.5, 1.0))
            return out
        except Exception as e:  # noqa: BLE001 - retry any API/parse failure
            err = f"{type(e).__name__}: {e}"
    return {"error": err}


def judge_pair(model: str, task: str, key_a: str, key_b: str) -> dict:
    """Judge both orders. Returns p_a plus the raw verdicts."""
    a, b = storyboard(key_a), storyboard(key_b)
    fwd = ask(model, task, a, b, "A", "B")
    rev = ask(model, task, b, a, "B", "A")
    ps = []
    for v in (fwd, rev):
        if "winner" in v:
            ps.append(v["confidence"] if v["winner"] == "A" else 1 - v["confidence"])
    return {"p_a": float(np.mean(ps)) if ps else None, "fwd": fwd, "rev": rev}


def run(pairs, task_text: dict, model: str, out_path: str, workers: int = 8):
    """Judge every pair, appending JSONL to out_path (resumable)."""
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {json.loads(l)["pair_key"] for l in f}
    todo = pairs[~pairs.pair_key.isin(done)]
    lock = threading.Lock()
    with ThreadPoolExecutor(workers) as ex, open(out_path, "a") as f:
        futs = {
            ex.submit(judge_pair, model, task_text[r.task_id], r.video_a_key, r.video_b_key): r.pair_key
            for r in todo.itertuples()
        }
        for i, fut in enumerate(as_completed(futs), 1):
            rec = {"pair_key": futs[fut], "model": model, **fut.result()}
            with lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()
            if i % 20 == 0:
                print(f"{i}/{len(todo)}  usage={usage}", flush=True)
