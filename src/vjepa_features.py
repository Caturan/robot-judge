"""Encode every clip with V-JEPA 2 (frozen) and save per-time-step features.

Each clip -> 16 frames (15 evenly over the run + the final frame) at 256x256.
V-JEPA 2 groups frames in pairs (tubelets), so we keep 8 time steps,
mean-pooled over space: features[video_key] = float16 array (8, 1024).

  .venv-vjepa/bin/python src/vjepa_features.py [--limit 5]
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch
from transformers import AutoVideoProcessor, VJEPA2Model

ROOT = os.path.join(os.path.dirname(__file__), "..")
CLIPS = os.path.join(ROOT, "data", "clips")
OUT = os.path.join(ROOT, "data", "vjepa_features.npz")
MODEL = "facebook/vjepa2-vitl-fpc64-256"
N_FRAMES = 16


def load_frames(path):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    run_end = max(1, total - int(1.5 * fps) - 1)  # skip the final-frame hold
    idx = list(np.linspace(0, run_end, N_FRAMES - 1, dtype=int)) + [total - 1]
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            f = frames[-1] if frames else np.zeros((360, 640, 3), np.uint8)
        else:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        frames.append(f)
    return np.stack(frames)  # (T, H, W, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoVideoProcessor.from_pretrained(MODEL)
    model = VJEPA2Model.from_pretrained(MODEL, torch_dtype=torch.float16).to(device).eval()

    feats = dict(np.load(OUT)) if os.path.exists(OUT) else {}
    paths = sorted(glob.glob(os.path.join(CLIPS, "*.mp4")))
    todo = [p for p in paths if os.path.basename(p)[:-4].replace("__", "/") not in feats][: args.limit]
    print(f"{len(feats)} done, {len(todo)} to go on {device}", flush=True)

    t0 = time.time()
    for i, p in enumerate(todo, 1):
        key = os.path.basename(p)[:-4].replace("__", "/")
        inputs = processor(load_frames(p), return_tensors="pt").to(device)
        with torch.no_grad():
            h = model(pixel_values_videos=inputs["pixel_values_videos"].half(), skip_predictor=True).last_hidden_state
        t_steps = N_FRAMES // 2
        feats[key] = h[0].float().reshape(t_steps, -1, h.shape[-1]).mean(1).cpu().numpy().astype(np.float16)
        if i % 50 == 0 or i == len(todo):
            np.savez(OUT, **feats)
            rate = (time.time() - t0) / i
            print(f"{i}/{len(todo)}  {rate:.2f}s/clip  eta {rate * (len(todo) - i) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
