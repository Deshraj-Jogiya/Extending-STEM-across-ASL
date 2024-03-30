"""Exports the word-sign model and test data for the portfolio Lab demo, which runs it in the browser.

Usage: python -m word_signs.export_lab <output_dir>

Writes, into <output_dir>:
- sign_gru.onnx: SignGRU trained on the same seeded split evaluate.py scores (so the held-out clips below were never seen in
  training), exported for ONNX Runtime Web. Input "features" (1, 32, 140), output "logits" (1, 29).
- held_out.json: the 58 held-out clips as feature sequences (no video: WLASL clips are for research use only), each with its true
  sign and the probabilities PyTorch gives it. Values are rounded to 5 decimals before scoring (after training), so the browser reads exactly the
  numbers Python scored.
- contour/: 30 synthetic hand-like frames (PNG, lossless) with landmark points and the 6 contour_features() values Python
  measures on each. The browser computes the same features with OpenCV.js; the Lab's CI check compares the two.
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
import torch

from word_signs.evaluate import load_features, topk
from word_signs.features import LANDMARKS, PER_HAND, SEQ_LEN, FEATURES_PER_FRAME, contour_features
from word_signs.model import predict_proba, split, train

CONTOUR_FRAMES = 30


def synthetic_frame(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A palm with 1-5 raised fingers on a noisy, shaded background, and 21 points spread over it (like landmarks)."""
    h, w = 120, 160
    yy, xx = np.mgrid[0:h, 0:w]
    base = rng.uniform(40, 200)
    bg = base + rng.uniform(-40, 40) * xx / w + rng.uniform(-30, 30) * yy / h
    frame = np.repeat(bg[..., None], 3, axis=2) + rng.normal(0, 6, (h, w, 3))
    cx, cy, r = rng.uniform(55, 105), rng.uniform(55, 80), rng.uniform(14, 22)
    skin = np.array([rng.uniform(60, 120), rng.uniform(100, 160), rng.uniform(150, 230)])  # BGR
    if abs(skin.mean() - base) < 40:
        skin = np.clip(skin + (60 if base < 128 else -60), 0, 255)
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (int(cx), int(cy)), (int(r), int(r * 1.15)), float(rng.uniform(-20, 20)), 0, 360, 255, -1)
    tips = []
    for k in range(int(rng.integers(1, 6))):
        a = np.deg2rad(-150 + k * 30 + rng.uniform(-8, 8))
        length = r * rng.uniform(1.4, 2.1)
        tip = (int(cx + length * np.cos(a)), int(cy + length * np.sin(a)))
        cv2.line(mask, (int(cx), int(cy)), tip, 255, int(rng.uniform(5, 8)))
        tips.append(tip)
    frame[mask > 0] = skin + rng.normal(0, 5, (int((mask > 0).sum()), 3))
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    pick = rng.choice(len(xs), size=LANDMARKS, replace=False)
    # points on a 1/1024 grid are exact in float32 and float64, so Python and JavaScript crop the same pixels
    points = np.stack([np.round(xs[pick] / w * 1024) / 1024, np.round(ys[pick] / h * 1024) / 1024,
                       np.zeros(LANDMARKS)], axis=1).astype(np.float32)
    return frame, points


def main(out_dir: str) -> None:
    os.makedirs(os.path.join(out_dir, "contour"), exist_ok=True)
    X, y, vocabulary = load_features()
    tr, te = split(y)
    model = train(X[tr], y[tr], len(vocabulary))  # same data and seed as evaluate.py, so the same 0.7586 top-1
    X = np.round(X, 5).astype(np.float32)
    proba = predict_proba(model, X[te])
    print(f"held-out top-1 {float((proba.argmax(1) == y[te]).mean()):.4f}, top-3 {topk(proba, y[te], 3):.4f}")

    torch.onnx.export(model, torch.zeros(1, SEQ_LEN, FEATURES_PER_FRAME), os.path.join(out_dir, "sign_gru.onnx"),
                      input_names=["features"], output_names=["logits"], opset_version=17, dynamo=False)
    clips = [{"sign": vocabulary[int(y[i])], "probs": [round(float(p), 6) for p in proba[k]],
              "features": [[float(v) for v in frame] for frame in X[i]]} for k, i in enumerate(te)]
    with open(os.path.join(out_dir, "held_out.json"), "w", encoding="utf-8") as f:
        json.dump({"vocabulary": vocabulary, "seq_len": SEQ_LEN, "features_per_frame": FEATURES_PER_FRAME,
                   "per_hand": PER_HAND, "clips": clips}, f, separators=(",", ":"))

    rng = np.random.default_rng(42)
    frames = []
    for n in range(CONTOUR_FRAMES):
        frame, points = synthetic_frame(rng)
        name = f"frame_{n:02d}.png"
        cv2.imwrite(os.path.join(out_dir, "contour", name), frame)
        frames.append({"file": name, "points": points[:, :2].tolist(),
                       "features": [round(float(v), 6) for v in contour_features(frame, points)]})
    with open(os.path.join(out_dir, "contour", "contour.json"), "w", encoding="utf-8") as f:
        json.dump({"frames": frames}, f, indent=1)
    print(f"{len(clips)} held-out clips and {len(frames)} contour frames written to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1])
