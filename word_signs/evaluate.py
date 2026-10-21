"""Train SignGRU on the saved features and measure it on held-out clips.

Usage: python -m word_signs.evaluate
Reads word_signs/data/features.npz, holds out 2 clips per sign (seeded), trains on the rest, and writes
exports/word_sign_metrics.json (top-1, top-3, per-sign accuracy, inference time per clip), exports/word_sign_confusion.csv and
word_signs/models/sign_gru.pt (the model trained on every clip, which the Flask app loads).
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from word_signs.model import predict_proba, save, split, train

HERE = os.path.dirname(os.path.abspath(__file__))
EXPORTS = os.path.join(os.path.dirname(HERE), "exports")


def load_features(path: str = os.path.join(HERE, "data", "features.npz")):
    d = np.load(path)
    return d["X"].astype(np.float32), d["y"].astype(np.int64), [str(v) for v in d["vocabulary"]]


def topk(proba: np.ndarray, y: np.ndarray, k: int) -> float:
    return float(np.mean([t in row for t, row in zip(y, np.argsort(-proba, axis=1)[:, :k])]))


def main() -> dict:
    X, y, vocabulary = load_features()
    n = len(vocabulary)
    tr, te = split(y)
    model = train(X[tr], y[tr], n)
    start = time.perf_counter()
    proba = predict_proba(model, X[te])
    per_clip_ms = (time.perf_counter() - start) * 1000 / len(te)
    pred = proba.argmax(axis=1)
    confusion = np.zeros((n, n), dtype=int)
    for t, p in zip(y[te], pred):
        confusion[t, p] += 1
    metrics = {
        "signs": n, "clips": int(len(y)), "train_clips": int(len(tr)), "test_clips": int(len(te)),
        "top1_accuracy": round(float((pred == y[te]).mean()), 4), "top3_accuracy": round(topk(proba, y[te], 3), 4),
        "chance_accuracy": round(1 / n, 4), "model_ms_per_clip": round(per_clip_ms, 2),
        "per_sign_accuracy": {vocabulary[c]: round(float(confusion[c, c] / max(confusion[c].sum(), 1)), 2) for c in range(n)},
    }
    os.makedirs(EXPORTS, exist_ok=True)
    with open(os.path.join(EXPORTS, "word_sign_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(EXPORTS, "word_sign_confusion.csv"), "w", encoding="utf-8") as f:
        f.write("actual," + ",".join(vocabulary) + "\n")
        for c in range(n):
            f.write(vocabulary[c] + "," + ",".join(map(str, confusion[c])) + "\n")
    save(train(X, y, n), vocabulary, os.path.join(HERE, "models", "sign_gru.pt"))
    print(f"{n} signs, {len(te)} held-out clips: top-1 {metrics['top1_accuracy']:.1%}, top-3 {metrics['top3_accuracy']:.1%} "
          f"(chance {metrics['chance_accuracy']:.1%}), {per_clip_ms:.1f} ms per clip")
    return metrics


if __name__ == "__main__":
    main()
