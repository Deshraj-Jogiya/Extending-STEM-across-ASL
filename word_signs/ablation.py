"""Do the OpenCV hand-shape values help? Retrains SignGRU on the same seeded split with the 12 contour values per frame set to zero
(landmarks only) and compares held-out accuracy with the full features.

Usage: python -m word_signs.ablation   (writes exports/word_sign_ablation.json)
"""

from __future__ import annotations

import json
import os

from word_signs.evaluate import EXPORTS, load_features, topk
from word_signs.features import LANDMARKS, PER_HAND
from word_signs.model import predict_proba, split, train


def main() -> dict:
    X, y, vocabulary = load_features()
    tr, te = split(y)
    results = {}
    for name, landmarks_only in (("landmarks_and_contours", False), ("landmarks_only", True)):
        Xv = X.copy()
        if landmarks_only:
            for slot in (0, 1):
                Xv[..., slot * PER_HAND + LANDMARKS * 3 + 1:(slot + 1) * PER_HAND] = 0
        proba = predict_proba(train(Xv[tr], y[tr], len(vocabulary)), Xv[te])
        results[name] = {"top1_accuracy": round(float((proba.argmax(1) == y[te]).mean()), 4),
                         "top3_accuracy": round(topk(proba, y[te], 3), 4)}
        print(name, results[name])
    results["test_clips"] = int(len(te))
    with open(os.path.join(EXPORTS, "word_sign_ablation.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    main()
