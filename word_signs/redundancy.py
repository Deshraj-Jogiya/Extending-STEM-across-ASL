"""Measure gesture redundancy across the vocabulary: how many signs share a hand shape.

Usage: python -m word_signs.redundancy
A clip's hand shape is how each finger is bent (the angle at each of its three joints) and how far apart neighbouring fingertips are,
for the hand seen in more frames, taken as the median over the middle half of the frames where the hand is seen (where a sign holds
its shape). Two signs share a hand shape when a permutation test cannot tell their clips apart (distance between the two signs'
mean shapes, 999 label shuffles, p >= 0.05). Signs are grouped greedily so that every pair inside a group shares a hand shape.

A lesson plan that teaches each shared hand shape once, then the signs built on it, has fewer hand shapes to practise than one sign
at a time; the reduction and the groups are written to exports/gesture_redundancy.json. With 6 to 14 clips per sign the test has
limited power, so a pair the data cannot separate is not proof the hand shapes are identical.
"""

from __future__ import annotations

import itertools
import json
import os

import numpy as np

from word_signs.evaluate import EXPORTS, load_features
from word_signs.features import LANDMARKS, PER_HAND

FINGERS = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]  # base to tip
NEIGHBOUR_TIPS = [(4, 8), (8, 12), (12, 16), (16, 20)]


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    u, w = a - b, c - b
    cos = (u * w).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(w, axis=-1) + 1e-9)
    return np.arccos(np.clip(cos, -1, 1))


def hand_shape(clip: np.ndarray) -> np.ndarray | None:
    """19 numbers: 15 joint angles and 4 fingertip gaps, or None when no hand is seen."""
    presence = clip[:, [LANDMARKS * 3, PER_HAND + LANDMARKS * 3]]
    slot = int(presence.sum(axis=0).argmax())
    seen = np.flatnonzero(presence[:, slot] > 0)
    if len(seen) == 0:
        return None
    hold = seen[len(seen) // 4:max(len(seen) * 3 // 4, len(seen) // 4 + 1)]
    pts = clip[hold, slot * PER_HAND:slot * PER_HAND + LANDMARKS * 3].reshape(len(hold), LANDMARKS, 3)
    parts = []
    for a, b, c, d in FINGERS:
        parts += [_angle(pts[:, 0], pts[:, a], pts[:, b]), _angle(pts[:, a], pts[:, b], pts[:, c]), _angle(pts[:, b], pts[:, c], pts[:, d])]
    parts += [np.linalg.norm(pts[:, i] - pts[:, j], axis=-1) for i, j in NEIGHBOUR_TIPS]
    return np.median(np.stack(parts, axis=1), axis=0)


def same_shape(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, shuffles: int = 999, alpha: float = 0.05) -> bool:
    """True when shuffling the clips between the two signs separates their mean shapes as often as the real labels do."""
    z, n = np.vstack([a, b]), len(a)
    observed = np.linalg.norm(a.mean(axis=0) - b.mean(axis=0))
    hits = 0
    for _ in range(shuffles):
        p = rng.permutation(len(z))
        hits += np.linalg.norm(z[p[:n]].mean(axis=0) - z[p[n:]].mean(axis=0)) >= observed
    return (hits + 1) / (shuffles + 1) >= alpha


def group(shared: np.ndarray) -> list[list[int]]:
    """Greedy groups in which every pair shares a hand shape, starting from the sign that shares with the most others."""
    left, groups = set(range(len(shared))), []
    while left:
        seed = max(left, key=lambda i: (shared[i, list(left)].sum(), -i))
        members = [seed]
        for j in sorted(left - {seed}, key=lambda j: (-shared[j, list(left)].sum(), j)):
            if all(shared[j, m] for m in members):
                members.append(j)
        groups.append(members)
        left -= set(members)
    return groups


def main() -> dict:
    X, y, vocabulary = load_features()
    shapes = [hand_shape(c) for c in X]
    keep = [i for i, s in enumerate(shapes) if s is not None]
    S, ys = np.array([shapes[i] for i in keep]), y[keep]
    S = (S - S.mean(axis=0)) / S.std(axis=0)
    signs = np.unique(ys)
    rng = np.random.default_rng(42)
    shared = np.zeros((len(signs), len(signs)), dtype=bool)
    for i, j in itertools.combinations(range(len(signs)), 2):
        shared[i, j] = shared[j, i] = same_shape(S[ys == signs[i]], S[ys == signs[j]], rng)
    groups = [[vocabulary[signs[i]] for i in g] for g in group(shared)]
    result = {
        "signs": int(len(signs)), "distinct_hand_shapes": len(groups), "hand_shapes_saved": int(len(signs) - len(groups)),
        "reduction": round(1 - len(groups) / len(signs), 4), "pairs_sharing": int(shared.sum() // 2),
        "shared_groups": [g for g in groups if len(g) > 1],
    }
    os.makedirs(EXPORTS, exist_ok=True)
    with open(os.path.join(EXPORTS, "gesture_redundancy.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"{result['signs']} signs use {result['distinct_hand_shapes']} distinct hand shapes: teaching shared shapes once "
          f"cuts the hand shapes to practise by {result['reduction']:.0%}")
    for g in result["shared_groups"]:
        print("  shared:", ", ".join(g))
    return result


if __name__ == "__main__":
    main()
