"""Extract hand-feature sequences for every clip in a WLASL selection and save them to word_signs/data/features.npz.

Usage: python -m word_signs.build_dataset <data_dir>
<data_dir> holds selection.json (vocabulary and clip list) and the clips under the paths it names, as written by
word_signs/download.py. The saved features let the model train and be tested without the videos.
"""

import json
import os
import sys
import time

import numpy as np

from word_signs.features import FEATURES_PER_FRAME, SEQ_LEN, HandFeatureExtractor

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "features.npz")


def main(data_dir: str) -> None:
    selection = json.load(open(os.path.join(data_dir, "selection.json"), encoding="utf-8"))
    vocabulary = selection["vocabulary"]
    extractor = HandFeatureExtractor()
    X, y, paths, hand_frames = [], [], [], []
    start = time.perf_counter()
    try:
        for clip in selection["clips"]:
            seq = extractor.video(os.path.join(data_dir, clip["filepath"]))
            X.append(seq)
            y.append(vocabulary.index(clip["gloss"]))
            paths.append(clip["filepath"])
            hand_frames.append(float((seq[:, 63] + seq[:, 63 + 70] > 0).mean()))
    finally:
        extractor.close()
    X = np.stack(X).astype(np.float32)
    assert X.shape[1:] == (SEQ_LEN, FEATURES_PER_FRAME)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, X=X.astype(np.float16), y=np.array(y), vocabulary=np.array(vocabulary), paths=np.array(paths))
    print(f"{len(X)} clips, {len(vocabulary)} signs, hands found in {np.mean(hand_frames):.0%} of sampled frames, "
          f"{time.perf_counter() - start:.0f} s -> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
