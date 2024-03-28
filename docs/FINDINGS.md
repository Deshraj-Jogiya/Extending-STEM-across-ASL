# Findings

Measured results from the word-sign rebuild. Every number below comes from a file in this repository, named under each finding.
Clips are from the public WLASL set (29 STEM and computing signs, 214 clips).

## How well are word signs recognised on held-out clips?

**How it was tested:** 2 clips of every sign are held out (seeded); the model is trained on the rest (`word_signs/evaluate.py`).
Each clip becomes MediaPipe hand landmarks plus OpenCV hand-crop features, resampled to 32 frames, fed to a 1-D convolution and a
bidirectional GRU.

| Result | Value |
|---|---|
| Top-1 accuracy | 0.7586 |
| Top-3 accuracy | 0.9483 |
| Chance (1 of 29) | 0.0345 |
| Model time per clip | 0.34 ms |

**Decision:** the browser page (`/signs`) shows the top three signs for each recording, not only the top one.

**Skills shown:** MediaPipe Hand Landmarker, OpenCV, PyTorch (Conv1d + bidirectional GRU), seeded holdout evaluation.

**Source:** `exports/word_sign_metrics.json`, `word_signs/evaluate.py`, `word_signs/model.py`.

## Which signs does the model miss, and why?

**How it was tested:** per-sign accuracy and the confusion matrix on the same held-out clips.

| Sign | Held-out accuracy | Predicted as |
|---|---|---|
| network | 0.0 | internet (both clips) |
| chemistry | 0.0 | email, experiment |
| internet | 0.5 | network (one clip) |
| paper | 0.5 | print (one clip) |
| school | 0.5 | paper (one clip) |

**What it shows:** the misses fall between related computing and school signs (network and internet; paper, print and school),
the kind of signs the hand-shape analysis below finds close together.

**Skills shown:** confusion-matrix error analysis, pandas.

**Source:** `exports/word_sign_metrics.json` (`per_sign_accuracy`), `exports/word_sign_confusion.csv`,
`exports/gesture_redundancy.json`.

## How many distinct hand shapes do the 29 signs need?

**How it was tested:** each clip's hand shape is how far each finger is bent and spread while the sign is held; two signs share a
hand shape when a permutation test cannot tell their clips apart (p ≥ 0.05) (`word_signs/redundancy.py`).

| Result | Value |
|---|---|
| Distinct hand shapes | 18 |
| Hand shapes saved | 11 |
| Reduction in shapes to practise | 0.3793 |
| Sign pairs sharing a shape | 55 |

Shared groups include email, program, write and learn; letter, energy and research; computer and network; machine, paper and school.

**Decision:** a lesson teaches each shared shape once, and lesson search merges concepts that share a hand shape (`search/`).

**Skills shown:** feature engineering on landmarks, permutation testing, Python.

**Source:** `exports/gesture_redundancy.json`, `word_signs/redundancy.py`.

## Is the webcam path fast enough to use live?

**How it was tested:** 20 requests of 30 JPEG frames each (a two-second recording) sent to `/recognize-sign`, timed per stage on a
laptop CPU (`word_signs/latency.py`).

| Stage (median) | Time |
|---|---|
| Decode frames | 76.2 ms |
| Hand features | 997.0 ms |
| Model | 8.9 ms |
| Total | 1081.2 ms |
| Whole request | 1090.5 ms |

**Decision:** the hand landmarker, which takes nearly all the time, is built once and reused across requests, and each answer
returns its time per stage.

**Skills shown:** Flask endpoint, browser getUserMedia capture, OpenCV decoding, per-stage latency profiling.

**Source:** `exports/webcam_latency.json`, `word_signs/latency.py`.
