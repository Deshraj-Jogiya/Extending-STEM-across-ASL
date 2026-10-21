"""Time the webcam path end to end through the Flask route.

Usage: python -m word_signs.latency <data_dir>
Takes 20 clips, turns 30 evenly spaced frames of each into 640x480 JPEG data URLs (what the /signs page sends for two seconds at 15
frames per second), posts them to /recognize-sign with Flask's test client and records the time per stage. Writes
exports/webcam_latency.json with the medians.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time

import cv2
import numpy as np

from word_signs.evaluate import EXPORTS

FRAMES = 30


def data_urls(path: str) -> list[str]:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    urls = []
    for i in np.linspace(0, len(frames) - 1, FRAMES).round().astype(int):
        ok, jpg = cv2.imencode(".jpg", cv2.resize(frames[i], (640, 480)), [cv2.IMWRITE_JPEG_QUALITY, 80])
        urls.append("data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode())
    return urls


def main(data_dir: str, n: int = 20) -> dict:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from learning_platform.app import app
    selection = json.load(open(os.path.join(data_dir, "selection.json"), encoding="utf-8"))
    clips = selection["clips"][::max(len(selection["clips"]) // n, 1)][:n]
    client = app.test_client()
    client.post("/recognize-sign", json={"frames": data_urls(os.path.join(data_dir, clips[0]["filepath"]))})  # load the model
    rows = []
    for clip in clips:
        frames = data_urls(os.path.join(data_dir, clip["filepath"]))
        t0 = time.perf_counter()
        body = client.post("/recognize-sign", json={"frames": frames}).get_json()
        rows.append({**{k: body[k] for k in ("decode_ms", "feature_ms", "model_ms", "total_ms")},
                     "request_ms": (time.perf_counter() - t0) * 1000})
    result = {"requests": len(rows), "frames_per_request": FRAMES,
              **{f"median_{k}": round(float(np.median([r[k] for r in rows])), 1)
                 for k in ("decode_ms", "feature_ms", "model_ms", "total_ms", "request_ms")}}
    os.makedirs(EXPORTS, exist_ok=True)
    with open(os.path.join(EXPORTS, "webcam_latency.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"{len(rows)} requests of {FRAMES} frames: median {result['median_request_ms']:.0f} ms per request "
          f"(decode {result['median_decode_ms']:.0f}, hand features {result['median_feature_ms']:.0f}, "
          f"model {result['median_model_ms']:.0f} ms)")
    return result


if __name__ == "__main__":
    main(sys.argv[1])
