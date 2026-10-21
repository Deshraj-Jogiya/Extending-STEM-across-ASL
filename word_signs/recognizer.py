"""Recognise a word sign from webcam frames or an uploaded video, for the Flask app.

The browser captures about two seconds of webcam video, takes frames at 15 per second, and posts them as JPEG data URLs. OpenCV
decodes and downsizes each frame, features.py turns them into one hand-feature sequence, and the trained SignGRU returns the
top signs with their probabilities. Timings for each stage come back with the answer.
"""

from __future__ import annotations

import base64
import os
import tempfile
import threading
import time

import cv2
import numpy as np

from word_signs.features import HandFeatureExtractor
from word_signs.model import load, predict_proba

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "sign_gru.pt")
MAX_SIDE = 480


def decode_frame(data_url: str) -> np.ndarray | None:
    """JPEG/PNG data URL (or bare base64) -> BGR frame no larger than MAX_SIDE on its long side."""
    raw = base64.b64decode(data_url.split(",", 1)[-1])
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return None
    scale = MAX_SIDE / max(frame.shape[:2])
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame


class SignRecognizer:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model, self.vocabulary = load(model_path)
        self._extractor = HandFeatureExtractor()  # built once: loading the landmarker costs more than a request's frames
        self._lock = threading.Lock()  # the MediaPipe landmarker is not safe to share between request threads

    def _rank(self, seq: np.ndarray, top: int) -> list[dict]:
        proba = predict_proba(self.model, seq[None])[0]
        return [{"sign": self.vocabulary[i], "probability": round(float(proba[i]), 4)} for i in np.argsort(-proba)[:top]]

    def from_frames(self, frames_bgr: list[np.ndarray], top: int = 3) -> dict:
        t0 = time.perf_counter()
        with self._lock:
            seq = self._extractor.sequence(frames_bgr)
        t1 = time.perf_counter()
        ranked = self._rank(seq, top)
        t2 = time.perf_counter()
        return {"predictions": ranked, "frames": len(frames_bgr),
                "hands_found_share": round(float((seq[:, 63] + seq[:, 133] > 0).mean()), 2),
                "feature_ms": round((t1 - t0) * 1000, 1), "model_ms": round((t2 - t1) * 1000, 1)}

    def from_data_urls(self, data_urls: list[str], top: int = 3) -> dict:
        t0 = time.perf_counter()
        frames = [f for f in (decode_frame(u) for u in data_urls) if f is not None]
        decode_ms = (time.perf_counter() - t0) * 1000
        if not frames:
            raise ValueError("none of the frames could be decoded")
        result = self.from_frames(frames, top)
        result["decode_ms"] = round(decode_ms, 1)
        result["total_ms"] = round(decode_ms + result["feature_ms"] + result["model_ms"], 1)
        return result

    def from_video_bytes(self, data: bytes, suffix: str = ".mp4", top: int = 3) -> dict:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
            path = f.name
        try:
            cap = cv2.VideoCapture(path)
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            cap.release()
        finally:
            os.unlink(path)
        if not frames:
            raise ValueError("the video has no readable frames")
        return self.from_frames(frames, top)
