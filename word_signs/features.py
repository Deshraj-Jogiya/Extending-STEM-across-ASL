"""Turn a sign video into a fixed-length sequence of hand features.

Per frame, MediaPipe's hand landmarker gives 21 landmarks for up to two hands. Each hand's landmarks are made relative to its wrist
and scaled by the wrist-to-middle-knuckle distance, so where the signer stands and how close they are to the camera drop out. OpenCV
then crops each hand from the frame, segments it, and measures the largest contour (area share of the crop, solidity, aspect ratio,
convexity defects), which describes the hand shape independently of the landmark model. Frames are resampled to SEQ_LEN.

Feature layout per frame (FEATURES_PER_FRAME = 140): for each of the two hand slots (right, then left) 63 landmark values
(21 x, y, z), 1 presence flag and 6 contour values.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

SEQ_LEN = 32
LANDMARKS = 21
CONTOUR_FEATURES = 6
PER_HAND = LANDMARKS * 3 + 1 + CONTOUR_FEATURES
FEATURES_PER_FRAME = 2 * PER_HAND
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task")


def normalise_landmarks(points: np.ndarray) -> np.ndarray:
    """(21, 3) landmarks -> relative to the wrist (point 0), scaled by the wrist to middle-finger knuckle (point 9) distance."""
    rel = points - points[0]
    scale = np.linalg.norm(rel[9, :2])
    return rel / scale if scale > 1e-6 else rel


def contour_features(frame_bgr: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Shape of the hand's largest contour inside the landmarks' bounding box (padded 20%)."""
    h, w = frame_bgr.shape[:2]
    xs, ys = points[:, 0] * w, points[:, 1] * h
    pad_x, pad_y = 0.2 * (xs.max() - xs.min() + 1), 0.2 * (ys.max() - ys.min() + 1)
    x0, x1 = int(max(xs.min() - pad_x, 0)), int(min(xs.max() + pad_x, w))
    y0, y1 = int(max(ys.min() - pad_y, 0)), int(min(ys.max() + pad_y, h))
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return np.zeros(CONTOUR_FEATURES, dtype=np.float32)
    gray = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # keep the side of the threshold that holds the hand: the one containing the crop's centre
    cy, cx = mask.shape[0] // 2, mask.shape[1] // 2
    if mask[cy, cx] == 0:
        mask = cv2.bitwise_not(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(CONTOUR_FEATURES, dtype=np.float32)
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    x, y, bw, bh = cv2.boundingRect(c)
    perimeter = cv2.arcLength(c, True)
    defects = 0
    if len(c) > 3:
        hull_idx = cv2.convexHull(c, returnPoints=False)
        if hull_idx is not None and len(hull_idx) > 3:
            try:
                d = cv2.convexityDefects(c, hull_idx)
                defects = 0 if d is None else int((d.reshape(-1, 4)[:, 3] > 256 * 0.1 * max(bw, bh)).sum())
            except cv2.error:
                defects = 0
    return np.array([
        area / float(mask.size),
        area / hull_area if hull_area > 0 else 0.0,
        bw / float(bh) if bh > 0 else 0.0,
        4 * np.pi * area / perimeter ** 2 if perimeter > 0 else 0.0,
        min(defects, 5) / 5.0,
        len(contours) / 10.0,
    ], dtype=np.float32)


def frame_features(frame_bgr: np.ndarray, result) -> np.ndarray:
    """Features for one frame from a HandLandmarkerResult; missing hands stay zero with presence 0."""
    out = np.zeros(FEATURES_PER_FRAME, dtype=np.float32)
    for points_list, handedness in zip(result.hand_landmarks, result.handedness):
        slot = 0 if handedness[0].category_name == "Right" else 1
        points = np.array([[p.x, p.y, p.z] for p in points_list], dtype=np.float32)
        start = slot * PER_HAND
        out[start:start + LANDMARKS * 3] = normalise_landmarks(points).ravel()
        out[start + LANDMARKS * 3] = 1.0
        out[start + LANDMARKS * 3 + 1:start + PER_HAND] = contour_features(frame_bgr, points)
    return out


def resample(frames: np.ndarray, length: int = SEQ_LEN) -> np.ndarray:
    """Pick `length` evenly spaced frames (repeating frames for short clips)."""
    if len(frames) == 0:
        return np.zeros((length, FEATURES_PER_FRAME), dtype=np.float32)
    idx = np.linspace(0, len(frames) - 1, length).round().astype(int)
    return frames[idx]


class HandFeatureExtractor:
    """Wraps MediaPipe's hand landmarker in video mode."""

    def __init__(self, model_path: str = DEFAULT_MODEL):
        from mediapipe.tasks.python import BaseOptions, vision
        options = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path),
                                               running_mode=vision.RunningMode.VIDEO, num_hands=2)
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._clock_ms = 0

    def sequence(self, frames_bgr) -> np.ndarray:
        """Features for an iterable of BGR frames, resampled to SEQ_LEN."""
        import mediapipe as mp
        rows = []
        for frame in frames_bgr:
            self._clock_ms += 40  # the video-mode landmarker needs strictly increasing timestamps
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            rows.append(frame_features(frame, self._landmarker.detect_for_video(image, self._clock_ms)))
        return resample(np.array(rows, dtype=np.float32))

    def video(self, path: str) -> np.ndarray:
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        return self.sequence(frames)

    def close(self):
        self._landmarker.close()
