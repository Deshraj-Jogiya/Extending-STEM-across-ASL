import base64
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from word_signs.features import (CONTOUR_FEATURES, FEATURES_PER_FRAME, LANDMARKS, PER_HAND, SEQ_LEN, contour_features,
                                 normalise_landmarks, resample)
from word_signs.model import SignGRU, mirror, predict_proba, split
from word_signs.recognizer import MAX_SIDE, decode_frame


def test_normalised_landmarks_ignore_position_and_scale():
    rng = np.random.default_rng(0)
    pts = rng.random((LANDMARKS, 3)).astype(np.float32)
    moved = pts * 2.5 + np.array([0.3, -0.1, 0.0], dtype=np.float32)
    moved[:, 2] = pts[:, 2] * 2.5
    assert np.allclose(normalise_landmarks(pts), normalise_landmarks(moved), atol=1e-5)


def test_contour_features_describe_a_drawn_hand_blob():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(frame, (100, 100), 40, (255, 255, 255), -1)
    pts = np.array([[0.3 + 0.4 * (i % 2), 0.3 + 0.4 * (i // 11), 0] for i in range(LANDMARKS)], dtype=np.float32)
    feats = contour_features(frame, pts)
    assert feats.shape == (CONTOUR_FEATURES,)
    assert 0.9 < feats[1] <= 1.0          # a disc is convex: solidity near 1
    assert 0.8 < feats[3] <= 1.1          # and round: circularity near 1


def test_resample_to_fixed_length():
    assert resample(np.ones((5, FEATURES_PER_FRAME), np.float32)).shape == (SEQ_LEN, FEATURES_PER_FRAME)
    assert resample(np.ones((90, FEATURES_PER_FRAME), np.float32)).shape == (SEQ_LEN, FEATURES_PER_FRAME)
    assert not resample(np.zeros((0, FEATURES_PER_FRAME), np.float32)).any()


def test_mirror_swaps_hands_and_is_its_own_inverse():
    x = np.random.default_rng(1).random((SEQ_LEN, FEATURES_PER_FRAME)).astype(np.float32)
    m = mirror(x)
    assert np.allclose(m[:, PER_HAND + 1], x[:, 1])          # y of the right hand's wrist moves to the left slot
    assert np.allclose(m[:, PER_HAND], -x[:, 0])             # and x is flipped
    assert np.allclose(mirror(m), x)


def test_split_holds_out_two_per_sign_and_never_overlaps():
    y = np.repeat(np.arange(5), 7)
    tr, te = split(y)
    assert len(np.intersect1d(tr, te)) == 0 and len(tr) + len(te) == len(y)
    assert all((y[te] == c).sum() == 2 for c in range(5))
    assert np.array_equal(te, split(y)[1])                    # seeded


def test_model_outputs_a_distribution_per_clip():
    model = SignGRU(n_classes=7).eval()
    proba = predict_proba(model, np.zeros((3, SEQ_LEN, FEATURES_PER_FRAME), np.float32))
    assert proba.shape == (3, 7) and np.allclose(proba.sum(axis=1), 1, atol=1e-5)


def test_decode_frame_downsizes_large_frames():
    ok, jpg = cv2.imencode(".jpg", np.zeros((960, 1280, 3), np.uint8))
    frame = decode_frame("data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode())
    assert max(frame.shape[:2]) == MAX_SIDE
    assert decode_frame(base64.b64encode(b"not an image").decode()) is None


@pytest.fixture
def client(monkeypatch):
    from learning_platform import app as app_module

    class FakeRecognizer:
        vocabulary = ["computer", "email"]

        def from_data_urls(self, frames, top=3):
            return {"predictions": [{"sign": "computer", "probability": 0.9}], "frames": len(frames)}

    monkeypatch.setattr(app_module, "_get_sign_recognizer", lambda: FakeRecognizer())
    return app_module.app.test_client()


def test_recognize_sign_route(client):
    assert client.post("/recognize-sign", json={}).status_code == 400
    res = client.post("/recognize-sign", json={"frames": ["a", "b"]})
    assert res.status_code == 200 and res.get_json()["predictions"][0]["sign"] == "computer"
    assert b"Sign a STEM word" in client.get("/signs").data


def test_redundancy_groups_only_signs_that_all_share_a_shape():
    from word_signs.redundancy import group, same_shape
    shared = np.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]], dtype=bool)
    groups = group(shared)
    assert sorted(map(sorted, groups)) == [[0, 1, 2], [3]]
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (8, 19))
    assert same_shape(a, rng.normal(0, 1, (8, 19)), rng)
    assert not same_shape(a, rng.normal(5, 1, (8, 19)), rng)
