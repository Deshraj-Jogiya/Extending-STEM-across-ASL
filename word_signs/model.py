"""Recognise a word sign from its sequence of hand features (see features.py).

SignGRU: a 1-D convolution over time picks up short motion patterns, a bidirectional GRU reads the whole sign, and the mean of its
outputs goes to a linear classifier. Training augments each clip by mirroring it (a left-handed signer), shifting it by a couple of
frames and adding small noise, since every sign has only a handful of example clips.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from word_signs.features import FEATURES_PER_FRAME, LANDMARKS, PER_HAND


class SignGRU(nn.Module):
    def __init__(self, n_classes: int, hidden: int = 96, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(FEATURES_PER_FRAME, hidden, kernel_size=3, padding=1), nn.BatchNorm1d(hidden), nn.ReLU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, frames, features)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.gru(h)
        return self.head(out.mean(dim=1))


def mirror(x: np.ndarray) -> np.ndarray:
    """Swap the right and left hand slots and flip x, as if the sign were made with the other hand."""
    out = np.concatenate([x[..., PER_HAND:], x[..., :PER_HAND]], axis=-1).copy()
    for slot in (0, 1):
        start = slot * PER_HAND
        out[..., start:start + LANDMARKS * 3:3] *= -1
    return out


def augment(batch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = batch.copy()
    for i in range(len(out)):
        if rng.random() < 0.5:
            out[i] = mirror(out[i])
        out[i] = np.roll(out[i], rng.integers(-2, 3), axis=0)
    return out + rng.normal(0, 0.02, out.shape).astype(np.float32)


def split(y: np.ndarray, test_per_class: int = 2, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Stratified hold-out: `test_per_class` clips of every sign go to the test set (WLASL gives no signer ids to split on)."""
    rng = np.random.default_rng(seed)
    test = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        test.extend(rng.choice(idx, size=min(test_per_class, len(idx) - 1), replace=False))
    test = np.sort(np.array(test))
    return np.setdiff1d(np.arange(len(y)), test), test


def train(X: np.ndarray, y: np.ndarray, n_classes: int, epochs: int = 120, batch_size: int = 16, lr: float = 2e-3,
          seed: int = 42) -> SignGRU:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SignGRU(n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(X))
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            if len(idx) < 2:
                continue
            xb = torch.from_numpy(augment(X[idx], rng))
            opt.zero_grad()
            loss_fn(model(xb), torch.from_numpy(y[idx]).long()).backward()
            opt.step()
        sched.step()
    return model.eval()


@torch.no_grad()
def predict_proba(model: SignGRU, X: np.ndarray) -> np.ndarray:
    model.eval()
    return torch.softmax(model(torch.from_numpy(np.asarray(X, dtype=np.float32))), dim=1).numpy()


def save(model: SignGRU, vocabulary: list[str], path: str) -> None:
    torch.save({"state_dict": model.state_dict(), "vocabulary": list(vocabulary)}, path)


def load(path: str) -> tuple[SignGRU, list[str]]:
    ckpt = torch.load(path, map_location="cpu")
    model = SignGRU(len(ckpt["vocabulary"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt["vocabulary"]

