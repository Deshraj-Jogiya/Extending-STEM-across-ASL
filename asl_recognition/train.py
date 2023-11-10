"""
asl_recognition/train.py
------------------------
Full training script for the ASL CNN.

Steps
-----
1. Load synthetic dataset via dataset.get_dataloaders()
2. Build the CNN via model.build_asl_cnn()
3. Train with EarlyStopping (patience=5, restore_best_weights=True)
4. Evaluate on the test set
5. Save model as  asl_model.h5  (in the project root)
6. Plot and save training curves as  asl_recognition/training_curves.png

Usage
-----
    python asl_recognition/train.py
"""

import os
import sys

# Allow imports from the project root regardless of where the script is run
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (works without a display)
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from asl_recognition.dataset import get_dataloaders
from asl_recognition.model  import build_asl_cnn

# ── Hyper-parameters ──────────────────────────────────────────────────────────
BATCH_SIZE       = 32
MAX_EPOCHS       = 50
PATIENCE         = 5
SAMPLES_PER_CLASS = 200    # increase for better accuracy; 200 keeps training fast
OUTPUT_DIR       = os.path.join(PROJECT_ROOT, "asl_recognition")
MODEL_PATH       = os.path.join(PROJECT_ROOT, "asl_model.h5")
CURVE_PATH       = os.path.join(OUTPUT_DIR,   "training_curves.png")


def plot_training_curves(history, save_path: str) -> None:
    """Plot and save accuracy + loss curves from training history."""
    epochs = range(1, len(history.history["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("ASL CNN – Training Curves", fontsize=14, fontweight="bold")

    # Accuracy
    axes[0].plot(epochs, history.history["accuracy"],     label="Train Acc",  color="royalblue")
    axes[0].plot(epochs, history.history["val_accuracy"], label="Val Acc",    color="darkorange", linestyle="--")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(epochs, history.history["loss"],     label="Train Loss", color="royalblue")
    axes[1].plot(epochs, history.history["val_loss"], label="Val Loss",   color="darkorange", linestyle="--")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Categorical Crossentropy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[train] Training curves saved to: {save_path}")


def main():
    print("=" * 60)
    print("  Extending STEM across ASL — CNN Training")
    print("=" * 60)
    print(f"TensorFlow version : {tf.__version__}")
    print(f"GPU available      : {bool(tf.config.list_physical_devices('GPU'))}")
    print()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    data = get_dataloaders(samples_per_class=SAMPLES_PER_CLASS)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    X_test,  y_test  = data["X_test"],  data["y_test"]
    num_classes      = data["num_classes"]

    print(f"\n[train] Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # ── 2. Build model ────────────────────────────────────────────────────────
    model = build_asl_cnn(num_classes=num_classes)
    print("\n[train] Model built successfully.")

    # ── 3. Callbacks ──────────────────────────────────────────────────────────
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ── 4. Train ──────────────────────────────────────────────────────────────
    print(f"\n[train] Starting training (max {MAX_EPOCHS} epochs, batch={BATCH_SIZE})…\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    print("\n[train] Evaluating on test set…")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[train] Test Loss     : {test_loss:.4f}")
    print(f"[train] Test Accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")

    # ── 6. Save model ─────────────────────────────────────────────────────────
    model.save(MODEL_PATH)
    print(f"\n[train] Model saved to: {MODEL_PATH}")

    # ── 7. Plot curves ────────────────────────────────────────────────────────
    plot_training_curves(history, CURVE_PATH)

    print("\n" + "=" * 60)
    print("  Training complete! ✅")
    print(f"  Model  : {MODEL_PATH}")
    print(f"  Curves : {CURVE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
