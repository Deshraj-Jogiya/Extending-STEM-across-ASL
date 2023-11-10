"""
asl_recognition/model.py
------------------------
TensorFlow/Keras CNN for ASL alphabet recognition (A–Z, 26 classes).

Architecture:
  3 × (Conv2D → BatchNorm → ReLU → MaxPool → Dropout)
  Flatten → Dense(256) → BatchNorm → ReLU → Dropout
  Dense(26, softmax)

Input:  64×64 greyscale images  (shape: 64, 64, 1)
Output: probability vector over 26 ASL letter classes
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ── Constants ────────────────────────────────────────────────────────────────
IMG_HEIGHT = 64
IMG_WIDTH  = 64
NUM_CLASSES = 26          # A–Z
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 1)


def build_asl_cnn(input_shape: tuple = INPUT_SHAPE,
                  num_classes: int = NUM_CLASSES,
                  dropout_rate_conv: float = 0.25,
                  dropout_rate_dense: float = 0.50) -> keras.Model:
    """
    Build and compile the ASL CNN model.

    Parameters
    ----------
    input_shape      : (H, W, C) tuple for the model input.
    num_classes      : Number of output classes (default 26 for A–Z).
    dropout_rate_conv: Dropout probability after each conv block.
    dropout_rate_dense: Dropout probability after the dense block.

    Returns
    -------
    Compiled keras.Model ready for training.
    """

    inputs = keras.Input(shape=input_shape, name="asl_input")

    # ── Block 1 ──────────────────────────────────────────────────────────────
    x = layers.Conv2D(32, kernel_size=(3, 3), padding="same",
                      use_bias=False, name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = layers.Dropout(dropout_rate_conv, name="drop1")(x)

    # ── Block 2 ──────────────────────────────────────────────────────────────
    x = layers.Conv2D(64, kernel_size=(3, 3), padding="same",
                      use_bias=False, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = layers.Dropout(dropout_rate_conv, name="drop2")(x)

    # ── Block 3 ──────────────────────────────────────────────────────────────
    x = layers.Conv2D(128, kernel_size=(3, 3), padding="same",
                      use_bias=False, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = layers.Dropout(dropout_rate_conv, name="drop3")(x)

    # ── Dense head ───────────────────────────────────────────────────────────
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(256, use_bias=False, name="dense1")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)
    x = layers.Dropout(dropout_rate_dense, name="drop4")(x)

    # ── Output ───────────────────────────────────────────────────────────────
    outputs = layers.Dense(num_classes, activation="softmax",
                           name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ASL_CNN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_label_map() -> dict:
    """Return a mapping from class index to ASL letter."""
    return {i: chr(ord("A") + i) for i in range(NUM_CLASSES)}


def get_reverse_label_map() -> dict:
    """Return a mapping from ASL letter to class index."""
    return {chr(ord("A") + i): i for i in range(NUM_CLASSES)}


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  ASL CNN Model – Architecture Summary")
    print("=" * 60)

    model = build_asl_cnn()
    model.summary()

    print()
    print(f"Input shape  : {INPUT_SHAPE}")
    print(f"Output classes: {NUM_CLASSES}  (A–Z)")
    print()

    # Sanity-check: forward pass with a random batch
    import numpy as np
    dummy_batch = np.random.rand(4, IMG_HEIGHT, IMG_WIDTH, 1).astype("float32")
    preds = model.predict(dummy_batch, verbose=0)
    print(f"Batch prediction shape : {preds.shape}")   # (4, 26)
    print(f"Sum of first row probs : {preds[0].sum():.6f}")  # ≈ 1.0

    label_map = get_label_map()
    pred_letters = [label_map[int(p.argmax())] for p in preds]
    print(f"Predicted letters (random input): {pred_letters}")
    print()
    print("✅  Model compiled and forward pass verified successfully.")
