"""
asl_recognition/dataset.py
--------------------------
Synthetic image dataset generator for ASL alphabet recognition.

Generates 64×64 greyscale mock images for all 26 ASL letter classes (A–Z).
Each image is a noise pattern with a class-specific bias so the model has
*something* to learn from even on synthetic data.

Pipeline
--------
1. generate_synthetic_dataset()  → raw numpy arrays  (X, y)
2. split_dataset()               → train / val / test splits
3. normalize_images()            → scale pixel values to [0, 1]
4. get_dataloaders()             → returns all three splits, normalized

No external datasets or downloads are required.
"""

import numpy as np
from sklearn.model_selection import train_test_split

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_HEIGHT  = 64
IMG_WIDTH   = 64
NUM_CLASSES = 26        # A–Z
SEED        = 42


def generate_synthetic_dataset(
    samples_per_class: int = 200,
    img_h: int = IMG_HEIGHT,
    img_w: int = IMG_WIDTH,
    num_classes: int = NUM_CLASSES,
    noise_std: float = 40.0,
    seed: int = SEED,
) -> tuple:
    """
    Generate a synthetic ASL image dataset.

    Each class gets `samples_per_class` images.  Images are 64×64 greyscale
    uint8 arrays.  The mean pixel intensity is class-dependent (class_idx *
    (255 / num_classes)), which gives the CNN a learnable signal.

    Parameters
    ----------
    samples_per_class : Number of images to generate per ASL class.
    img_h, img_w      : Height and width of each image in pixels.
    num_classes       : Number of letter classes (26 for A–Z).
    noise_std         : Standard deviation of the Gaussian noise overlay.
    seed              : Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray, shape (N, img_h, img_w, 1), dtype float32, values [0,255]
    y : np.ndarray, shape (N,), dtype int,    class indices 0–25
    """
    rng = np.random.default_rng(seed)
    total_samples = samples_per_class * num_classes

    X = np.zeros((total_samples, img_h, img_w, 1), dtype=np.float32)
    y = np.zeros(total_samples, dtype=np.int32)

    for class_idx in range(num_classes):
        start = class_idx * samples_per_class
        end   = start + samples_per_class

        # Class-specific base intensity gradient (gives discriminative signal)
        base_intensity = class_idx * (255.0 / num_classes)

        # Horizontal gradient across the image (unique per class)
        gradient = np.linspace(
            base_intensity * 0.5,
            base_intensity * 1.5,
            img_w,
            dtype=np.float32,
        ).reshape(1, img_w)

        # Vertical sinusoidal pattern (adds texture per class)
        freq = (class_idx + 1) * 0.3
        vertical = (np.sin(np.linspace(0, freq * np.pi, img_h)) * 30 + 128).reshape(img_h, 1)

        pattern = (gradient + vertical) / 2.0  # shape: (img_h, img_w)

        # Tile and add noise for each sample
        noise = rng.normal(0, noise_std, (samples_per_class, img_h, img_w)).astype(np.float32)
        images = pattern[np.newaxis, :, :] + noise          # (samples, H, W)
        images = np.clip(images, 0, 255)
        images = images[:, :, :, np.newaxis]                # (samples, H, W, 1)

        X[start:end] = images
        y[start:end] = class_idx

    return X, y


def normalize_images(X: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values from [0, 255] to [0.0, 1.0].

    Parameters
    ----------
    X : np.ndarray of pixel values.

    Returns
    -------
    np.ndarray of the same shape with values in [0.0, 1.0], dtype float32.
    """
    return (X / 255.0).astype(np.float32)


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = SEED,
) -> tuple:
    """
    Split dataset into train / validation / test partitions.

    Parameters
    ----------
    X         : Feature array.
    y         : Label array.
    val_size  : Fraction of total data to use for validation.
    test_size : Fraction of total data to use for testing.
    seed      : Random seed for reproducibility.

    Returns
    -------
    (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    # Then split the remaining data into train / val
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=relative_val,
        random_state=seed,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def to_categorical(y: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """One-hot encode label array."""
    from tensorflow.keras.utils import to_categorical as tf_to_categorical
    return tf_to_categorical(y, num_classes=num_classes)


def get_dataloaders(
    samples_per_class: int = 200,
    num_classes: int = NUM_CLASSES,
    seed: int = SEED,
) -> dict:
    """
    Full pipeline: generate → split → normalize → one-hot encode.

    Returns
    -------
    dict with keys:
        'X_train', 'X_val', 'X_test'  : float32 arrays, values in [0,1]
        'y_train', 'y_val', 'y_test'  : one-hot encoded int32 arrays
        'num_classes'                  : int
        'label_map'                    : dict {index: letter}
    """
    print(f"[dataset] Generating synthetic dataset ({samples_per_class} samples/class)…")
    X, y = generate_synthetic_dataset(samples_per_class=samples_per_class,
                                      num_classes=num_classes, seed=seed)

    print(f"[dataset] Total samples: {len(X)}  |  Classes: {num_classes}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y, seed=seed)
    print(f"[dataset] Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    X_train = normalize_images(X_train)
    X_val   = normalize_images(X_val)
    X_test  = normalize_images(X_test)

    y_train_oh = to_categorical(y_train, num_classes)
    y_val_oh   = to_categorical(y_val,   num_classes)
    y_test_oh  = to_categorical(y_test,  num_classes)

    label_map = {i: chr(ord("A") + i) for i in range(num_classes)}

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train_oh, "y_val": y_val_oh, "y_test": y_test_oh,
        "num_classes": num_classes,
        "label_map": label_map,
    }


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  ASL Dataset Generator – Self-test")
    print("=" * 60)

    data = get_dataloaders(samples_per_class=50)  # small for quick test

    print()
    print(f"X_train shape : {data['X_train'].shape}")
    print(f"X_val   shape : {data['X_val'].shape}")
    print(f"X_test  shape : {data['X_test'].shape}")
    print(f"y_train shape : {data['y_train'].shape}")
    print(f"Pixel range   : [{data['X_train'].min():.3f}, {data['X_train'].max():.3f}]")
    print(f"Label map     : {data['label_map']}")
    print()
    print("✅  Dataset generator verified successfully.")
