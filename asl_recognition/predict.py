"""
asl_recognition/predict.py
--------------------------
Inference engine for ASL letter prediction.

Workflow
--------
1. Load the saved Keras model (asl_model.h5).
   If the model file does not exist, fall back to building an untrained model.
2. Accept an image path or generate a mock 64×64 greyscale input.
3. Pre-process (resize if needed, normalize to [0,1]).
4. Run a forward pass and return (predicted_letter, confidence_score).

Usage
-----
    # Default — runs a mock prediction
    python asl_recognition/predict.py

    # Predict from a real image file
    python asl_recognition/predict.py --image /path/to/hand.png
"""

import os
import sys
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from asl_recognition.model import build_asl_cnn, get_label_map, IMG_HEIGHT, IMG_WIDTH

# Default model path (written by train.py)
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "asl_model.h5")


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """
    Load the trained Keras model.

    If the .h5 file does not exist, build an untrained model and warn the user.

    Parameters
    ----------
    model_path : Absolute path to the saved model file.

    Returns
    -------
    keras.Model
    """
    import tensorflow as tf

    if os.path.exists(model_path):
        print(f"[predict] Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
    else:
        print(f"[predict] WARNING: Model not found at '{model_path}'.")
        print("[predict] Building an untrained model for demonstration purposes.")
        model = build_asl_cnn()

    return model


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk, convert to greyscale, resize to 64×64,
    normalise to [0, 1], and add batch + channel dimensions.

    Parameters
    ----------
    image_path : Path to the input image file.

    Returns
    -------
    np.ndarray of shape (1, 64, 64, 1), dtype float32.
    """
    from PIL import Image

    img = Image.open(image_path).convert("L")               # greyscale
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0           # [0, 1]
    arr = arr[np.newaxis, :, :, np.newaxis]                  # (1, H, W, 1)
    return arr


def generate_mock_input(seed: int = 0) -> np.ndarray:
    """
    Generate a random 64×64 greyscale image as mock ASL input.

    Returns
    -------
    np.ndarray of shape (1, 64, 64, 1), dtype float32.
    """
    rng = np.random.default_rng(seed)
    arr = rng.random((1, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
    return arr.astype(np.float32)


def predict(image: np.ndarray, model, label_map: dict) -> tuple:
    """
    Run inference on a pre-processed image.

    Parameters
    ----------
    image     : np.ndarray of shape (1, H, W, 1), values in [0, 1].
    model     : Loaded keras.Model.
    label_map : Dict mapping class index → letter string.

    Returns
    -------
    (predicted_letter: str, confidence: float)
    """
    probs = model.predict(image, verbose=0)[0]   # shape: (26,)
    class_idx   = int(np.argmax(probs))
    confidence  = float(probs[class_idx])
    letter      = label_map[class_idx]
    return letter, confidence


def predict_from_path(image_path: str,
                      model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """
    High-level helper: load model + image, run prediction.

    Parameters
    ----------
    image_path : Path to the input image.
    model_path : Path to the saved model (.h5).

    Returns
    -------
    dict with keys: 'letter', 'confidence', 'class_index', 'all_probs'
    """
    import tensorflow as tf

    model     = load_model(model_path)
    label_map = get_label_map()
    image     = preprocess_image(image_path)

    probs      = model.predict(image, verbose=0)[0]
    class_idx  = int(np.argmax(probs))
    confidence = float(probs[class_idx])
    letter     = label_map[class_idx]

    return {
        "letter":      letter,
        "confidence":  confidence,
        "class_index": class_idx,
        "all_probs":   probs.tolist(),
    }


def predict_mock(model_path: str = DEFAULT_MODEL_PATH, seed: int = 42) -> dict:
    """
    Run a mock prediction using a randomly generated image.

    Returns
    -------
    dict with keys: 'letter', 'confidence', 'class_index', 'all_probs'
    """
    model     = load_model(model_path)
    label_map = get_label_map()
    image     = generate_mock_input(seed=seed)

    probs      = model.predict(image, verbose=0)[0]
    class_idx  = int(np.argmax(probs))
    confidence = float(probs[class_idx])
    letter     = label_map[class_idx]

    return {
        "letter":      letter,
        "confidence":  confidence,
        "class_index": class_idx,
        "all_probs":   probs.tolist(),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="ASL letter prediction engine",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an input image file (optional; mock input used if omitted).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the saved Keras model. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for mock input generation (default: 42).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("  ASL Inference Engine")
    print("=" * 60)

    if args.image:
        print(f"[predict] Input image  : {args.image}")
        result = predict_from_path(args.image, model_path=args.model)
    else:
        print("[predict] No image provided — using mock (random) input.")
        result = predict_mock(model_path=args.model, seed=args.seed)

    print()
    print(f"  Predicted Letter : {result['letter']}")
    print(f"  Confidence Score : {result['confidence']:.4f}  ({result['confidence']*100:.2f}%)")
    print(f"  Class Index      : {result['class_index']}")
    print()
    print("  Top-5 Predictions:")
    label_map = get_label_map()
    probs_arr  = np.array(result["all_probs"])
    top5_idx   = np.argsort(probs_arr)[::-1][:5]
    for rank, idx in enumerate(top5_idx, 1):
        print(f"    {rank}. {label_map[idx]}  →  {probs_arr[idx]*100:.2f}%")
    print()
    print("✅  Prediction complete.")
