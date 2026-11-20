"""Shared constants and helpers used by every method (no PyTorch here).

The three methods, the evaluation script and the app all import from this file,
so the class list, the image preprocessing and the metrics are identical for all of them.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = ROOT / "splits"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

SEED = 42
CACHE_SIZE = 256  # every image is stored as a 256x256 RGB JPEG after preprocessing

# ---- the task: 3 waste bins (RA 9003 segregation at source) -------------------------
CLASSES = ["biodegradable", "recyclable", "residual"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
DISPLAY = {
    "biodegradable": "Biodegradable (Nabubulok)",
    "recyclable": "Recyclable",
    "residual": "Non-biodegradable / Residual",
}
# RGB colours used when drawing results (green / blue / dark grey bins)
COLORS = {
    "biodegradable": (34, 160, 60),
    "recyclable": (30, 110, 230),
    "residual": (70, 70, 70),
}
ADVICE = {
    "biodegradable": "Food scraps, peelings, leaves, garden waste. Goes to the green bin / compost pit.",
    "recyclable": "Clean paper, cardboard, cans, glass and plastic bottles. Goes to the blue bin / junk shop.",
    "residual": "Not recyclable and not compostable: sachets, wrappers, styrofoam, old textiles. Goes to the black bin for disposal.",
}

METHOD_INFO = {
    "classical": {
        "slot": "A. Classical",
        "name": "HOG + HSV colour hist + LBP -> linear SVM",
        "short": "Classical (HOG+SVM)",
    },
    "scratch": {
        "slot": "B. Trained from scratch",
        "name": "WasteNet: 4-block CNN (ours)",
        "short": "Scratch CNN",
    },
    "pretrained": {
        "slot": "C. Pretrained, modern",
        "name": "MobileNetV3-Large (ImageNet) fine-tuned",
        "short": "MobileNetV3",
    },
}
METHOD_KEYS = list(METHOD_INFO)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---- image preprocessing (shared by dataset prep, training, evaluation and the app) --
def center_square(img: Image.Image) -> Image.Image:
    """Crop the largest centred square (the app draws this square as the guide box)."""
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def preprocess_pil(img: Image.Image, size: int = CACHE_SIZE) -> Image.Image:
    """RGB -> centre square crop -> size x size (bilinear, antialiased)."""
    img = img.convert("RGB")
    return center_square(img).resize((size, size), Image.BILINEAR)


def preprocess_array(arr: np.ndarray, size: int = CACHE_SIZE) -> np.ndarray:
    """Same as preprocess_pil but for an HxWx3 uint8 RGB array (webcam frames, uploads)."""
    return np.asarray(preprocess_pil(Image.fromarray(arr), size))


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def softmax(scores: np.ndarray) -> np.ndarray:
    z = scores - scores.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---- splits -----------------------------------------------------------------------
def load_split(name: str):
    """Return the DataFrame for 'train', 'val', 'test' or 'local_test'."""
    import pandas as pd

    path = SPLITS_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python prepare_data.py` first.")
    df = pd.read_csv(path)
    df["abs_path"] = [str(ROOT / p) for p in df["path"]]
    return df


def class_weights(labels: np.ndarray, n_classes: int = len(CLASSES)) -> np.ndarray:
    """Inverse-frequency weights (mean 1) used by all three methods to offset class imbalance."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    return (w / w.mean()).astype(np.float32)


# ---- metrics ----------------------------------------------------------------------
def classification_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(CLASSES)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred) * 100),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0) * 100),
        "per_class": {
            c: {"precision": float(p[i] * 100), "recall": float(r[i] * 100), "f1": float(f[i] * 100), "support": int(s[i])}
            for i, c in enumerate(CLASSES)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def time_per_image(predict_proba, images: list[np.ndarray], warmup: int = 5, sync=None) -> float:
    """Median wall-clock milliseconds to classify ONE preprocessed 256x256 image (batch size 1).

    Includes the method's own resize + feature extraction / forward pass, i.e. what the app pays per frame.
    """
    for im in images[:warmup]:
        predict_proba([im])
    if sync:
        sync()
    times = []
    for im in images:
        t0 = time.perf_counter()
        predict_proba([im])
        if sync:
            sync()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


# ---- small IO helpers ---------------------------------------------------------------
def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def load_json(path: str | Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text())


def plot_history(history: dict, title: str, ax_loss=None, ax_metric=None):
    """Training/validation curves (loss and accuracy/F1 vs epoch) with labelled axes."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    if ax_loss is None or ax_metric is None:
        fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(11, 3.8))
        fig.suptitle(title)
    ep = np.arange(1, len(history["train_loss"]) + 1)
    ax_loss.plot(ep, history["train_loss"], label="train")
    if "val_loss" in history:
        ax_loss.plot(ep, history["val_loss"], label="validation")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)
    if "train_acc" in history:
        ax_metric.plot(ep, history["train_acc"], label="train accuracy")
    ax_metric.plot(ep, history["val_acc"], label="val accuracy")
    if "val_f1" in history:
        ax_metric.plot(ep, history["val_f1"], label="val macro F1", linestyle="--")
    ax_metric.set_xlabel("epoch")
    ax_metric.set_ylabel("%")
    ax_metric.legend()
    ax_metric.grid(alpha=0.3)
    for ax in (ax_loss, ax_metric):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return ax_loss, ax_metric
