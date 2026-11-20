"""Method A - Classical: hand-designed features + linear SVM, written from scratch in NumPy.

Features (all implemented below with NumPy only, no OpenCV / scikit-image calls):
  * HOG   - histogram of oriented gradients (Dalal & Triggs 2005): shape / edges
  * HSV   - joint 8x4x4 colour histogram: colour (food = brown/green, cardboard = tan, ...)
  * LBP   - uniform local binary patterns (Ojala et al. 2002): surface texture
Classifier: multi-class linear SVM (Weston-Watkins hinge loss + L2), trained with mini-batch SGD + momentum.

    python classical.py sanity     # sanity checks on synthetic images (required by the rubric)
    python classical.py train      # extract features, grid-search C on val, save models/classical.npz
"""
from __future__ import annotations

import argparse
import colorsys
import time

import numpy as np
from PIL import Image
from tqdm import tqdm

from common import (CLASSES, MODELS_DIR, PROCESSED_DIR, SEED, class_weights, classification_metrics, load_rgb,
                    load_split, save_json, softmax)

FEAT_SIZE = 128  # images are resized to 128x128 before feature extraction
HOG_CELL, HOG_BLOCK, HOG_BINS = 16, 2, 9
HSV_BINS = (8, 4, 4)
MODEL_PATH = MODELS_DIR / "classical.npz"


# =============================================================================================
# Feature extraction (NumPy)
# =============================================================================================
def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 luma, float32 in [0, 255]."""
    img = img.astype(np.float32)
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    """uint8 RGB -> float HSV, every channel in [0, 1] (same convention as colorsys)."""
    rgb = img.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    c = mx - mn
    safe_c = np.where(c == 0, 1.0, c)
    h = np.where(mx == r, ((g - b) / safe_c) % 6.0, np.where(mx == g, (b - r) / safe_c + 2.0, (r - g) / safe_c + 4.0))
    h = np.where(c == 0, 0.0, h) / 6.0
    s = np.where(mx == 0, 0.0, c / np.where(mx == 0, 1.0, mx))
    return np.stack([h, s, mx], axis=-1)


def hog_cell_histograms(gray: np.ndarray, cell: int = HOG_CELL, nbins: int = HOG_BINS) -> np.ndarray:
    """Per-cell histograms of unsigned gradient orientation (0-180 deg), magnitude-weighted,
    with linear interpolation between the two nearest orientation bins. Returns (ncy, ncx, nbins)."""
    g = gray.astype(np.float32)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]  # centred difference [-1, 0, 1]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    mag = np.hypot(gx, gy)
    ang = np.rad2deg(np.arctan2(gy, gx)) % 180.0

    ncy, ncx = g.shape[0] // cell, g.shape[1] // cell
    mag, ang = mag[: ncy * cell, : ncx * cell], ang[: ncy * cell, : ncx * cell]
    pos = ang / (180.0 / nbins) - 0.5  # bin centres at 10, 30, ..., 170 deg
    lo = np.floor(pos).astype(np.int64)
    w_hi = (pos - lo).astype(np.float32)
    lo_idx, hi_idx = lo % nbins, (lo + 1) % nbins

    hist = np.empty((ncy, ncx, nbins), np.float32)
    for k in range(nbins):
        contrib = mag * ((lo_idx == k) * (1.0 - w_hi) + (hi_idx == k) * w_hi)
        hist[..., k] = contrib.reshape(ncy, cell, ncx, cell).sum(axis=(1, 3))
    return hist


def hog(gray: np.ndarray, cell: int = HOG_CELL, block: int = HOG_BLOCK, nbins: int = HOG_BINS) -> np.ndarray:
    """HOG descriptor: overlapping block x block cell groups, L2-Hys normalised, concatenated."""
    hist = hog_cell_histograms(gray, cell, nbins)
    blocks = np.lib.stride_tricks.sliding_window_view(hist, (block, block), axis=(0, 1))  # (by, bx, nbins, b, b)
    blocks = blocks.reshape(blocks.shape[0], blocks.shape[1], -1)
    eps = 1e-5
    blocks = blocks / np.sqrt((blocks**2).sum(-1, keepdims=True) + eps**2)
    blocks = np.minimum(blocks, 0.2)  # L2-Hys clipping
    blocks = blocks / np.sqrt((blocks**2).sum(-1, keepdims=True) + eps**2)
    return blocks.ravel().astype(np.float32)


def hsv_histogram(img: np.ndarray, bins=HSV_BINS) -> np.ndarray:
    hsv = rgb_to_hsv(img)
    idx = [np.minimum((hsv[..., i] * bins[i]).astype(np.int64), bins[i] - 1) for i in range(3)]
    flat = (idx[0] * bins[1] + idx[1]) * bins[2] + idx[2]
    hist = np.bincount(flat.ravel(), minlength=int(np.prod(bins))).astype(np.float32)
    return np.sqrt(hist / hist.sum())  # Hellinger (square-root) normalisation


def _uniform_lbp_table() -> np.ndarray:
    """Map each 8-bit LBP code to one of 58 'uniform' patterns (<= 2 bit transitions) or bin 58."""
    table = np.full(256, 58, np.int64)
    nxt = 0
    for code in range(256):
        bits = [(code >> i) & 1 for i in range(8)]
        transitions = sum(bits[i] != bits[(i + 1) % 8] for i in range(8))
        if transitions <= 2:
            table[code] = nxt
            nxt += 1
    assert nxt == 58
    return table


LBP_TABLE = _uniform_lbp_table()
# 8 neighbours at radius 1, in circular order
LBP_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def lbp_codes(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    c = g[1:-1, 1:-1]
    code = np.zeros(c.shape, np.int64)
    H, W = g.shape
    for bit, (dy, dx) in enumerate(LBP_OFFSETS):
        nb = g[1 + dy : H - 1 + dy, 1 + dx : W - 1 + dx]
        code |= (nb >= c).astype(np.int64) << bit
    return code


def lbp_histogram(gray: np.ndarray) -> np.ndarray:
    hist = np.bincount(LBP_TABLE[lbp_codes(gray)].ravel(), minlength=59).astype(np.float32)
    return np.sqrt(hist / hist.sum())


def extract_features(img: np.ndarray, parts=("hog", "hsv", "lbp")) -> np.ndarray:
    """uint8 RGB image (any size) -> 1D float32 feature vector."""
    if img.shape[0] != FEAT_SIZE or img.shape[1] != FEAT_SIZE:
        img = np.asarray(Image.fromarray(img).resize((FEAT_SIZE, FEAT_SIZE), Image.BILINEAR))
    gray = rgb_to_gray(img)
    feats = []
    if "hog" in parts:
        feats.append(hog(gray))
    if "hsv" in parts:
        feats.append(hsv_histogram(img))
    if "lbp" in parts:
        feats.append(lbp_histogram(gray))
    return np.concatenate(feats).astype(np.float32)


FEATURE_SLICES = {}  # filled lazily: name -> slice into the full feature vector


def feature_slices() -> dict[str, slice]:
    if not FEATURE_SLICES:
        dummy = np.zeros((FEAT_SIZE, FEAT_SIZE, 3), np.uint8)
        start = 0
        for part in ("hog", "hsv", "lbp"):
            n = extract_features(dummy, (part,)).size
            FEATURE_SLICES[part] = slice(start, start + n)
            start += n
    return FEATURE_SLICES


def extract_split(split: str, use_cache: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Features for every image of a split, cached in data/processed/features_<split>.npz."""
    df = load_split(split)
    cache = PROCESSED_DIR / f"features_{split}.npz"
    if use_cache and cache.exists():
        z = np.load(cache, allow_pickle=False)
        if list(z["paths"]) == list(df["path"]):
            return z["X"], z["y"]
    X = np.stack([extract_features(load_rgb(p)) for p in tqdm(df["abs_path"], desc=f"features [{split}]")])
    y = df["label_idx"].to_numpy()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y, paths=df["path"].to_numpy().astype(str))
    return X, y


# =============================================================================================
# Linear SVM (NumPy)
# =============================================================================================
class LinearSVM:
    """Multi-class linear SVM: minimise  mean_i w_i * sum_{j != y_i} max(0, s_j - s_{y_i} + 1) + reg * ||W||^2
    with s = xW + b, by mini-batch SGD with momentum and a cosine learning-rate schedule."""

    def __init__(self, n_classes=len(CLASSES), reg=1e-3, lr=0.01, epochs=60, batch_size=256, momentum=0.9, seed=SEED):
        self.n_classes, self.reg, self.lr, self.epochs = n_classes, reg, lr, epochs
        self.batch_size, self.momentum, self.seed = batch_size, momentum, seed
        self.W = self.b = self.mean = self.std = None

    # --- standardisation (statistics from the training set only) ---
    def _fit_scaler(self, X):
        self.mean = X.mean(0)
        self.std = X.std(0) + 1e-6

    def _scale(self, X):
        return (X - self.mean) / self.std

    def loss_and_grad(self, X, y, sw):
        n = X.shape[0]
        rows = np.arange(n)
        scores = X @ self.W + self.b
        margins = np.maximum(0.0, scores - scores[rows, y][:, None] + 1.0)
        margins[rows, y] = 0.0
        sw = sw / sw.sum()
        loss = float((sw * margins.sum(1)).sum() + self.reg * (self.W**2).sum())
        coef = (margins > 0).astype(np.float32)
        coef[rows, y] = -coef.sum(1)
        coef *= sw[:, None]
        dW = X.T @ coef + 2 * self.reg * self.W
        db = coef.sum(0)
        return loss, dW, db

    def decision_function(self, X):
        return self._scale(X) @ self.W + self.b

    def predict(self, X):
        return self.decision_function(X).argmax(1)

    def fit(self, X, y, X_val=None, y_val=None, weights=None, verbose=False):
        rng = np.random.default_rng(self.seed)
        self._fit_scaler(X)
        Xs = self._scale(X).astype(np.float32)
        d = Xs.shape[1]
        self.W = (0.001 * rng.standard_normal((d, self.n_classes))).astype(np.float32)
        self.b = np.zeros(self.n_classes, np.float32)
        vW, vb = np.zeros_like(self.W), np.zeros_like(self.b)
        cw = weights if weights is not None else np.ones(self.n_classes, np.float32)
        sample_w = cw[y].astype(np.float32)
        hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
        if X_val is not None:
            Xv_s, sw_val = self._scale(X_val).astype(np.float32), cw[y_val].astype(np.float32)
        best = (-1.0, None)
        self.best_epoch = self.epochs
        steps_per_epoch = int(np.ceil(len(y) / self.batch_size))
        total = self.epochs * steps_per_epoch
        step = 0
        for ep in range(self.epochs):
            order = rng.permutation(len(y))
            losses = []
            for i in range(0, len(y), self.batch_size):
                idx = order[i : i + self.batch_size]
                lr = 0.5 * self.lr * (1 + np.cos(np.pi * step / total))
                loss, dW, db = self.loss_and_grad(Xs[idx], y[idx], sample_w[idx])
                vW = self.momentum * vW - lr * dW
                vb = self.momentum * vb - lr * db
                self.W += vW
                self.b += vb
                losses.append(loss)
                step += 1
            hist["train_loss"].append(float(np.mean(losses)))
            hist["train_acc"].append(float((self.predict(X) == y).mean() * 100))
            if X_val is not None:
                m = classification_metrics(y_val, self.predict(X_val))
                hist["val_loss"].append(self.loss_and_grad(Xv_s, y_val, sw_val)[0])
                hist["val_acc"].append(m["accuracy"])
                hist["val_f1"].append(m["macro_f1"])
                if m["macro_f1"] > best[0]:
                    best = (m["macro_f1"], (self.W.copy(), self.b.copy(), ep + 1))
            if verbose:
                print(f"  epoch {ep + 1:3d} loss {hist['train_loss'][-1]:.4f} train acc {hist['train_acc'][-1]:.1f}"
                      + (f" val F1 {hist['val_f1'][-1]:.1f}" if X_val is not None else ""))
        if best[1] is not None:  # keep the epoch with the best validation macro-F1 (early stopping)
            self.W, self.b, self.best_epoch = best[1]
        if X_val is None:
            del hist["val_loss"], hist["val_acc"], hist["val_f1"]
        return hist

    def save(self, path, **meta):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, W=self.W, b=self.b, mean=self.mean, std=self.std, reg=self.reg, lr=self.lr)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        m = cls(n_classes=z["W"].shape[1], reg=float(z["reg"]), lr=float(z["lr"]))
        m.W, m.b, m.mean, m.std = z["W"], z["b"], z["mean"], z["std"]
        return m


# =============================================================================================
# Training / prediction
# =============================================================================================
def train(regs=(1e-4, 1e-3, 1e-2, 1e-1, 1.0), lrs=(0.003, 0.01), epochs=60, parts=("hog", "hsv", "lbp"), verbose=True,
          save=True) -> dict:
    """Grid-search the regularisation strength and learning rate on the validation split."""
    parts = tuple(parts)
    t0 = time.perf_counter()
    X_tr, y_tr = extract_split("train")
    X_va, y_va = extract_split("val")
    t_feat = time.perf_counter() - t0
    cols = np.concatenate([np.arange(feature_slices()[p].start, feature_slices()[p].stop) for p in parts])
    X_tr, X_va = X_tr[:, cols], X_va[:, cols]
    cw = class_weights(y_tr)

    t1 = time.perf_counter()
    grid, best = [], None
    for reg in regs:
        for lr in lrs:
            svm = LinearSVM(reg=reg, lr=lr, epochs=epochs)
            hist = svm.fit(X_tr, y_tr, X_va, y_va, weights=cw)
            f1 = max(hist["val_f1"])
            grid.append({"reg": reg, "lr": lr, "val_macro_f1": f1, "best_epoch": svm.best_epoch})
            if verbose:
                print(f"reg={reg:g} lr={lr:g}: best val macro-F1 {f1:.2f} (epoch {svm.best_epoch})")
            if best is None or f1 > best[0]:
                best = (f1, svm, hist)
    t_svm = time.perf_counter() - t1
    f1, svm, hist = best
    if not save:
        return {"features": "+".join(parts), "val_macro_f1": f1}

    svm.save(MODEL_PATH)
    val_metrics = classification_metrics(y_va, svm.predict(X_va))
    meta = {
        "method": "classical",
        "features": {p: int(feature_slices()[p].stop - feature_slices()[p].start) for p in parts},
        "feature_dim": int(X_tr.shape[1]),
        "image_size": FEAT_SIZE,
        "chosen": {"reg": svm.reg, "lr": svm.lr, "epochs": epochs, "best_epoch": svm.best_epoch},
        "grid": grid,
        "val": val_metrics,
        "params_millions": 0.0,  # rubric: report 0 for classical methods
        "svm_weights": int(svm.W.size + svm.b.size),
        "feature_extraction_min": t_feat / 60,
        "train_time_min": (t_feat + t_svm) / 60,
        "history": hist,
    }
    save_json(meta, MODELS_DIR / "classical_meta.json")
    print(f"\nsaved {MODEL_PATH}  | val acc {val_metrics['accuracy']:.2f}%  macro-F1 {val_metrics['macro_f1']:.2f}%"
          f"  | train time {meta['train_time_min']:.2f} min")
    return meta


def feature_ablation(epochs=40) -> list[dict]:
    """Which hand-designed feature matters? Train the SVM on each feature group alone (val macro-F1)."""
    rows = []
    for parts in [("hog",), ("hsv",), ("lbp",), ("hsv", "lbp"), ("hog", "hsv", "lbp")]:
        rows.append(train(regs=(1e-3,), lrs=(0.01,), epochs=epochs, parts=parts, verbose=False, save=False))
        print(f"  {rows[-1]['features']:<12} val macro-F1 {rows[-1]['val_macro_f1']:.2f}%")
    save_json(rows, MODELS_DIR / "classical_ablation.json")
    return rows


def visualize_features(img: np.ndarray):
    """Figure for the report: input, gradient magnitude, HOG cell glyphs, LBP codes and HSV histogram."""
    import matplotlib.pyplot as plt

    im = np.asarray(Image.fromarray(img).resize((FEAT_SIZE, FEAT_SIZE), Image.BILINEAR))
    gray = rgb_to_gray(im)
    gx, gy = np.zeros_like(gray), np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    hist = hog_cell_histograms(gray)
    fig, ax = plt.subplots(1, 5, figsize=(20, 4))
    ax[0].imshow(im)
    ax[0].set_title(f"input ({FEAT_SIZE}x{FEAT_SIZE})")
    ax[1].imshow(np.hypot(gx, gy), cmap="gray")
    ax[1].set_title("gradient magnitude")
    ax[2].set_facecolor("black")
    ncy, ncx, nb = hist.shape
    scale = hist.max() + 1e-6
    for cy in range(ncy):
        for cx in range(ncx):
            for k in range(nb):
                theta = np.deg2rad((k + 0.5) * 180 / nb + 90)  # draw the edge direction (perpendicular to gradient)
                L = 0.5 * HOG_CELL * hist[cy, cx, k] / scale
                x0, y0 = (cx + 0.5) * HOG_CELL, (cy + 0.5) * HOG_CELL
                ax[2].plot([x0 - L * np.cos(theta), x0 + L * np.cos(theta)], [y0 - L * np.sin(theta), y0 + L * np.sin(theta)],
                           color="white", linewidth=1)
    ax[2].set_xlim(0, FEAT_SIZE)
    ax[2].set_ylim(FEAT_SIZE, 0)
    ax[2].set_aspect("equal")
    ax[2].set_title(f"HOG cells ({HOG_CELL}x{HOG_CELL} px, {HOG_BINS} bins)")
    ax[3].imshow(LBP_TABLE[lbp_codes(gray)], cmap="nipy_spectral")
    ax[3].set_title("uniform LBP code per pixel")
    h = hsv_histogram(im) ** 2
    ax[4].bar(np.arange(h.size), h, width=1.0)
    ax[4].set_title("joint HSV histogram (8x4x4 bins)")
    ax[4].set_xlabel("bin index (hue-major)")
    ax[4].set_ylabel("fraction of pixels")
    for a in ax[:4]:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    return fig


class ClassicalPredictor:
    key = "classical"

    def __init__(self, path=MODEL_PATH):
        self.svm = LinearSVM.load(path)
        self.n_params = 0
        self.device = "cpu"

    def to(self, device):  # API parity with the torch predictors; NumPy always runs on the CPU
        return self

    def predict_proba(self, images: list[np.ndarray]) -> np.ndarray:
        """SVM scores turned into softmax 'confidences' for display (not calibrated probabilities)."""
        X = np.stack([extract_features(im) for im in images])
        return softmax(self.svm.decision_function(X))


# =============================================================================================
# Sanity checks on synthetic data
# =============================================================================================
def _stripes(direction: str, size=FEAT_SIZE, period=16, color=(255, 255, 255)) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    t = xx if direction == "vertical" else yy
    on = ((t // (period // 2)) % 2 == 0)[..., None]
    return (on * np.array(color, np.uint8)).astype(np.uint8)


def sanity_check(verbose=True) -> dict:
    """Checks that each hand-written component does what the theory says (run before the real data)."""
    rng = np.random.default_rng(0)
    results = {}

    # 1) HSV conversion matches Python's reference colorsys implementation
    px = rng.integers(0, 256, (500, 3), dtype=np.uint8)
    ours = rgb_to_hsv(px[None])[0]
    ref = np.array([colorsys.rgb_to_hsv(*(p / 255.0)) for p in px])
    results["hsv_max_abs_error"] = float(np.abs(ours - ref).max())
    assert results["hsv_max_abs_error"] < 1e-5

    # 2) HOG orientation: horizontal stripes -> vertical gradients -> all energy in the 90 deg bin (index 4);
    #    vertical stripes -> horizontal gradients -> energy split between the 10 and 170 deg bins (0 and 8)
    h_hist = hog_cell_histograms(rgb_to_gray(_stripes("horizontal"))).sum((0, 1))
    v_hist = hog_cell_histograms(rgb_to_gray(_stripes("vertical"))).sum((0, 1))
    results["hog_horizontal_stripes_frac_in_90deg_bin"] = float(h_hist[4] / h_hist.sum())
    results["hog_vertical_stripes_frac_in_0_180deg_bins"] = float((v_hist[0] + v_hist[8]) / v_hist.sum())
    assert results["hog_horizontal_stripes_frac_in_90deg_bin"] > 0.99
    assert results["hog_vertical_stripes_frac_in_0_180deg_bins"] > 0.99
    d = hog(rgb_to_gray(_stripes("vertical")))
    n_blocks = (FEAT_SIZE // HOG_CELL - HOG_BLOCK + 1) ** 2
    results["hog_dim"] = int(d.size)
    assert d.size == n_blocks * HOG_BLOCK**2 * HOG_BINS and np.all(d <= 1.0 + 1e-6)

    # 3) LBP: a flat image gives code 255 everywhere (all neighbours >= centre) -> a single uniform bin
    flat = np.full((32, 32), 100.0)
    codes = lbp_codes(flat)
    results["lbp_flat_unique_codes"] = np.unique(codes).tolist()
    assert np.all(codes == 255)
    assert np.isclose((lbp_histogram(flat) ** 2).max(), 1.0)

    # 4) SVM learns 3 linearly separable Gaussian blobs
    centers = np.array([[0, 4], [4, -2], [-4, -2]], np.float32)
    Xb = np.concatenate([c + rng.standard_normal((200, 2)).astype(np.float32) for c in centers])
    yb = np.repeat(np.arange(3), 200)
    svm = LinearSVM(reg=1e-4, lr=0.01, epochs=30)
    svm.fit(Xb, yb, Xb, yb)
    results["svm_blob_accuracy"] = float((svm.predict(Xb) == yb).mean() * 100)
    assert results["svm_blob_accuracy"] > 95

    # 5) End to end on synthetic "waste": green blobs / blue vertical stripes / grey horizontal stripes
    def synth(cls):
        if cls == 0:
            img = np.zeros((FEAT_SIZE, FEAT_SIZE, 3), np.uint8)
            img[..., 1] = rng.integers(120, 200)
            for _ in range(6):
                cy, cx, r = rng.integers(10, 118, 2).tolist() + [int(rng.integers(6, 16))]
                yy, xx = np.ogrid[:FEAT_SIZE, :FEAT_SIZE]
                img[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = (110, 70, 30)
        elif cls == 1:
            img = _stripes("vertical", period=int(rng.choice([8, 16, 32])), color=(40, 90, 230))
        else:
            img = _stripes("horizontal", period=int(rng.choice([8, 16, 32])), color=(150, 150, 150))
        noise = rng.normal(0, 12, img.shape)
        return np.clip(img + noise, 0, 255).astype(np.uint8)

    ys = np.repeat(np.arange(3), 40)
    Xs = np.stack([extract_features(synth(c)) for c in ys])
    perm = rng.permutation(len(ys))
    tr, te = perm[:90], perm[90:]
    svm = LinearSVM(reg=1e-3, lr=0.01, epochs=30)
    svm.fit(Xs[tr], ys[tr])
    results["synthetic_end_to_end_test_accuracy"] = float((svm.predict(Xs[te]) == ys[te]).mean() * 100)
    assert results["synthetic_end_to_end_test_accuracy"] > 90

    if verbose:
        for k, v in results.items():
            print(f"  {k}: {v}")
        print("All classical sanity checks passed.")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["sanity", "train", "ablation"])
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args(argv)
    if args.command == "sanity":
        sanity_check()
    elif args.command == "train":
        train(epochs=args.epochs)
    else:
        feature_ablation()


if __name__ == "__main__":
    main()
