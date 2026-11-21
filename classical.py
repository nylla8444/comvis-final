"""Method A - Classical: hand-designed features + SVM, written from scratch in NumPy.

Features (all implemented below with NumPy only, no OpenCV / scikit-image calls):
  * HOG   - histogram of oriented gradients (Dalal & Triggs 2005): shape / edges
  * HSV   - joint 8x4x4 colour histogram: colour (food = brown/green, cardboard = tan, ...)
  * LBP   - uniform local binary patterns (Ojala et al. 2002): surface texture
  HSV and LBP use a 2-level spatial pyramid (Lazebnik et al. 2006): one histogram for the whole image plus one
  for each quarter, so the features also know WHERE a colour or texture is (item in the middle, background
  at the edges).
Classifier: multi-class SVM (Weston-Watkins hinge loss + L2), trained with mini-batch SGD + momentum, either
  * linear, or
  * RBF kernel, approximated with random Fourier features (Rahimi & Recht 2007), which lets the SVM draw curved
    decision boundaries while training with the same SGD code in seconds.
The kernel, its width and the regularisation strength are chosen on the validation split.

    python classical.py sanity     # sanity checks on synthetic images (required by the rubric)
    python classical.py train      # extract features, grid search on val, save models/classical.npz
    python classical.py ablation   # which feature matters, and does the spatial pyramid help?
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
LBP_BINS = 59
PYRAMID_GRID = 2  # colour/texture histograms for the whole image + each cell of a 2x2 grid
FEATURE_VERSION = "v2-pyramid"  # stored with the cached features; change it whenever the features change
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


def pyramid_histogram(bin_idx: np.ndarray, n_bins: int, grid: int | None = PYRAMID_GRID) -> np.ndarray:
    """Histogram of a per-pixel bin-index map for the whole image, then (grid given) for each cell of a grid x grid
    split. Every histogram is square-root (Hellinger) normalised, then all are concatenated."""
    H, W = bin_idx.shape
    regions = [bin_idx]
    if grid:
        regions += [bin_idx[y * H // grid:(y + 1) * H // grid, x * W // grid:(x + 1) * W // grid]
                    for y in range(grid) for x in range(grid)]
    out = []
    for r in regions:
        hist = np.bincount(r.ravel(), minlength=n_bins).astype(np.float32)
        out.append(np.sqrt(hist / max(hist.sum(), 1.0)))
    return np.concatenate(out)


def hsv_histogram(img: np.ndarray, bins=HSV_BINS, pyramid: bool = True) -> np.ndarray:
    """Joint HSV colour histogram (bins[0] x bins[1] x bins[2] values per region)."""
    hsv = rgb_to_hsv(img)
    idx = [np.minimum((hsv[..., i] * bins[i]).astype(np.int64), bins[i] - 1) for i in range(3)]
    flat = (idx[0] * bins[1] + idx[1]) * bins[2] + idx[2]
    return pyramid_histogram(flat, int(np.prod(bins)), PYRAMID_GRID if pyramid else None)


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


def lbp_histogram(gray: np.ndarray, pyramid: bool = True) -> np.ndarray:
    """Histogram of uniform LBP codes (59 values per region)."""
    return pyramid_histogram(LBP_TABLE[lbp_codes(gray)], LBP_BINS, PYRAMID_GRID if pyramid else None)


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
    """hog / hsv / lbp = the three feature groups; hsv0 / lbp0 = only their whole-image histogram (no pyramid)."""
    if not FEATURE_SLICES:
        dummy = np.zeros((FEAT_SIZE, FEAT_SIZE, 3), np.uint8)
        start = 0
        for part in ("hog", "hsv", "lbp"):
            n = extract_features(dummy, (part,)).size
            FEATURE_SLICES[part] = slice(start, start + n)
            start += n
        hsv, lbp = FEATURE_SLICES["hsv"], FEATURE_SLICES["lbp"]
        FEATURE_SLICES["hsv0"] = slice(hsv.start, hsv.start + int(np.prod(HSV_BINS)))
        FEATURE_SLICES["lbp0"] = slice(lbp.start, lbp.start + LBP_BINS)
    return FEATURE_SLICES


def feature_columns(parts) -> np.ndarray:
    """Column indices of the full feature vector that belong to the given feature groups."""
    fs = feature_slices()
    return np.concatenate([np.arange(fs[p].start, fs[p].stop) for p in parts])


def extract_split(split: str, use_cache: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Features for every image of a split, cached in data/processed/features_<split>.npz."""
    df = load_split(split)
    cache = PROCESSED_DIR / f"features_{split}.npz"
    if use_cache and cache.exists():
        z = np.load(cache, allow_pickle=False)
        if "version" in z.files and str(z["version"]) == FEATURE_VERSION and list(z["paths"]) == list(df["path"]):
            return z["X"], z["y"]
    X = np.stack([extract_features(load_rgb(p)) for p in tqdm(df["abs_path"], desc=f"features [{split}]")])
    y = df["label_idx"].to_numpy()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y, paths=df["path"].to_numpy().astype(str), version=np.array(FEATURE_VERSION))
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



class RandomFourierFeatures:
    """Explicit feature map whose dot products approximate the RBF kernel  k(x, y) = exp(-gamma ||x - y||^2)
    (Rahimi & Recht 2007):   z(x) = sqrt(2 / D) * cos(x Omega + beta),  Omega ~ N(0, 2 gamma),  beta ~ U[0, 2 pi).

    A LINEAR SVM trained on z(x) therefore behaves like an RBF-kernel SVM (curved decision boundaries) but still
    trains with plain SGD in seconds. gamma = gamma_scale / (median squared distance between training points), the
    usual 'median heuristic'. Omega and beta are regenerated from the seed when a model is loaded."""

    def __init__(self, n_components: int = 4096, gamma_scale: float = 1.0, seed: int = SEED):
        self.D, self.gamma_scale, self.seed = n_components, gamma_scale, seed
        self.gamma = self.omega = self.beta = None

    def fit(self, X: np.ndarray) -> "RandomFourierFeatures":
        rng = np.random.default_rng(self.seed)
        S = X[rng.choice(len(X), size=min(len(X), 1000), replace=False)].astype(np.float64)
        sq = (S**2).sum(1)
        d2 = sq[:, None] + sq[None, :] - 2.0 * S @ S.T  # all pairwise squared distances of the sample
        median = float(np.median(d2[np.triu_indices(len(S), 1)])) if len(S) > 1 else 1.0
        self.gamma = self.gamma_scale / max(median, 1e-12)
        self.draw(X.shape[1])
        return self

    def draw(self, in_dim: int) -> None:
        rng = np.random.default_rng(self.seed + 1)
        self.omega = (rng.standard_normal((in_dim, self.D)) * np.sqrt(2.0 * self.gamma)).astype(np.float32)
        self.beta = rng.uniform(0.0, 2.0 * np.pi, self.D).astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.sqrt(2.0 / self.D) * np.cos(X.astype(np.float32) @ self.omega + self.beta)).astype(np.float32)


class ClassicalModel:
    """The full classifier: pick feature columns -> standardise -> (RBF: random Fourier features) -> SVM."""

    def __init__(self, kernel: str = "linear", reg: float = 1e-3, lr: float = 0.003, epochs: int = 60,
                 gamma_scale: float = 1.0, n_components: int = 4096, cols=None, seed: int = SEED):
        assert kernel in ("linear", "rbf")
        self.kernel, self.cols = kernel, cols
        self.rff = RandomFourierFeatures(n_components, gamma_scale, seed) if kernel == "rbf" else None
        self.svm = LinearSVM(reg=reg, lr=lr, epochs=epochs, seed=seed)
        self.mean = self.std = None

    def _map(self, X: np.ndarray) -> np.ndarray:
        X = X[:, self.cols] if self.cols is not None else X
        Z = ((X - self.mean) / self.std).astype(np.float32)
        return self.rff.transform(Z) if self.rff is not None else Z

    def fit(self, X, y, X_val=None, y_val=None, weights=None, verbose=False) -> dict:
        Xc = X[:, self.cols] if self.cols is not None else X
        self.mean, self.std = Xc.mean(0), Xc.std(0) + 1e-6  # training-set statistics only
        if self.rff is not None:
            self.rff.fit(((Xc - self.mean) / self.std).astype(np.float32))
        Z_val = self._map(X_val) if X_val is not None else None
        return self.svm.fit(self._map(X), y, Z_val, y_val, weights=weights, verbose=verbose)

    @property
    def best_epoch(self) -> int:
        return self.svm.best_epoch

    def decision_function(self, X):
        return self.svm.decision_function(self._map(X))

    def predict(self, X):
        return self.decision_function(X).argmax(1)

    def describe(self) -> dict:
        d = {"kernel": self.kernel, "reg": self.svm.reg, "lr": self.svm.lr, "epochs": self.svm.epochs,
             "best_epoch": self.svm.best_epoch}
        if self.rff is not None:
            d.update(gamma_scale=self.rff.gamma_scale, gamma=self.rff.gamma, random_features=self.rff.D)
        return d

    def save(self, path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rff = self.rff
        np.savez(path, kernel=np.array(self.kernel), cols=np.asarray(self.cols, np.int64), mean=self.mean, std=self.std,
                 svm_W=self.svm.W, svm_b=self.svm.b, svm_mean=self.svm.mean, svm_std=self.svm.std,
                 reg=self.svm.reg, lr=self.svm.lr, epochs=self.svm.epochs, best_epoch=self.svm.best_epoch,
                 rff_D=rff.D if rff else 0, rff_gamma_scale=rff.gamma_scale if rff else 0.0,
                 rff_gamma=rff.gamma if rff else 0.0, rff_seed=rff.seed if rff else 0)

    @classmethod
    def load(cls, path) -> "ClassicalModel":
        z = np.load(path, allow_pickle=False)
        if "kernel" not in z.files:
            raise ValueError(f"{path} was saved by an older version of classical.py - run `python classical.py train`")
        m = cls(kernel=str(z["kernel"]), reg=float(z["reg"]), lr=float(z["lr"]), epochs=int(z["epochs"]),
                gamma_scale=float(z["rff_gamma_scale"]), n_components=int(z["rff_D"]) or 1, cols=z["cols"],
                seed=int(z["rff_seed"]))
        m.mean, m.std = z["mean"], z["std"]
        m.svm.W, m.svm.b, m.svm.mean, m.svm.std = z["svm_W"], z["svm_b"], z["svm_mean"], z["svm_std"]
        m.svm.best_epoch = int(z["best_epoch"])
        if m.rff is not None:
            m.rff.gamma = float(z["rff_gamma"])
            m.rff.draw(len(m.cols))
        return m


# =============================================================================================
# Training / prediction
# =============================================================================================
def train(epochs=60, lr=0.003, kernels=("linear", "rbf"), linear_regs=(1e-3, 1e-2, 1e-1, 1.0),
          rbf_gammas=(0.5, 1.0, 2.0), rbf_regs=(1e-2, 1e-1, 1.0), n_random_features=4096,
          parts=("hog", "hsv", "lbp"), verbose=True, save=True) -> dict:
    """Grid search on the validation split over the kernel (linear / RBF), the RBF width (gamma_scale) and the
    regularisation strength. The setting with the best validation macro-F1 is kept, at its best epoch."""
    parts = tuple(parts)
    t0 = time.perf_counter()
    X_tr, y_tr = extract_split("train")
    X_va, y_va = extract_split("val")
    t_feat = time.perf_counter() - t0
    cols = feature_columns(parts)
    cw = class_weights(y_tr)

    configs = []
    if "linear" in kernels:
        configs += [{"kernel": "linear", "reg": r} for r in linear_regs]
    if "rbf" in kernels:
        configs += [{"kernel": "rbf", "gamma_scale": g, "reg": r} for g in rbf_gammas for r in rbf_regs]
    t1 = time.perf_counter()
    grid, best = [], None
    for cfg in configs:
        model = ClassicalModel(lr=lr, epochs=epochs, n_components=n_random_features, cols=cols, **cfg)
        hist = model.fit(X_tr, y_tr, X_va, y_va, weights=cw)
        f1 = max(hist["val_f1"])
        grid.append({**cfg, "lr": lr, "val_macro_f1": f1, "best_epoch": model.best_epoch})
        if verbose:
            width = f" gamma_scale={cfg['gamma_scale']:g}" if cfg["kernel"] == "rbf" else ""
            print(f"{cfg['kernel']:<6}{width} reg={cfg['reg']:g}: best val macro-F1 {f1:.2f} (epoch {model.best_epoch})")
        if best is None or f1 > best[0]:
            best = (f1, model, hist)
    t_svm = time.perf_counter() - t1
    f1, model, hist = best
    if not save:
        return {"features": "+".join(parts), "dims": int(len(cols)), "chosen": model.describe(), "val_macro_f1": f1}

    model.save(MODEL_PATH)
    val_metrics = classification_metrics(y_va, model.predict(X_va))
    kernel_name = "RBF-kernel SVM (random Fourier features)" if model.kernel == "rbf" else "linear SVM"
    meta = {
        "method": "classical",
        "display_name": f"HOG + colour + LBP (spatial pyramid) -> {kernel_name}",
        "features": {p: int(feature_slices()[p].stop - feature_slices()[p].start) for p in parts},
        "feature_dim": int(len(cols)),
        "image_size": FEAT_SIZE,
        "chosen": model.describe(),
        "grid": grid,
        "val": val_metrics,
        "params_millions": 0.0,  # rubric: report 0 for classical methods
        "svm_weights": int(model.svm.W.size + model.svm.b.size),
        "feature_extraction_min": t_feat / 60,
        "train_time_min": (t_feat + t_svm) / 60,
        "history": hist,
    }
    save_json(meta, MODELS_DIR / "classical_meta.json")
    print(f"\nchosen: {model.describe()}")
    print(f"saved {MODEL_PATH}  | val acc {val_metrics['accuracy']:.2f}%  macro-F1 {val_metrics['macro_f1']:.2f}%"
          f"  | train time {meta['train_time_min']:.2f} min")
    return meta


ABLATION_SETS = [
    (("hog",), "HOG only (shape)"),
    (("hsv",), "colour only (pyramid)"),
    (("lbp",), "texture only (pyramid)"),
    (("hsv", "lbp"), "colour + texture (pyramid)"),
    (("hog", "hsv0", "lbp0"), "all three, whole-image histograms only (no pyramid)"),
    (("hog", "hsv", "lbp"), "all three, with spatial pyramid (used)"),
]


def feature_ablation(epochs=40, reg=0.1, lr=0.003) -> list[dict]:
    """Which hand-designed cue matters, and does the spatial pyramid help? Every row uses the same linear SVM
    setting, so only the features change (validation macro-F1)."""
    rows = []
    for parts, desc in ABLATION_SETS:
        r = train(epochs=epochs, lr=lr, kernels=("linear",), linear_regs=(reg,), parts=parts, verbose=False, save=False)
        rows.append({"features": r["features"], "description": desc, "dims": r["dims"], "val_macro_f1": r["val_macro_f1"]})
        print(f"  {desc:<55} {r['dims']:>5} dims  val macro-F1 {r['val_macro_f1']:.2f}%")
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
    h = hsv_histogram(im, pyramid=False) ** 2
    ax[4].bar(np.arange(h.size), h, width=1.0)
    ax[4].set_title("joint HSV histogram, whole image (8x4x4 bins)")
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
        self.model = ClassicalModel.load(path)
        self.n_params = 0
        self.device = "cpu"

    def to(self, device):  # API parity with the torch predictors; NumPy always runs on the CPU
        return self

    def predict_proba(self, images: list[np.ndarray]) -> np.ndarray:
        """SVM scores turned into softmax 'confidences' for display (not calibrated probabilities)."""
        X = np.stack([extract_features(im) for im in images])
        return softmax(self.model.decision_function(X))


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
    assert np.isclose((lbp_histogram(flat, pyramid=False) ** 2).max(), 1.0)

    # 3b) spatial pyramid: whole image + 4 quarters; its first block is exactly the whole-image histogram, and a
    #     picture that is green on the left and blue on the right gets different quarter histograms
    img = np.zeros((FEAT_SIZE, FEAT_SIZE, 3), np.uint8)
    img[:, : FEAT_SIZE // 2] = (30, 200, 40)
    img[:, FEAT_SIZE // 2 :] = (30, 60, 220)
    pyr, glob = hsv_histogram(img), hsv_histogram(img, pyramid=False)
    nb = int(np.prod(HSV_BINS))
    results["pyramid_dims"] = int(pyr.size)
    assert pyr.size == (1 + PYRAMID_GRID**2) * nb and np.allclose(pyr[:nb], glob)
    assert not np.allclose(pyr[nb : 2 * nb], pyr[2 * nb : 3 * nb])  # top-left quarter != top-right quarter

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

    # 6) Random Fourier features approximate the RBF kernel: z(x).z(y) ~ exp(-gamma ||x - y||^2)
    Xk = rng.standard_normal((200, 10)).astype(np.float32)
    rff = RandomFourierFeatures(n_components=4096, gamma_scale=1.0).fit(Xk)
    Zk = rff.transform(Xk).astype(np.float64)
    sq = (Xk.astype(np.float64) ** 2).sum(1)
    K = np.exp(-rff.gamma * np.maximum(sq[:, None] + sq[None, :] - 2 * Xk @ Xk.T, 0))
    results["rff_kernel_mean_abs_error"] = float(np.abs(Zk @ Zk.T - K).mean())
    assert results["rff_kernel_mean_abs_error"] < 0.03

    # 7) A problem no straight line can solve (two concentric rings): linear SVM ~ chance, RBF SVM solves it
    n = 300
    radius = np.concatenate([rng.uniform(0, 1, n), rng.uniform(2, 3, n)])
    angle = rng.uniform(0, 2 * np.pi, 2 * n)
    Xr = np.stack([radius * np.cos(angle), radius * np.sin(angle)], 1).astype(np.float32)
    yr = np.repeat([0, 1], n)
    acc = {}
    for kernel in ("linear", "rbf"):
        m = ClassicalModel(kernel=kernel, reg=1e-3, lr=0.01, epochs=40, n_components=512)
        m.svm.n_classes = 2
        m.fit(Xr, yr)
        acc[kernel] = float((m.predict(Xr) == yr).mean() * 100)
    results["rings_linear_svm_accuracy"], results["rings_rbf_svm_accuracy"] = acc["linear"], acc["rbf"]
    assert acc["rbf"] > 95 and acc["rbf"] > acc["linear"] + 20

    if verbose:
        for k, v in results.items():
            print(f"  {k}: {v}")
        print("All classical sanity checks passed.")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["sanity", "train", "ablation"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--kernel", choices=["both", "linear", "rbf"], default="both", help="which SVM kernels to try")
    args = ap.parse_args(argv)
    if args.command == "sanity":
        sanity_check()
    elif args.command == "train":
        train(epochs=args.epochs, kernels=("linear", "rbf") if args.kernel == "both" else (args.kernel,))
    else:
        feature_ablation()


if __name__ == "__main__":
    main()
