"""Evaluate all trained methods on the SAME held-out test split and fill the results table.

    python evaluate.py

Outputs (results/):
    results.json            everything below in one file (the app reads the recommendation from here)
    results_table.md / .csv the comparison table for the report (quality + efficiency metrics)
    confusion_matrices.png  one confusion matrix per method (test split)
    training_curves.png     train/val loss and accuracy per epoch for every method
    robustness.png          macro-F1 under blur / darker / brighter / noise / rotation
    misclassified_<m>.png   test images each method got wrong (for the error analysis)

Experiments run here:
    1. main comparison on the test split (accuracy, macro-F1, confusion matrix, per-class F1)
    2. efficiency: runtime per image on CPU (and GPU), parameters, training time
    3. per-source accuracy (web photos from Garbage-12 vs real landfill photos from RealWaste)
    4. robustness to image corruptions a phone/webcam produces
    5. local photos (if data/raw/local/ was provided): generalisation to our own Iloilo pictures
"""
from __future__ import annotations

import argparse

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

from common import (CLASSES, DISPLAY, METHOD_INFO, MODELS_DIR, RESULTS_DIR, SPLITS_DIR, classification_metrics, load_json,
                    load_rgb, load_split, save_json, time_per_image)
from methods import load_predictors

# why a method wins (strength) and why it loses (weakness): the theory behind the numbers
STRENGTH = {
    "classical": "Its hand-designed cues (edge directions, colour histogram, texture codes) are cheap to compute, need no "
                 "GPU and are fully explainable.",
    "scratch": "It learned its own filters directly from our waste photos and is small enough for real-time use.",
    "pretrained": "It starts from features learned on 1.2 million ImageNet photos (edges, textures, object parts) and only "
                  "adapts them to waste, so it generalises best to new photos taken with a phone or webcam.",
}
WEAKNESS = {
    "classical": "fixed features cannot tell apart items with similar colour and texture, e.g. a plastic bottle vs a "
                 "sachet or cardboard vs food waste",
    "scratch": "learned only from ~5k images, so it has less general visual knowledge for unusual items and lighting",
    "pretrained": "largest model and slowest per image of the three",
}


def batched(predictor, images, bs=64):
    out = [predictor.predict_proba(images[i : i + bs]) for i in range(0, len(images), bs)]
    return np.concatenate(out)


# ---- corruptions for the robustness experiment ----------------------------------------
def _blur(im):
    return np.asarray(Image.fromarray(im).filter(ImageFilter.GaussianBlur(2.5)))


def _dark(im):
    return (im.astype(np.float32) * 0.55).astype(np.uint8)


def _bright(im):
    return np.clip(im.astype(np.float32) * 1.5 + 20, 0, 255).astype(np.uint8)


def _noise(im, rng=np.random.default_rng(0)):
    return np.clip(im + rng.normal(0, 20, im.shape), 0, 255).astype(np.uint8)


def _rotate(im):
    return np.asarray(Image.fromarray(im).rotate(30, resample=Image.BILINEAR, fillcolor=(128, 128, 128)))


CORRUPTIONS = {"clean": lambda im: im, "blur": _blur, "dark": _dark, "bright": _bright, "noise": _noise, "rotate 30": _rotate}


# ---- plots ------------------------------------------------------------------------------
def plot_confusions(results, path):
    import matplotlib.pyplot as plt

    keys = list(results)
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4.6), squeeze=False)
    for ax, k in zip(axes[0], keys):
        cm = np.array(results[k]["test"]["confusion_matrix"])
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, f"{cm[i, j]}\n{norm[i, j] * 100:.0f}%", ha="center", va="center",
                        color="white" if norm[i, j] > 0.6 else "black", fontsize=9)
        ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=20)
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        ax.set_xlabel("predicted bin")
        ax.set_ylabel("true bin")
        ax.set_title(f"{METHOD_INFO[k]['short']}\nacc {results[k]['test']['accuracy']:.1f}%  macro-F1 {results[k]['test']['macro_f1']:.1f}%")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_training_curves(path):
    import matplotlib.pyplot as plt

    from common import plot_history

    metas = {k: load_json(MODELS_DIR / f"{f}_meta.json") for k, f in
             [("classical", "classical"), ("scratch", "scratch_cnn"), ("pretrained", "pretrained")]}
    metas = {k: m for k, m in metas.items() if m}
    if not metas:
        return
    fig, axes = plt.subplots(len(metas), 2, figsize=(11, 3.6 * len(metas)), squeeze=False)
    for row, (k, m) in zip(axes, metas.items()):
        plot_history(m["history"], METHOD_INFO[k]["short"], row[0], row[1])
        if k == "pretrained" and m.get("stage1_epochs_run"):
            for ax in row:
                ax.axvline(m["stage1_epochs_run"] + 0.5, color="grey", linestyle=":", label="backbone unfrozen")
            row[0].legend()
        row[0].set_title(f"{METHOD_INFO[k]['short']}: loss" + (" (multiclass hinge)" if k == "classical" else " (cross-entropy)"))
        row[1].set_title(f"{METHOD_INFO[k]['short']}: accuracy / macro-F1")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_robustness(rob, path):
    import matplotlib.pyplot as plt

    names = list(CORRUPTIONS)
    x = np.arange(len(names))
    w = 0.8 / max(len(rob), 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (k, vals) in enumerate(rob.items()):
        ax.bar(x + i * w - 0.4 + w / 2, [vals[n] for n in names], w, label=METHOD_INFO[k]["short"])
    ax.set_xticks(x, names)
    ax.set_ylabel("macro F1 (%)")
    ax.set_xlabel("test-image corruption")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Robustness: macro-F1 on corrupted test images")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_misclassified(images, y, pred, probs, key, path, n=12):
    import matplotlib.pyplot as plt

    wrong = np.flatnonzero(pred != y)
    if len(wrong) == 0:
        return
    rng = np.random.default_rng(0)
    pick = rng.choice(wrong, min(n, len(wrong)), replace=False)
    cols = 6
    rows = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.6 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), pick):
        ax.imshow(images[i])
        ax.set_title(f"true: {CLASSES[y[i]]}\npred: {CLASSES[pred[i]]} {probs[i].max() * 100:.0f}%", fontsize=8)
    fig.suptitle(f"{METHOD_INFO[key]['short']}: misclassified test images")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def prediction_grid(split="test", n=12, seed=1):
    """Qualitative figure: the same random images classified by every method (green = correct, red = wrong)."""
    import matplotlib.pyplot as plt

    preds, _ = load_predictors(verbose=False)
    df = load_split(split)
    rng = np.random.default_rng(seed)
    rows = df.iloc[rng.permutation(len(df))[:n]]
    images = [load_rgb(p) for p in rows["abs_path"]]
    probs = {k: batched(p, images) for k, p in preds.items()}
    cols = 4
    fig, axes = plt.subplots(int(np.ceil(len(images) / cols)), cols, figsize=(4.2 * cols, 4.6 * np.ceil(len(images) / cols)),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, (ax, im, y) in enumerate(zip(axes.ravel(), images, rows["label_idx"])):
        ax.imshow(im)
        ax.set_title(f"true: {CLASSES[y]}", fontsize=10, fontweight="bold")
        for j, k in enumerate(preds):
            pi = probs[k][i]
            ok = pi.argmax() == y
            ax.text(4, 250 - 22 * (len(preds) - 1 - j), f"{METHOD_INFO[k]['short']}: {CLASSES[pi.argmax()]} {pi.max() * 100:.0f}%",
                    fontsize=8, color="white", bbox=dict(facecolor="green" if ok else "red", alpha=0.8, pad=1.5))
    fig.suptitle(f"All methods on the same {split} images (green = correct, red = wrong)")
    fig.tight_layout()
    return fig


# ---- recommendation ----------------------------------------------------------------------
def recommend(results: dict) -> dict:
    """Best = highest test macro-F1; if another method is within 1 point and faster on CPU, prefer the faster one."""
    order = sorted(results, key=lambda k: -results[k]["test"]["macro_f1"])
    best = order[0]
    for k in order[1:]:
        if results[best]["test"]["macro_f1"] - results[k]["test"]["macro_f1"] <= 1.0 and \
                results[k]["runtime_cpu_ms"] < results[best]["runtime_cpu_ms"]:
            best = k
    r = results[best]
    fps = 1000 / max(r["runtime_cpu_ms"], 1e-6)
    lines = [
        f"**{METHOD_INFO[best]['short']}** ({METHOD_INFO[best]['name']}) is recommended: it has the highest macro-F1 on the "
        f"held-out test set ({r['test']['macro_f1']:.1f}%, accuracy {r['test']['accuracy']:.1f}%) and takes "
        f"{r['runtime_cpu_ms']:.1f} ms per image on the CPU (about {fps:.0f} frames/s)"
        + (", fast enough for the live camera." if fps >= 10 else "."),
        STRENGTH[best],
    ]
    others = [f"{METHOD_INFO[k]['short']} reaches {results[k]['test']['macro_f1']:.1f}% macro-F1 at "
              f"{results[k]['runtime_cpu_ms']:.1f} ms/img ({WEAKNESS[k]})" for k in order if k != best]
    if others:
        lines.append("By comparison, " + "; ".join(others) + ".")
    return {"best": best, "explanation": " ".join(lines), "ranking": order}


def results_table(results: dict) -> tuple[str, list[dict]]:
    rows = []
    for k, r in results.items():
        rows.append({
            "Method": METHOD_INFO[k]["name"],
            "Slot": METHOD_INFO[k]["slot"],
            "Top-1 acc (%)": round(r["test"]["accuracy"], 2),
            "Macro F1 (%)": round(r["test"]["macro_f1"], 2),
            "Runtime CPU (ms/img)": round(r["runtime_cpu_ms"], 2),
            "Runtime GPU (ms/img)": round(r["runtime_gpu_ms"], 2) if r.get("runtime_gpu_ms") else "-",
            "Params (M)": round(r["params_millions"], 3),
            "Train time (min)": round(r["train_time_min"], 2) if r.get("train_time_min") is not None else "-",
        })
    header = list(rows[0])
    md = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    md += "\n".join("| " + " | ".join(str(row[h]) for h in header) + " |" for row in rows)
    return md, rows


def evaluate(n_timing=100, robustness_n=600, robustness=True) -> dict:
    import torch

    import pandas as pd

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    preds, missing = load_predictors()
    if not preds:
        raise SystemExit("No trained models found in models/. Train at least one method first.")
    df = load_split("test")
    images = [load_rgb(p) for p in tqdm(df["abs_path"], desc="loading test images")]
    y = df["label_idx"].to_numpy()
    sources = df["source"].to_numpy()
    has_gpu = torch.cuda.is_available()
    import platform

    cpu_name = platform.processor() or platform.machine() or "CPU"
    gpu_name = torch.cuda.get_device_name(0) if has_gpu else None

    local = None
    if (SPLITS_DIR / "local_test.csv").exists():
        ldf = load_split("local_test")
        local = ([load_rgb(p) for p in ldf["abs_path"]], ldf["label_idx"].to_numpy())

    results = {}
    for key, p in preds.items():
        print(f"\n=== {METHOD_INFO[key]['short']} ===")
        dev = "cuda" if has_gpu and key != "classical" else "cpu"
        p.to(dev)
        probs = batched(p, images)
        pred = probs.argmax(1)
        m = classification_metrics(y, pred)
        per_source = {s: classification_metrics(y[sources == s], pred[sources == s])["accuracy"] for s in np.unique(sources)}
        meta = load_json(MODELS_DIR / {"classical": "classical_meta.json", "scratch": "scratch_cnn_meta.json",
                                       "pretrained": "pretrained_meta.json"}[key], {})
        timing_imgs = images[:n_timing]
        p.to("cpu")
        cpu_ms = time_per_image(p.predict_proba, timing_imgs)
        gpu_ms = None
        if has_gpu and key != "classical":
            p.to("cuda")
            gpu_ms = time_per_image(p.predict_proba, timing_imgs, sync=p.sync)
        r = {
            "test": m,
            "per_source_accuracy": per_source,
            "runtime_cpu_ms": cpu_ms,
            "runtime_gpu_ms": gpu_ms,
            "params_millions": float(p.n_params),
            "train_time_min": meta.get("train_time_min"),
            "train_device": meta.get("device", "cpu"),
            "best_epoch": meta.get("best_epoch") or meta.get("chosen", {}).get("best_epoch"),
            "val_macro_f1": meta.get("val", {}).get("macro_f1"),
        }
        if local is not None:
            lp = batched(p, local[0]).argmax(1)
            r["local_test"] = classification_metrics(local[1], lp)
        print(f"test acc {m['accuracy']:.2f}%  macro-F1 {m['macro_f1']:.2f}%  | CPU {cpu_ms:.2f} ms/img"
              + (f"  GPU {gpu_ms:.2f} ms/img" if gpu_ms else ""))
        plot_misclassified(images, y, pred, probs, key, RESULTS_DIR / f"misclassified_{key}.png")
        results[key] = r

    rob = {}
    if robustness:
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(images))[:robustness_n]
        sub, ys = [images[i] for i in idx], y[idx]
        for key, p in preds.items():
            p.to("cuda" if has_gpu and key != "classical" else "cpu")
            rob[key] = {}
            for name, fn in CORRUPTIONS.items():
                corrupted = [fn(im) for im in sub]
                rob[key][name] = classification_metrics(ys, batched(p, corrupted).argmax(1))["macro_f1"]
            print(f"robustness {METHOD_INFO[key]['short']}: " + ", ".join(f"{n} {v:.1f}" for n, v in rob[key].items()))
        plot_robustness(rob, RESULTS_DIR / "robustness.png")

    plot_confusions(results, RESULTS_DIR / "confusion_matrices.png")
    plot_training_curves(RESULTS_DIR / "training_curves.png")
    rec = recommend(results)
    md, rows = results_table(results)
    (RESULTS_DIR / "results_table.md").write_text(
        md + f"\n\nRuntime = median time to classify one 256x256 image (batch 1), CPU: {cpu_name}"
        + (f", GPU: {gpu_name}" if gpu_name else "") + ". Params = 0 for the classical method (rubric convention).\n")
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "results_table.csv", index=False)
    out = {
        "results": results,
        "robustness_macro_f1": rob,
        "recommendation": rec,
        "missing_methods": missing,
        "test_images": len(images),
        "gpu": gpu_name,
        "classes": {c: DISPLAY[c] for c in CLASSES},
    }
    save_json(out, RESULTS_DIR / "results.json")
    print("\n" + md)
    print("\nRECOMMENDATION:", rec["explanation"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-robustness", action="store_true")
    ap.add_argument("--timing-images", type=int, default=100)
    a = ap.parse_args(argv)
    evaluate(n_timing=a.timing_images, robustness=not a.no_robustness)


if __name__ == "__main__":
    main()
