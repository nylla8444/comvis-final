"""Evaluate the three fine-tuned models on the SAME held-out test split and fill the results table.

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
    2. efficiency: runtime per image on CPU and GPU, parameters, multiply-adds, training time
    3. per-source accuracy (web photos from Garbage-12 vs real landfill photos from RealWaste)
    4. robustness to image corruptions a phone/webcam produces
    5. frozen vs fine-tuned: validation macro-F1 with only the new head trained (ImageNet features frozen) vs after
       fine-tuning the whole network (from each model's training log)
    6. local photos (if data/raw/local/ was provided): generalisation to our own pictures
"""
from __future__ import annotations

import argparse

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

from common import (CLASSES, DISPLAY, METHOD_INFO, MODELS_DIR, RESULTS_DIR, SPLITS_DIR, classification_metrics, load_json,
                    load_rgb, load_split, save_json, time_per_image)
from methods import load_predictors

# the design idea behind each model: the theory used to explain the measured numbers
DESIGN = {
    "mobilenet_v3_large": "MobileNetV3 was designed for phones: depthwise-separable convolutions and squeeze-and-excitation give it "
                          "the fewest multiply-adds of the three.",
    "efficientnet_b0": "EfficientNet-B0 scales depth, width and input resolution together (compound scaling), which gives a lot "
                       "of accuracy per parameter.",
    "resnet18": "ResNet-18 uses full 3x3 convolutions with skip connections: the most parameters and multiply-adds of the three, "
                "which suits a GPU better than a CPU.",
}


def batched(predictor, images, bs=64):
    out = [predictor.predict_proba(images[i : i + bs]) for i in range(0, len(images), bs)]
    return np.concatenate(out)


def interleaved_timing(preds: dict, images: list, device: str, rounds: int = 5) -> dict:
    """Milliseconds to classify one image (batch 1) for every model, measured in alternating rounds.

    Laptop CPUs change speed all the time (power modes, turbo, and on hybrid CPUs fast vs power-saving cores), so timing
    one model after the other can make the same model look 2x slower in one pass than in the next. Here every round
    times every model on the same images, in a rotated order, so all models share the same conditions; the result is
    the median over the rounds. Returns {key: {"ms": median, "rounds": [median of each round]}}."""
    keys = list(preds)
    for p in preds.values():
        p.to(device)
    per = {k: [] for k in keys}
    for i, chunk in enumerate(np.array_split(np.arange(len(images)), rounds)):
        sub = [images[j] for j in chunk]
        for k in keys[i % len(keys):] + keys[: i % len(keys)]:  # rotate which model goes first
            p = preds[k]
            sync = getattr(p, "sync", None) if device == "cuda" else None
            per[k].append(time_per_image(p.predict_proba, sub, warmup=3, sync=sync))
    return {k: {"ms": float(np.median(v)), "rounds": [round(float(x), 2) for x in v]} for k, v in per.items()}


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

    metas = {k: load_json(MODELS_DIR / f"{k}_meta.json") for k in METHOD_INFO}
    metas = {k: m for k, m in metas.items() if m}
    if not metas:
        return
    fig, axes = plt.subplots(len(metas), 2, figsize=(11, 3.6 * len(metas)), squeeze=False)
    for row, (k, m) in zip(axes, metas.items()):
        plot_history(m["history"], METHOD_INFO[k]["short"], row[0], row[1])
        if m.get("stage1_epochs_run"):
            for ax in row:
                ax.axvline(m["stage1_epochs_run"] + 0.5, color="grey", linestyle=":", label="backbone unfrozen")
            row[0].legend()
        row[0].set_title(f"{METHOD_INFO[k]['short']}: loss (cross-entropy)")
        row[1].set_title(f"{METHOD_INFO[k]['short']}: accuracy / macro-F1")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_robustness(rob, path):
    import matplotlib.pyplot as plt

    names = list(CORRUPTIONS)
    x = np.arange(len(names))
    w = 0.8 / max(len(rob), 1)
    fig, ax = plt.subplots(figsize=(11.5, 4), layout="constrained")
    for i, (k, vals) in enumerate(rob.items()):
        bars = ax.bar(x + i * w - 0.4 + w / 2, [vals[n] for n in names], w, label=METHOD_INFO[k]["short"])
        ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=1)
    ax.set_xticks(x, names)
    ax.set_ylabel("macro F1 (%)")
    ax.set_xlabel("test-image corruption")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)  # outside the bars
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Robustness: macro-F1 on corrupted test images")
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
    # constrained layout reserves room for the two-line titles, so they never cover the row above
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 3.0 * rows), squeeze=False, layout="constrained")
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), pick):
        ax.imshow(images[i])
        ax.set_title(f"true: {CLASSES[y[i]]}\npred: {CLASSES[pred[i]]} {probs[i].max() * 100:.0f}%", fontsize=8)
    fig.suptitle(f"{METHOD_INFO[key]['short']}: misclassified test images")
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
def _why_not(k: str, best: str, results: dict, rob: dict) -> list[str]:
    """Why model k is not the recommended one, from the measured numbers."""
    r, b = results[k], results[best]
    out = []
    gap = b["test"]["macro_f1"] - r["test"]["macro_f1"]
    if gap > 1.0:
        out.append(f"{gap:.1f} points less accurate")
    if r["runtime_cpu_ms"] > 1.1 * b["runtime_cpu_ms"]:
        txt = f"{r['runtime_cpu_ms'] / b['runtime_cpu_ms']:.1f}x slower on the CPU"
        if r.get("macs_millions") and b.get("macs_millions") and r["macs_millions"] > 1.3 * b["macs_millions"]:
            txt += f" with {r['macs_millions'] / b['macs_millions']:.1f}x the multiply-adds"
        out.append(txt)
    if rob.get(k) and rob.get(best):
        c = max(rob[k], key=lambda n: rob[best][n] - rob[k][n])
        if rob[best][c] - rob[k][c] >= 5:
            out.append(f"weaker under {c} ({rob[k][c]:.0f}% vs {rob[best][c]:.0f}% macro-F1)")
    if not out and gap > 0:
        out.append(f"{gap:.1f} point less accurate")
    return out


def recommend(results: dict, rob: dict | None = None) -> dict:
    """Recommended = the fastest model on the CPU among those within 1 macro-F1 point of the most accurate one.
    With ~1,000 test images a 1-point gap is about the size of the test-set noise, while speed decides how smooth
    the live camera feels. The explanation is built from the measured numbers."""
    rob = rob or {}
    order = sorted(results, key=lambda k: -results[k]["test"]["macro_f1"])
    top = order[0]
    top_f1 = results[top]["test"]["macro_f1"]
    close = [k for k in order if top_f1 - results[k]["test"]["macro_f1"] <= 1.0]
    best = min(close, key=lambda k: results[k]["runtime_cpu_ms"])
    r = results[best]
    fps = 1000 / max(r["runtime_cpu_ms"], 1e-6)
    score = f"{r['test']['macro_f1']:.1f}% macro-F1, accuracy {r['test']['accuracy']:.1f}% on the held-out test set"
    if best == top:
        why = f"it has the highest score ({score})"
        if len(close) > 1:
            why += " and is also the fastest on the CPU of the models within 1 point of it"
    else:
        why = (f"its score ({score}) is within {top_f1 - r['test']['macro_f1']:.1f} point of the most accurate model "
               f"({METHOD_INFO[top]['short']}, {top_f1:.1f}%), a gap about the size of the test-set noise, and it is the "
               f"fastest of those on the CPU")
    lines = [f"**{METHOD_INFO[best]['short']}** ({r.get('model_name', METHOD_INFO[best]['name'])}) is recommended: {why}. "
             f"It takes {r['runtime_cpu_ms']:.1f} ms per image on the CPU (about {fps:.0f} frames/s)"
             + (", fast enough for the live camera." if fps >= 10 else ".")]
    if rob.get(best) and all(min(rob[best].values()) >= min(v.values()) for v in rob.values()):
        worst = min(rob[best], key=rob[best].get)
        lines.append(f"It is also the most robust: its lowest macro-F1 in the corruption tests is {rob[best][worst]:.0f}% ({worst}).")
    lines.append(DESIGN[best])
    others = []
    for k in order:
        if k == best:
            continue
        reasons = _why_not(k, best, results, rob)
        others.append(f"{METHOD_INFO[k]['short']} reaches {results[k]['test']['macro_f1']:.1f}% macro-F1 at "
                      f"{results[k]['runtime_cpu_ms']:.1f} ms/img" + (f" ({', '.join(reasons)})" if reasons else ""))
    if others:
        lines.append("By comparison, " + "; ".join(others) + ".")
    gpu = {k: v["runtime_gpu_ms"] for k, v in results.items() if v.get("runtime_gpu_ms")}
    if len(gpu) == len(results) and min(gpu, key=gpu.get) != best:
        fastest = min(gpu, key=gpu.get)
        lines.append(f"On a GPU the fastest is {METHOD_INFO[fastest]['short']} ({gpu[fastest]:.1f} ms vs {gpu[best]:.1f} ms), "
                     f"but the app must also run smoothly on laptops without one.")
    return {"best": best, "explanation": " ".join(lines), "ranking": order, "most_accurate": top}


def results_table(results: dict) -> tuple[str, list[dict]]:
    rows = []
    for k, r in results.items():
        rows.append({
            "Method": r.get("model_name", METHOD_INFO[k]["name"]),
            "Slot": METHOD_INFO[k]["slot"],
            "Top-1 acc (%)": round(r["test"]["accuracy"], 2),
            "Macro F1 (%)": round(r["test"]["macro_f1"], 2),
            "Runtime CPU (ms/img)": round(r["runtime_cpu_ms"], 2),
            "Runtime GPU (ms/img)": round(r["runtime_gpu_ms"], 2) if r.get("runtime_gpu_ms") else "-",
            "Params (M)": round(r["params_millions"], 3),
            "MACs (M/img)": round(r["macs_millions"]) if r.get("macs_millions") else "-",
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
        raise SystemExit("No trained models found in models/. Train at least one model first.")
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
        dev = "cuda" if has_gpu else "cpu"
        p.to(dev)
        probs = batched(p, images)
        pred = probs.argmax(1)
        m = classification_metrics(y, pred)
        per_source = {s: classification_metrics(y[sources == s], pred[sources == s])["accuracy"] for s in np.unique(sources)}
        meta = load_json(MODELS_DIR / f"{key}_meta.json", {})
        r = {
            "model_name": meta.get("display_name", METHOD_INFO[key]["name"]),
            "test": m,
            "per_source_accuracy": per_source,
            "runtime_cpu_ms": None,  # filled in below by the interleaved timing
            "runtime_gpu_ms": None,
            "params_millions": float(p.n_params),
            "macs_millions": meta.get("macs_millions"),
            "train_time_min": meta.get("train_time_min"),
            "train_device": meta.get("device", "cpu"),
            "best_epoch": meta.get("best_epoch") or meta.get("chosen", {}).get("best_epoch"),
            "val_macro_f1": meta.get("val", {}).get("macro_f1"),
            "val_macro_f1_head_only": meta.get("stage1_best_val_f1"),  # experiment 5: frozen ImageNet features
        }
        if local is not None:
            lp = batched(p, local[0]).argmax(1)
            r["local_test"] = classification_metrics(local[1], lp)
        print(f"test acc {m['accuracy']:.2f}%  macro-F1 {m['macro_f1']:.2f}%")
        plot_misclassified(images, y, pred, probs, key, RESULTS_DIR / f"misclassified_{key}.png")
        results[key] = r

    # efficiency: all models timed in alternating rounds on the same images (see interleaved_timing)
    timing_imgs = images[:n_timing]
    print("\ntiming (batch 1, models alternate over 5 rounds) ...")
    cpu_t = interleaved_timing(preds, timing_imgs, "cpu")
    gpu_t = interleaved_timing(preds, timing_imgs, "cuda") if has_gpu else {}
    for key, r in results.items():
        r["runtime_cpu_ms"], r["runtime_cpu_rounds_ms"] = cpu_t[key]["ms"], cpu_t[key]["rounds"]
        if gpu_t:
            r["runtime_gpu_ms"], r["runtime_gpu_rounds_ms"] = gpu_t[key]["ms"], gpu_t[key]["rounds"]
        print(f"{METHOD_INFO[key]['short']}: CPU {r['runtime_cpu_ms']:.2f} ms/img (rounds {cpu_t[key]['rounds']})"
              + (f" | GPU {r['runtime_gpu_ms']:.2f} ms/img" if gpu_t else ""))

    rob = {}
    if robustness:
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(images))[:robustness_n]
        sub, ys = [images[i] for i in idx], y[idx]
        for key, p in preds.items():
            p.to("cuda" if has_gpu else "cpu")
            rob[key] = {}
            for name, fn in CORRUPTIONS.items():
                corrupted = [fn(im) for im in sub]
                rob[key][name] = classification_metrics(ys, batched(p, corrupted).argmax(1))["macro_f1"]
            print(f"robustness {METHOD_INFO[key]['short']}: " + ", ".join(f"{n} {v:.1f}" for n, v in rob[key].items()))
        plot_robustness(rob, RESULTS_DIR / "robustness.png")

    plot_confusions(results, RESULTS_DIR / "confusion_matrices.png")
    plot_training_curves(RESULTS_DIR / "training_curves.png")
    rec = recommend(results, rob)
    md, rows = results_table(results)
    (RESULTS_DIR / "results_table.md").write_text(
        md + f"\n\nRuntime = median time to classify one 256x256 image (batch 1), the models timed in 5 alternating "
        f"rounds on the same images; CPU: {cpu_name}"
        + (f", GPU: {gpu_name}" if gpu_name else "") + ". "
        "MACs = multiply-accumulate operations per image in the network's conv/linear layers (the compute that "
        "decides CPU speed; more parameters does not always mean slower).\n")
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
