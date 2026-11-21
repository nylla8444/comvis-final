"""BasuraBin: the waste-segregation app. Runs all three methods on an uploaded image, an uploaded
video or the live webcam, draws the result on the input, and names the recommended method.

    python app.py              # opens http://127.0.0.1:7860 in the browser
    python app.py --share      # also prints a temporary public gradio.live link
    python app.py --cpu        # force CPU (to show real-time speed without a GPU)
"""
from __future__ import annotations

import argparse
import tempfile
from collections import Counter
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

from annotate import UNSURE_THRESHOLD, Smoother, draw_all, draw_single, run_methods
from common import ADVICE, CLASSES, COLORS, DISPLAY, METHOD_INFO, METHOD_KEYS, MODELS_DIR, RESULTS_DIR, SPLITS_DIR, load_json
from methods import load_predictors

BIN_COLOR_NAME = {"biodegradable": "green bin", "recyclable": "blue bin", "residual": "black bin"}
SHORT_BIN = {"biodegradable": "Biodegradable", "recyclable": "Recyclable", "residual": "Residual"}
EXAMPLES_TEXT = {
    "biodegradable": "food scraps, fruit peel, leftover rice, leaves, garden waste",
    "recyclable": "clean paper, cardboard, cans, glass bottles, PET bottles",
    "residual": "sachets, candy wrappers, styrofoam, used tissue, old textiles and shoes",
}
UNSURE_COLOR = "#c27c0e"


HTML_COLORS = {"residual": "#5b6470"}  # a lighter slate than the drawing colour, so bars stay visible in dark mode


def hex_color(c: str) -> str:
    return HTML_COLORS.get(c, "#%02x%02x%02x" % COLORS[c])


class State:
    preds: dict = {}
    missing: list = []
    best: str | None = None
    explanation: str = ""
    results: dict | None = None
    device: str = "CPU"


S = State()


def load_everything(device=None):
    S.preds, S.missing = load_predictors(device)
    S.results = load_json(RESULTS_DIR / "results.json")
    if S.results and S.results["recommendation"]["best"] in S.preds:
        S.best = S.results["recommendation"]["best"]
        S.explanation = S.results["recommendation"]["explanation"]
    else:
        S.best = "pretrained" if "pretrained" in S.preds else next(iter(S.preds), None)
        S.explanation = ("No evaluation results yet - run `python evaluate.py` to compute the test metrics. "
                         "Until then the pretrained model is used as the default.")
    devs = {str(getattr(p, "device", "cpu")) for k, p in S.preds.items() if k != "classical"}
    if any(d.startswith("cuda") for d in devs):
        import torch

        S.device = f"GPU ({torch.cuda.get_device_name(0)})"
    else:
        S.device = "CPU"


# ---- HTML building blocks ---------------------------------------------------------------------
CSS = """
.bb-header {display:flex; flex-wrap:wrap; align-items:flex-end; justify-content:space-between; gap:12px; margin-bottom:4px}
.bb-title {font-size:2rem; font-weight:800; line-height:1.1; margin:0}
.bb-sub {color:var(--body-text-color-subdued); margin-top:4px; font-size:1rem}
.bb-chips {display:flex; gap:8px; flex-wrap:wrap}
.bb-chip {color:#fff; border-radius:999px; padding:5px 14px; font-weight:700; font-size:.9rem; white-space:nowrap}
.bb-status {color:var(--body-text-color-subdued); font-size:.85rem; margin-top:6px}
.bb-verdict {border-radius:16px; padding:18px 22px; color:#fff; box-shadow:0 2px 10px rgba(0,0,0,.12)}
.bb-verdict .kicker {text-transform:uppercase; letter-spacing:.09em; font-size:.8rem; opacity:.9; font-weight:700}
.bb-verdict .bin {font-size:2.1rem; font-weight:800; line-height:1.15; margin-top:2px}
.bb-verdict .bin small {font-size:1rem; font-weight:600; opacity:.9}
.bb-verdict .advice {margin:10px 0 0; font-size:1rem; line-height:1.4}
.bb-verdict .meta {font-size:.88rem; opacity:.92}
.bb-meter {height:10px; background:rgba(255,255,255,.3); border-radius:6px; overflow:hidden; margin:12px 0 5px}
.bb-meter > div {height:100%; background:#fff; border-radius:6px}
.bb-note {margin-top:10px; font-size:.95rem; color:var(--body-text-color)}
.bb-cards {display:grid; grid-template-columns:repeat(auto-fit, minmax(230px, 1fr)); gap:12px}
.bb-card {border:1px solid var(--border-color-primary); border-radius:14px; padding:12px 14px; background:var(--block-background-fill)}
.bb-card.best {border:2px solid var(--color-accent)}
.bb-card .name {font-weight:700; font-size:1rem}
.bb-card .slot {font-size:.8rem; color:var(--body-text-color-subdued); margin:2px 0 8px}
.bb-card .pred {display:inline-block; color:#fff; border-radius:8px; padding:2px 10px; font-weight:700; font-size:.9rem; margin-bottom:8px}
.bb-row {display:grid; grid-template-columns:7.2rem 1fr 3.2rem; align-items:center; gap:8px; font-size:.84rem; margin:4px 0}
.bb-row .pct {text-align:right; font-variant-numeric:tabular-nums}
.bb-bar {height:9px; background:var(--border-color-primary); border-radius:5px; overflow:hidden}
.bb-bar > div {height:100%; border-radius:5px}
.bb-badge {font-size:.72rem; font-weight:700; padding:2px 8px; border-radius:999px; background:var(--color-accent); color:#fff; margin-left:6px; vertical-align:middle}
.bb-empty {padding:28px 18px; text-align:center; color:var(--body-text-color-subdued); border:2px dashed var(--border-color-primary); border-radius:16px; font-size:1rem}
.bb-kpi .big {font-size:1.9rem; font-weight:800; line-height:1.1}
.bb-kpi .lbl {font-size:.8rem; color:var(--body-text-color-subdued)}
.bb-kpi .grid {display:grid; grid-template-columns:1fr 1fr; gap:6px 12px; margin-top:10px; font-size:.86rem}
.bb-kpi .grid b {font-variant-numeric:tabular-nums}
.bb-bins {display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:12px}
.bb-bin {border-radius:14px; padding:14px 16px; color:#fff}
.bb-bin h3 {margin:0 0 4px; color:#fff}
.bb-bin p {margin:0; opacity:.95}
/* white text on the coloured blocks in BOTH light and dark theme (Gradio's prose styles would recolour it) */
.bb-verdict, .bb-verdict *, .bb-chip, .bb-bin, .bb-bin *, .bb-card .pred, .bb-badge {color:#fff !important}
"""


def empty_html(text: str) -> str:
    return f'<div class="bb-empty">{text}</div>'


def header_html() -> str:
    chips = "".join(f'<span class="bb-chip" style="background:{hex_color(c)}">{DISPLAY[c]}</span>' for c in CLASSES)
    loaded = ", ".join(METHOD_INFO[k]["short"] for k in S.preds) or "none"
    best = METHOD_INFO[S.best]["short"] if S.best else "-"
    missing = f" &middot; not trained yet: {', '.join(S.missing)}" if S.missing else ""
    return f"""
<div class="bb-header">
  <div><div class="bb-title">BasuraBin</div>
       <div class="bb-sub">Which bin does it go in? Segregation at source under RA 9003, judged by three computer-vision methods.</div></div>
  <div class="bb-chips">{chips}</div>
</div>
<div class="bb-status">Methods: {loaded} &middot; recommended: <b>{best}</b> &middot; running on {S.device}{missing}</div>"""


def verdict_html(results: dict) -> str:
    if not results or S.best not in results:
        return empty_html("No trained models found. Train the methods first (see README).")
    r = results[S.best]
    label, conf = r["label"], r["conf"] * 100
    unsure = r["conf"] < UNSURE_THRESHOLD
    color = UNSURE_COLOR if unsure else hex_color(label)
    if unsure:
        kicker, title = "Not sure", f"Maybe {SHORT_BIN[label]}?"
        advice = "Hold ONE item inside the square, closer to the camera, on a plain background with good light."
    else:
        kicker, title = "Put it in", f"{DISPLAY[label]} <small>({BIN_COLOR_NAME[label]})</small>"
        advice = ADVICE[label]
    labels = {x["label"] for x in results.values()}
    if len(results) > 1 and len(labels) == 1:
        note = f"&#10003; All {len(results)} methods agree."
    elif len(results) > 1:
        note = "&#9888; The methods disagree on this item. The recommended method is the most accurate one on our test set."
    else:
        note = ""
    return f"""
<div class="bb-verdict" style="background:{color}">
  <div class="kicker">{kicker}</div>
  <div class="bin">{title}</div>
  <div class="bb-meter"><div style="width:{conf:.0f}%"></div></div>
  <div class="meta">{conf:.0f}% confident &middot; decided by {METHOD_INFO[S.best]['short']} (recommended method)</div>
  <div class="advice">{advice}</div>
</div>
<div class="bb-note">{note}</div>"""


def cards_html(results: dict) -> str:
    cards = []
    for k in METHOD_KEYS:
        info = METHOD_INFO[k]
        badge = '<span class="bb-badge">recommended</span>' if k == S.best else ""
        if k not in results:
            cards.append(f'<div class="bb-card"><div class="name">{info["short"]}</div>'
                         f'<div class="slot">{info["slot"]}</div><i>not trained</i></div>')
            continue
        r = results[k]
        unsure = r["conf"] < UNSURE_THRESHOLD
        pred_color = UNSURE_COLOR if unsure else hex_color(r["label"])
        pred_text = f"{SHORT_BIN[r['label']]} {r['conf'] * 100:.0f}%" + (" (not sure)" if unsure else "")
        rows = "".join(
            f'<div class="bb-row"><span>{SHORT_BIN[c]}</span>'
            f'<div class="bb-bar"><div style="width:{r["probs"][i] * 100:.1f}%; background:{hex_color(c)}"></div></div>'
            f'<span class="pct">{r["probs"][i] * 100:.0f}%</span></div>'
            for i, c in enumerate(CLASSES))
        cards.append(f'<div class="bb-card{" best" if k == S.best else ""}"><div class="name">{info["short"]}{badge}</div>'
                     f'<div class="slot">{info["slot"]} &middot; {r["ms"]:.0f} ms</div>'
                     f'<span class="pred" style="background:{pred_color}">{pred_text}</span>{rows}</div>')
    return f'<div class="bb-cards">{"".join(cards)}</div>'


# ---- handlers ----------------------------------------------------------------------------------
def on_image(img):
    if img is None:
        return None, empty_html("Upload a photo, take one with the webcam, or click an example."), "", []
    if not S.preds:
        return None, verdict_html({}), "", []
    results = run_methods(S.preds, img)
    gallery = [(draw_single(img, k, results[k]), f"{METHOD_INFO[k]['slot']}: {METHOD_INFO[k]['short']}")
               for k in METHOD_KEYS if k in results]
    main = draw_single(img, S.best, results[S.best], max_side=900)
    return main, verdict_html(results), cards_html(results), gallery


def on_webcam(frame, smoother=None):
    """One streamed frame. `smoother` is per-browser-session state: predictions are averaged over the last frames."""
    if frame is None or not S.preds:
        return None, "", "", smoother
    smoother = smoother or Smoother()
    results = smoother.update(run_methods(S.preds, frame))
    return draw_all(frame, results, S.best), verdict_html(results), cards_html(results), smoother


def on_video(path, analyse_fps=5.0, max_seconds=60, progress=gr.Progress()):
    import cv2
    import imageio

    if not path or not S.preds:
        return None, empty_html("Upload a short video first.")
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, int(round(src_fps / analyse_fps)))
    out_path = Path(tempfile.mkdtemp()) / "annotated.mp4"
    writer = imageio.get_writer(out_path, fps=src_fps / step, codec="libx264", pixelformat="yuv420p", macro_block_size=2)
    votes = {k: Counter() for k in S.preds}
    times = {k: [] for k in S.preds}
    smoother = Smoother(alpha=0.5, reset_after_s=None)  # analysed frames are 0.2 s apart: average over ~3 of them
    i = 0
    max_frames = int(max_seconds * src_fps)
    while i < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % step == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            results = smoother.update(run_methods(S.preds, rgb))
            for k, r in results.items():
                votes[k][r["label"]] += 1
                times[k].append(r["ms"])
            frame = draw_all(rgb, results, S.best)
            frame = frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]
            writer.append_data(frame)
            if n_total:
                progress(min(i / min(n_total, max_frames), 1.0), desc="analysing video")
        i += 1
    cap.release()
    writer.close()
    if not votes.get(S.best):
        return str(out_path), empty_html("No frames could be read from this video.")
    cards = []
    for k in METHOD_KEYS:
        if k not in votes or not votes[k]:
            continue
        tot = sum(votes[k].values())
        rows = "".join(
            f'<div class="bb-row"><span>{SHORT_BIN[c]}</span>'
            f'<div class="bb-bar"><div style="width:{votes[k][c] / tot * 100:.1f}%; background:{hex_color(c)}"></div></div>'
            f'<span class="pct">{votes[k][c] / tot * 100:.0f}%</span></div>' for c in CLASSES)
        lab = votes[k].most_common(1)[0][0]
        badge = '<span class="bb-badge">recommended</span>' if k == S.best else ""
        cards.append(f'<div class="bb-card{" best" if k == S.best else ""}"><div class="name">{METHOD_INFO[k]["short"]}{badge}</div>'
                     f'<div class="slot">{tot} frames analysed &middot; {np.mean(times[k]):.0f} ms per frame</div>'
                     f'<span class="pred" style="background:{hex_color(lab)}">mostly {SHORT_BIN[lab]}</span>'
                     f'<div class="slot">share of frames per bin:</div>{rows}</div>')
    lab, n = votes[S.best].most_common(1)[0]
    tot = sum(votes[S.best].values())
    head = (f'<div class="bb-verdict" style="background:{hex_color(lab)}"><div class="kicker">Most of the video</div>'
            f'<div class="bin">{DISPLAY[lab]} <small>({BIN_COLOR_NAME[lab]})</small></div>'
            f'<div class="meta">{n} of {tot} analysed frames ({n / tot * 100:.0f}%) according to '
            f'{METHOD_INFO[S.best]["short"]}</div></div><br>')
    return str(out_path), head + f'<div class="bb-cards">{"".join(cards)}</div>'


def example_images(n_per_class=3) -> list[str]:
    try:
        df = pd.read_csv(SPLITS_DIR / "test.csv")
    except FileNotFoundError:
        return []
    root = Path(__file__).resolve().parent
    picks = df.sample(frac=1.0, random_state=3).groupby("label").head(n_per_class).sort_values("label")["path"].tolist()
    return [str(root / p) for p in picks if (root / p).exists()]


# ---- model comparison tab ----------------------------------------------------------------------
def kpi_html() -> str:
    if not S.results:
        return empty_html("No results yet. Run <code>python evaluate.py</code> after training the three methods.")
    cards = []
    for k in METHOD_KEYS:
        r = S.results["results"].get(k)
        if not r:
            continue
        badge = '<span class="bb-badge">recommended</span>' if k == S.best else ""
        gpu = f"{r['runtime_gpu_ms']:.1f} ms" if r.get("runtime_gpu_ms") else "-"
        macs = f"{r['macs_millions']:.0f} M" if r.get("macs_millions") else "-"
        train = f"{r['train_time_min']:.1f} min" if r.get("train_time_min") is not None else "-"
        cards.append(f"""
<div class="bb-card bb-kpi{' best' if k == S.best else ''}">
  <div class="name">{METHOD_INFO[k]['short']}{badge}</div>
  <div class="slot">{METHOD_INFO[k]['slot']} &middot; {r.get('model_name', METHOD_INFO[k]['name'])}</div>
  <div class="big">{r['test']['macro_f1']:.1f}%</div><div class="lbl">macro-F1 on the test set ({S.results.get('test_images', '?')} images)</div>
  <div class="grid">
    <span>Accuracy</span><b>{r['test']['accuracy']:.1f}%</b>
    <span>CPU per image</span><b>{r['runtime_cpu_ms']:.1f} ms</b>
    <span>GPU per image</span><b>{gpu}</b>
    <span>Parameters</span><b>{r['params_millions']:.2f} M</b>
    <span>Compute (MACs)</span><b>{macs}</b>
    <span>Training time</span><b>{train}</b>
  </div>
</div>""")
    return f'<div class="bb-cards">{"".join(cards)}</div>'


def experiments_md() -> str:
    parts = []
    abl = load_json(MODELS_DIR / "classical_ablation.json")
    if abl and isinstance(abl[0], dict) and "description" in abl[0]:
        parts.append("**Classical feature ablation** (same linear SVM, only the features change; validation macro-F1)\n\n"
                     "| Features | Dimensions | Val macro-F1 (%) |\n|---|---|---|\n"
                     + "\n".join(f"| {r['description']} | {r['dims']} | {r['val_macro_f1']:.1f} |" for r in abl))
    cmeta = load_json(MODELS_DIR / "classical_meta.json")
    if cmeta and cmeta.get("grid"):
        best = {}
        for g in cmeta["grid"]:
            best[g["kernel"]] = max(best.get(g["kernel"], 0), g["val_macro_f1"])
        parts.append("**Classical SVM kernel** (best validation macro-F1 per kernel): "
                     + ", ".join(f"{k} {v:.1f}%" for k, v in best.items()) + f". Chosen: `{cmeta.get('chosen')}`")
    smeta = load_json(MODELS_DIR / "scratch_cnn_meta.json")
    if smeta and smeta.get("selection"):
        parts.append("**Scratch CNN designs** (same data and recipe; the better on validation is used)\n\n"
                     "| Design | Input | Params (M) | MACs (M) | Best epoch | Val macro-F1 (%) |\n|---|---|---|---|---|---|\n"
                     + "\n".join(f"| {s['model']} | {s['image_size']} px | {s['params_millions']:.2f} | {s['macs_millions']:.0f} | "
                                 f"{s['best_epoch']} | {s['val_macro_f1']:.1f} |" for s in smeta["selection"].values()))
    if S.results and S.results.get("robustness_macro_f1"):
        rob = S.results["robustness_macro_f1"]
        names = list(next(iter(rob.values())))
        parts.append("**Robustness** (test macro-F1 % under each corruption)\n\n| Method | " + " | ".join(names) + " |\n|---|"
                     + "---|" * len(names) + "\n" + "\n".join(
                         f"| {METHOD_INFO[k]['short']} | " + " | ".join(f"{v[n]:.0f}" for n in names) + " |" for k, v in rob.items()))
    return "\n\n".join(parts) or "No experiment logs yet."


def about_md() -> str:
    return """
**RA 9003** (Ecological Solid Waste Management Act of 2000) requires households to segregate waste at source. BasuraBin
looks at a photo of **one** item and says which of the three household bins it belongs to. Special/hazardous waste such as
batteries is not one of the three bins and is not covered.

**How to use it**
* *Photo*: upload a picture, take one with the webcam, or click an example from the test set.
* *Live camera*: allow camera access and hold one item inside the square. The answer is averaged over the last few frames.
* *Video*: upload a short clip (max 60 s); every 0.2 s a frame is classified and the result is drawn on the video.
* Below 55% confidence the app says **Not sure** instead of guessing. Fill most of the square, use a plain background and good light.

**The three methods** (all trained and tested on the same images and splits)
* **A. Classical**: hand-designed features (HOG edges, HSV colour and LBP texture histograms per image region) and an SVM, all written in NumPy.
* **B. Trained from scratch**: our own small CNN, learned only from our ~5,000 training photos.
* **C. Pretrained**: MobileNetV3-Large, which first learned general visual features from 1.2 million ImageNet photos, then was fine-tuned on our photos.

The recommended method is the one with the highest macro-F1 on the held-out test set (see *Model comparison*).

**Data**: Garbage Classification (Kaggle, mostafaabla) and RealWaste (UCI #908, CC BY 4.0), relabelled into the three bins.
"""


# ---- UI ------------------------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="emerald", secondary_hue="blue", neutral_hue="slate")
    with gr.Blocks(title="BasuraBin - waste segregation", theme=theme, css=CSS) as demo:
        gr.HTML(header_html())
        with gr.Tabs():
            with gr.Tab("Photo"):
                # outputs are created first (unrendered) so the examples under the input can target them
                verdict = gr.HTML(empty_html("Upload a photo, take one with the webcam, or click an example."), render=False)
                main_img = gr.Image(label="Result drawn on the photo (recommended method)", interactive=False, height=380,
                                    render=False)
                cards = gr.HTML(render=False)
                gallery = gr.Gallery(columns=3, height=320, object_fit="contain", show_label=False, render=False)
                outs = [main_img, verdict, cards, gallery]
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        inp = gr.Image(type="numpy", sources=["upload", "webcam", "clipboard"], label="Photo of ONE waste item",
                                       height=380)
                        btn = gr.Button("Which bin?", variant="primary", size="lg")
                        ex = example_images()
                        if ex:  # clicking an example classifies it straight away
                            gr.Examples(ex, inputs=inp, outputs=outs, fn=on_image, run_on_click=True, cache_examples=False,
                                        examples_per_page=12, label="Or click a test-set example (never seen during training)")
                    with gr.Column(scale=6):
                        verdict.render()
                        main_img.render()
                gr.Markdown("#### How each method saw it")
                cards.render()
                with gr.Accordion("Annotated photo from every method", open=False):
                    gallery.render()
                btn.click(on_image, inp, outs)
                inp.upload(on_image, inp, outs)

            with gr.Tab("Live camera"):
                gr.Markdown("Allow camera access, then hold **one** item inside the square. All three methods run on every "
                            "frame; each answer is averaged over the last few frames so it does not flicker.")
                with gr.Row(equal_height=False):
                    cam = gr.Image(sources=["webcam"], type="numpy", streaming=True, label="Camera", scale=4)
                    live = gr.Image(label="Result (box colour = bin of the recommended method)", interactive=False, scale=6)
                live_verdict = gr.HTML(empty_html("Start the camera to see the verdict here."))
                live_cards = gr.HTML()
                smooth_state = gr.State(None)  # one Smoother per browser session
                cam.stream(on_webcam, [cam, smooth_state], [live, live_verdict, live_cards, smooth_state], stream_every=0.15,
                           time_limit=900, concurrency_limit=1)

            with gr.Tab("Video"):
                gr.Markdown("Upload a short clip (max 60 s). A frame is classified every 0.2 s and the result is drawn on the video.")
                with gr.Row():
                    vin = gr.Video(label="Input video", sources=["upload"])
                    vout = gr.Video(label="Annotated video")
                vbtn = gr.Button("Analyse video", variant="primary", size="lg")
                vsum = gr.HTML()
                vbtn.click(on_video, vin, [vout, vsum])

            with gr.Tab("Model comparison"):
                gr.HTML(kpi_html())
                gr.Markdown(f"**Why the recommended method:** {S.explanation}")
                with gr.Accordion("Full results table", open=False):
                    md = RESULTS_DIR / "results_table.md"
                    gr.Markdown(md.read_text() if md.exists() else "No results yet.")
                figures = [(n, l) for n, l in [("confusion_matrices.png", "Confusion matrices"), ("training_curves.png", "Training curves"),
                                               ("robustness.png", "Robustness"),
                                               (f"misclassified_{S.best}.png", "Mistakes of the recommended method")]
                           if (RESULTS_DIR / n).exists()]
                if figures:
                    with gr.Tabs():
                        for name, label in figures:
                            with gr.Tab(label):
                                gr.Image(str(RESULTS_DIR / name), show_label=False, interactive=False)
                with gr.Accordion("Experiments", open=False):
                    gr.Markdown(experiments_md())

            with gr.Tab("About"):
                gr.HTML('<div class="bb-bins">' + "".join(
                    f'<div class="bb-bin" style="background:{hex_color(c)}"><h3>{DISPLAY[c]}</h3>'
                    f'<p><b>{BIN_COLOR_NAME[c].capitalize()}.</b> {EXAMPLES_TEXT[c].capitalize()}.</p></div>' for c in CLASSES) + "</div>")
                gr.Markdown(about_md())
    return demo


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--share", action="store_true", help="create a temporary public link")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--cpu", action="store_true", help="run the neural networks on the CPU")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    load_everything("cpu" if a.cpu else None)
    build_ui().queue().launch(server_name=a.host, server_port=a.port, share=a.share, inbrowser=not a.no_browser)


if __name__ == "__main__":
    main()
