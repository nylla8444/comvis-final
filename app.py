"""BasuraBin: the waste-segregation app. Runs all three methods on an uploaded image, an uploaded
video or the live webcam, draws each result on the input, and names the recommended method.

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
from common import ADVICE, CLASSES, DISPLAY, METHOD_INFO, METHOD_KEYS, RESULTS_DIR, SPLITS_DIR, load_json
from methods import load_predictors

BIN_COLOR_NAME = {"biodegradable": "green bin", "recyclable": "blue bin", "residual": "black bin"}


class State:
    preds: dict = {}
    missing: list = []
    best: str | None = None
    explanation: str = ""


S = State()


def load_everything(device=None):
    S.preds, S.missing = load_predictors(device)
    res = load_json(RESULTS_DIR / "results.json")
    if res and res["recommendation"]["best"] in S.preds:
        S.best = res["recommendation"]["best"]
        S.explanation = res["recommendation"]["explanation"]
    else:
        S.best = "pretrained" if "pretrained" in S.preds else next(iter(S.preds), None)
        S.explanation = ("_No evaluation results yet - run `python evaluate.py` to compute the test metrics. "
                         "Until then the pretrained model is used as the default._")


def verdict_md(results: dict) -> str:
    if not results:
        return "### No trained models found\nTrain the methods first (see README)."
    r = results[S.best]
    if r["conf"] < UNSURE_THRESHOLD:
        head = (f"### Not sure ({r['conf'] * 100:.0f}% confident)\nHold one item inside the box, closer to the camera, "
                f"on a plain background. Best guess: **{DISPLAY[r['label']]}**.")
    else:
        head = (f"### Put it in: **{DISPLAY[r['label']]}** ({BIN_COLOR_NAME[r['label']]})\n{ADVICE[r['label']]}\n\n"
                f"Decided by **{METHOD_INFO[S.best]['short']}** with {r['conf'] * 100:.0f}% confidence.")
    agree = len({x["label"] for x in results.values()}) == 1
    note = "All methods agree." if agree else "The methods disagree on this item; trust the recommended one (see below)."
    return f"{head}\n\n{note}\n\n---\n**Why this method:** {S.explanation}"


def table(results: dict) -> pd.DataFrame:
    rows = []
    for k in METHOD_KEYS:
        if k not in results:
            rows.append({"Method": METHOD_INFO[k]["short"], "Prediction": "not trained", "Confidence": "",
                         **{DISPLAY[c]: "" for c in CLASSES}, "Time (ms)": ""})
            continue
        r = results[k]
        rows.append({
            "Method": METHOD_INFO[k]["short"] + (" *" if k == S.best else ""),
            "Prediction": DISPLAY[r["label"]] + (" (not sure)" if r["conf"] < UNSURE_THRESHOLD else ""),
            "Confidence": f"{r['conf'] * 100:.1f}%",
            **{DISPLAY[c]: f"{r['probs'][i] * 100:.1f}%" for i, c in enumerate(CLASSES)},
            "Time (ms)": f"{r['ms']:.1f}",
        })
    return pd.DataFrame(rows)


# ---- handlers ----------------------------------------------------------------------------
def on_image(img):
    if img is None:
        return None, None, None, None, "### Upload or take a photo first."
    if not S.preds:
        return None, None, None, None, verdict_md({})
    results = run_methods(S.preds, img)
    outs = [draw_single(img, k, results[k]) if k in results else None for k in METHOD_KEYS]
    return (*outs, table(results), verdict_md(results))


def on_webcam(frame, smoother=None):
    """One streamed frame. `smoother` is per-browser-session state: predictions are averaged over the last frames."""
    if frame is None or not S.preds:
        return None, "", smoother
    smoother = smoother or Smoother()
    results = smoother.update(run_methods(S.preds, frame))
    return draw_all(frame, results, S.best), verdict_md(results), smoother


def on_video(path, analyse_fps=5.0, max_seconds=60, progress=gr.Progress()):
    import cv2
    import imageio

    if not path or not S.preds:
        return None, None
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
    rows = []
    for k in METHOD_KEYS:
        if k not in votes or not votes[k]:
            continue
        lab, n = votes[k].most_common(1)[0]
        tot = sum(votes[k].values())
        rows.append({"Method": METHOD_INFO[k]["short"] + (" *" if k == S.best else ""), "Majority bin": DISPLAY[lab],
                     "Frames with this bin": f"{n}/{tot} ({n / tot * 100:.0f}%)",
                     "Avg time (ms/frame)": f"{np.mean(times[k]):.1f}"})
    return str(out_path), pd.DataFrame(rows)


def example_images(n_per_class=2) -> list[str]:
    try:
        df = pd.read_csv(SPLITS_DIR / "test.csv")
    except FileNotFoundError:
        return []
    root = Path(__file__).resolve().parent
    picks = df.groupby("label").head(n_per_class)["path"].tolist()
    return [str(root / p) for p in picks if (root / p).exists()]


def results_tab_md() -> str:
    md = (RESULTS_DIR / "results_table.md")
    if not md.exists():
        return "No results yet. Run `python evaluate.py` after training the three methods."
    return md.read_text() + f"\n\n**Recommendation.** {S.explanation}"


# ---- UI ------------------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="BasuraBin - waste segregation", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# BasuraBin: which bin does it go in?\n"
            "Segregation at source (RA 9003): **Biodegradable** (green), **Recyclable** (blue), "
            "**Non-biodegradable / Residual** (black). Three computer-vision methods judge every input; "
            "the recommended one is marked with *."
            + ("" if not S.missing else f"\n\n> Not trained yet: {', '.join(S.missing)}")
        )
        with gr.Tab("Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    inp = gr.Image(type="numpy", sources=["upload", "webcam", "clipboard"], label="Photo of ONE waste item")
                    btn = gr.Button("Classify", variant="primary")
                with gr.Column(scale=2):
                    verdict = gr.Markdown()
                    with gr.Row():
                        outs = [gr.Image(label=f"{METHOD_INFO[k]['slot']}: {METHOD_INFO[k]['short']}") for k in METHOD_KEYS]
                    tbl = gr.Dataframe(label="All methods", interactive=False)
            ex = example_images()
            if ex:  # clicking an example classifies it straight away
                gr.Examples(ex, inputs=inp, outputs=[*outs, tbl, verdict], fn=on_image, run_on_click=True,
                            cache_examples=False, label="Test-set examples (never seen during training)")
            btn.click(on_image, inp, [*outs, tbl, verdict])
            inp.upload(on_image, inp, [*outs, tbl, verdict])

        with gr.Tab("Live webcam"):
            gr.Markdown("Allow camera access, then hold one item inside the square. All three methods run on every frame; "
                        "each method's answer is averaged over the last few frames so it does not flicker.")
            with gr.Row():
                cam = gr.Image(sources=["webcam"], type="numpy", streaming=True, label="Camera")
                live = gr.Image(label="Result (box colour = bin of the recommended method)")
            live_md = gr.Markdown()
            smooth_state = gr.State(None)  # one Smoother per browser session
            cam.stream(on_webcam, [cam, smooth_state], [live, live_md, smooth_state], stream_every=0.15, time_limit=900,
                       concurrency_limit=1)

        with gr.Tab("Video"):
            gr.Markdown("Upload a short clip (max 60 s). Frames are analysed 5 times per second.")
            with gr.Row():
                vin = gr.Video(label="Input video", sources=["upload"])
                vout = gr.Video(label="Annotated video")
            vbtn = gr.Button("Analyse video", variant="primary")
            vtbl = gr.Dataframe(label="Summary over all analysed frames", interactive=False)
            vbtn.click(on_video, vin, [vout, vtbl])

        with gr.Tab("Model comparison"):
            gr.Markdown(results_tab_md())
            for name, label in [("confusion_matrices.png", "Confusion matrices (test split)"),
                                ("training_curves.png", "Training curves"),
                                ("robustness.png", "Robustness to corruptions")]:
                if (RESULTS_DIR / name).exists():
                    gr.Image(str(RESULTS_DIR / name), label=label, interactive=False)
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
