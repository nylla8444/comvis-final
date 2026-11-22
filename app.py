"""BasuraBin: the waste-segregation app. Runs all three methods on an uploaded image, an uploaded
video or the live webcam, draws the result on the input, and names the recommended method.

    python app.py              # opens http://127.0.0.1:7860 in the browser
    python app.py --share      # also prints a temporary public gradio.live link
    python app.py --cpu        # force CPU (to show real-time speed without a GPU)

The look follows the "Bin Control Room" design (DESIGN.md): one navy answer tile per screen, light tiles around it.
"""
from __future__ import annotations

import argparse
import math
import tempfile
from collections import Counter
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

from annotate import UNSURE_THRESHOLD, Smoother, draw_all, draw_single, run_methods
from common import ADVICE, CLASSES, DISPLAY, METHOD_INFO, METHOD_KEYS, MODELS_DIR, RESULTS_DIR, SPLITS_DIR, load_json
from methods import load_predictors

BIN_COLOR_NAME = {"biodegradable": "green bin", "recyclable": "blue bin", "residual": "black bin"}
SHORT_BIN = {"biodegradable": "Biodegradable", "recyclable": "Recyclable", "residual": "Residual"}
CAST_NAME = {"biodegradable": "Nabubulok", "recyclable": "Recyclable", "residual": "Residual"}
CAST_EXAMPLES = {"biodegradable": "Fruit peel · leaves", "recyclable": "Cans · PET bottles", "residual": "Sachets · styrofoam"}
EXAMPLES_TEXT = {
    "biodegradable": ["Food scraps", "Fruit peel", "Leftover rice", "Leaves", "Garden waste"],
    "recyclable": ["Clean paper", "Cardboard", "Cans", "Glass bottles", "PET bottles"],
    "residual": ["Sachets", "Candy wrappers", "Styrofoam", "Used tissue", "Old textiles and shoes"],
}
UNSURE_ADVICE = "Hold ONE item inside the square, closer to the camera, on a plain background with good light."

# bin tones, mirrored from DESIGN.md: strong (marks), tint (fills), ink (text on the tint), on-hero (marks on navy)
STRONG = {"biodegradable": "#15924B", "recyclable": "#2F6BFF", "residual": "#262A33", "unsure": "#F2A516"}
TINT = {"biodegradable": "#DCF4E5", "recyclable": "#E1EAFF", "residual": "#E4E6EB", "unsure": "#FFF0D2"}
INK = {"biodegradable": "#0D4D29", "recyclable": "#17368F", "residual": "#1A1D24", "unsure": "#6E4300"}
ON_HERO = {"biodegradable": "#63E096", "recyclable": "#86A8FF", "residual": "#D6DAE3", "unsure": "#FFC552"}
LID = {"biodegradable": "#0F7A3D", "recyclable": "#1F55D9", "residual": "#3B404C"}
BIN_ICON = {"biodegradable": "leaf", "recyclable": "recycle", "residual": "bin"}
METHOD_COLOR = {"pretrained": "#151A45", "scratch": "#4A5177", "classical": "#8A90A7"}

# Lucide icon paths (ISC licence)
ICONS = {
    "bin": '<path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/>'
           '<path d="M10 11v6"/><path d="M14 11v6"/>',
    "leaf": '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/>'
            '<path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>',
    "recycle": '<path d="M7 19H4.815a1.83 1.83 0 0 1-1.57-.881 1.785 1.785 0 0 1-.004-1.784L7.196 9.5"/>'
               '<path d="M11 19h8.203a1.83 1.83 0 0 0 1.556-.89 1.784 1.784 0 0 0 0-1.775l-1.226-2.12"/><path d="m14 16-3 3 3 3"/>'
               '<path d="M8.293 13.596 7.196 9.5 3.1 10.598"/><path d="m9.344 5.811 1.093-1.892A1.83 1.83 0 0 1 11.985 3a1.784 1.784 0 0 1 1.546.888l3.943 6.843"/>'
               '<path d="m13.378 9.633 4.096 1.098 1.097-4.096"/>',
    "check": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "image": '<rect width="18" height="18" x="3" y="3" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.09-3.09a2 2 0 0 0-2.82 0L6 21"/>',
    "camera": '<path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/>',
    "video": '<path d="m16 13 5.22 3.48a.5.5 0 0 0 .78-.42V7.87a.5.5 0 0 0-.75-.43L16 10.5"/><rect x="2" y="6" width="14" height="12" rx="2"/>',
    "chart": '<path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    "cpu": '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/>'
           '<path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>',
    "book": '<path d="M12 7v14"/><path d="M3 18a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h5a4 4 0 0 1 4 4 4 4 0 0 1 4-4h5a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-6a3 3 0 0 0-3 3 3 3 0 0 0-3-3z"/>',
    "scan": '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><path d="M7 12h10"/>',
}


def icon(name: str, cls: str = "bb-ico") -> str:
    return (f'<svg class="{cls}" aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>')


def buddy(c: str, size: int = 120) -> str:
    """The bin's mascot (one per bin); decorative, so always paired with the bin's name in text."""
    return (f'<svg class="bb-buddy" width="{size}" height="{round(size * 1.2)}" viewBox="0 0 120 144" fill="none" aria-hidden="true">'
            '<ellipse cx="60" cy="137" rx="38" ry="5" fill="#0E1232" fill-opacity=".14"/>'
            f'<path d="M24 44H96L88.5 124.5C88 129.6 83.7 133.5 78.6 133.5H41.4C36.3 133.5 32 129.6 31.5 124.5L24 44Z" fill="{STRONG[c]}" '
            'stroke="#fff" stroke-width="4" stroke-linejoin="round"/>'
            f'<rect x="46" y="14" width="28" height="14" rx="7" fill="{LID[c]}" stroke="#fff" stroke-width="4"/>'
            f'<rect x="14" y="26" width="92" height="20" rx="10" fill="{LID[c]}" stroke="#fff" stroke-width="4"/>'
            '<circle cx="45" cy="72" r="9" fill="#fff"/><circle cx="75" cy="72" r="9" fill="#fff"/>'
            # pupil + highlight move together (LOAD_JS); the resting look is slightly down-right
            + "".join(f'<g class="bb-eye" transform="translate(1.5 1.5)"><circle cx="{x}" cy="72" r="5" fill="#0E1232"/>'
                      f'<circle cx="{x + 1.5}" cy="70" r="1.6" fill="#fff"/></g>' for x in (45, 75)) +
            '<circle cx="34" cy="86" r="5" fill="#FF9DB0" fill-opacity=".6"/><circle cx="86" cy="86" r="5" fill="#FF9DB0" fill-opacity=".6"/>'
            '<path d="M53 86C56.5 90 63.5 90 67 86" stroke="#fff" stroke-width="4" stroke-linecap="round"/>'
            '<g transform="translate(46 98) scale(1.15)" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
            f'{ICONS[BIN_ICON[c]]}</g></svg>')


def donut(shares: dict, value: str, caption: str, unsure: bool = False, size: int = 280) -> str:
    """Round-capped share ring, one segment per bin, with each bin's icon + share in a chip just outside the ring."""
    pad = 50
    vb, r, sw, gap = size + 2 * pad, size / 2 - 22, 30, 24
    cx = cy = vb / 2
    avail = 360 - gap * len(CLASSES)

    def pt(a, rr):
        return cx + rr * math.sin(math.radians(a)), cy - rr * math.cos(math.radians(a))

    arcs, chips, a = [], [], gap / 2
    for c in CLASSES:
        p = float(shares.get(c, 0.0))
        sweep = max(0.4, p * avail)  # tiny shares still show as a dot
        (x0, y0), (x1, y1) = pt(a, r), pt(a + sweep, r)
        d = f"M{x0:.2f} {y0:.2f} A{r} {r} 0 {1 if sweep > 180 else 0} 1 {x1:.2f} {y1:.2f}"
        if c == "residual":  # the black bin gets a light outline so it reads on navy
            arcs.append(f'<path d="{d}" stroke="#D6DAE3" stroke-width="{sw + 4}" stroke-linecap="round"/>')
        arcs.append(f'<path d="{d}" stroke="{STRONG[c]}" stroke-width="{sw}" stroke-linecap="round"/>')
        mx, my = pt(a + sweep / 2, size / 2 + 25)
        chips.append(f'<g transform="translate({mx:.1f} {my:.1f})"><circle r="23" fill="#1F2766" stroke="#151A45" stroke-width="3"/>'
                     f'<g transform="translate(-7.5 -14) scale(.625)" stroke="{ON_HERO[c]}" stroke-width="3.2" stroke-linecap="round" '
                     f'stroke-linejoin="round">{ICONS[BIN_ICON[c]]}</g>'
                     f'<text y="14" text-anchor="middle" class="bb-chip-txt">{p * 100:.0f}%</text></g>')
        a += sweep + gap
    label = ", ".join(f"{SHORT_BIN[c]} {float(shares.get(c, 0)) * 100:.0f}%" for c in CLASSES)
    return (f'<svg class="bb-donut" viewBox="0 0 {vb} {vb}" fill="none" role="img" aria-label="{label}">'
            f'<circle cx="{cx}" cy="{cy}" r="{r}" stroke="#1F2766" stroke-width="{sw + 14}"/>{"".join(arcs)}{"".join(chips)}'
            f'<text x="{cx}" y="{cy + 8}" text-anchor="middle" class="bb-donut-val" fill="{ON_HERO["unsure"] if unsure else "#fff"}">{value}</text>'
            f'<text x="{cx}" y="{cy + 34}" text-anchor="middle" class="bb-donut-cap">{caption}</text></svg>')


def state_pill(kind: str, text: str) -> str:
    return f'<span class="bb-state is-{kind}">{icon("check" if kind == "agree" else "alert")}{text}</span>'


def hero(head: str, body: str = "", side: str = "", foot: str = "", badge: str = "image", label: str = "Answer",
         idle: bool = False) -> str:
    """Navy answer tile: the head (title + lead) spans the top, then body | side, then foot. The notch holds the tab's icon."""
    parts = f'<div class="bb-hero-head">{head}</div>'
    parts += f'<div class="bb-hero-main">{body}</div>' if body else ""
    parts += f'<div class="bb-hero-side">{side}</div>' if side else ""
    parts += f'<div class="bb-hero-foot">{foot}</div>' if foot else ""
    cls = (" is-idle" if idle else "") + ("" if side else " no-side")
    return (f'<section class="bb bb-hero{cls}" aria-label="{label}"><span class="bb-notch" aria-hidden="true">{icon(badge)}</span>'
            f'<div class="bb-hero-body"><div class="bb-hero-surface">{parts}</div></div></section>')


def cast_html(dim: bool = False) -> str:
    items = "".join(f'<figure class="bb-cast-item">{buddy(c, 96)}<figcaption><b>{CAST_NAME[c]}</b><span>{CAST_EXAMPLES[c]}</span>'
                    f'</figcaption></figure>' for c in CLASSES)
    return f'<div class="bb-cast{" is-dim" if dim else ""}">{items}</div>'


def idle_hero(title: str, lead: str, badge: str, note: str = "") -> str:
    head = f'<h2 class="bb-verdict is-sentence">{title}</h2><p class="bb-lead">{lead}</p>'
    return hero(head, "", cast_html(), f'<p class="bb-note">{note}</p>' if note else "", badge, "Waiting for input", idle=True)


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


def letter(k: str) -> str:
    return METHOD_INFO[k]["slot"][0]


METHOD_META = {"classical": "HOG + colour + LBP → SVM", "scratch": "Trained from scratch", "pretrained": "Pretrained on ImageNet"}


def method_meta(k: str) -> str:
    return METHOD_META.get(k, METHOD_INFO[k]["slot"].split(". ", 1)[-1])


def others() -> list[str]:
    """Non-deciding methods, C/B/A order."""
    return [k for k in reversed(METHOD_KEYS) if k != S.best]


# ---- shell: rail and top bar ---------------------------------------------------------------------
def rail_html() -> str:
    rows = "".join(f'<li class="{"is-ready" if k in S.preds else "is-missing"}"><span class="bb-letter">{letter(k)}</span>'
                   f'<span>{METHOD_INFO[k]["short"].split(" (")[0]}</span><span class="bb-sr">'
                   f'{"loaded" if k in S.preds else "not trained yet"}</span></li>' for k in reversed(METHOD_KEYS))
    return (f'<nav class="bb bb-rail" aria-label="System status"><span class="bb-logo">{icon("bin")}</span>'
            f'<p class="bb-rail-label">Methods</p><ul class="bb-rail-methods">{rows}</ul>'
            f'<div class="bb-rail-device">{icon("cpu")}<span>{S.device.split(" (")[0]}</span></div></nav>')


def topbar_html() -> str:
    n = len(S.preds)
    return (f'<header class="bb bb-top"><div><h1 class="bb-wordmark">BasuraBin</h1>'
            f'<p class="bb-sub">Which bin does it go in? RA 9003 household segregation</p></div>'
            f'<span class="bb-status-pill"><span class="bb-dot{"" if n == len(METHOD_KEYS) else " is-warn"}"></span>'
            f'{n} of {len(METHOD_KEYS)} methods · {S.device.split(" (")[0]}</span></header>')


# ---- answers ----------------------------------------------------------------------------------------
def verdict_html(results: dict, badge: str = "image") -> str:
    if not results or S.best not in results:
        return idle_hero("No trained models found", "Train the three methods first (see README), then restart the app.", badge)
    r = results[S.best]
    label, conf = r["label"], r["conf"] * 100
    unsure = r["conf"] < UNSURE_THRESHOLD
    shares = dict(zip(CLASSES, r["probs"]))
    if unsure:
        head = f'<h2 class="bb-verdict">Not sure</h2><p class="bb-lead is-unsure">Leaning {SHORT_BIN[label].lower()} · {conf:.0f}%</p>'
        body = f'<p class="bb-bubble is-solo">{UNSURE_ADVICE}</p>{cast_html(dim=True)}'
    else:
        head = f'<h2 class="bb-verdict">{SHORT_BIN[label]}</h2><p class="bb-lead">{BIN_COLOR_NAME[label].capitalize()}</p>'
        body = f'<div class="bb-say">{buddy(label)}<p class="bb-bubble">{ADVICE[label]}</p></div>'
    pills = state_pill("unsure", "Not sure") if unsure else ""
    if len(results) > 1:
        agree = len({x["label"] for x in results.values()}) == 1
        pills += state_pill("agree", f"{len(results)} of {len(results)} agree") if agree else state_pill("disagree", "Methods disagree")
    foot = f'{pills}<span class="bb-decided">Decided by {METHOD_INFO[S.best]["short"]} · {r["ms"]:.0f} ms</span>'
    side = donut(shares, f"{conf:.0f}%", "not sure" if unsure else "sure", unsure)
    return hero(head, body, side, foot, badge, f"Answer: {'not sure' if unsure else SHORT_BIN[label]}")


def meter(c: str, value: float, text: str) -> str:
    return (f'<div class="bb-meter"><span class="bb-chip" style="background:{TINT[c]}; color:{STRONG[c]}">{icon(BIN_ICON[c])}</span>'
            f'<span class="bb-meter-name">{SHORT_BIN[c]}</span><span class="bb-bar" aria-hidden="true"><i style="width:{value * 100:.1f}%; '
            f'background:{STRONG[c]}"></i></span><b>{text}</b></div>')


def mini_bar(shares) -> str:
    a, b = float(shares[0]) * 100, float(shares[0] + shares[1]) * 100
    g = (f"linear-gradient(90deg,{STRONG['biodegradable']} 0 {a:.1f}%,{STRONG['recyclable']} {a:.1f}% {b:.1f}%,"
         f"{STRONG['residual']} {b:.1f}% 100%)")
    return f'<span class="bb-mini" style="background:{g}" aria-hidden="true"></span>'


def methods_html(rows: dict, hint: str) -> str:
    """rows: {method: {"verdict", "meta", "bin" (or None), "values" [3 floats 0..1], "texts" [3 str]}}; deciding tile + Also checked."""
    if S.best not in rows:
        return ""
    d = rows[S.best]
    call = (f'<div class="bb-call">{buddy(d["bin"], 44) if d["bin"] else ""}<b>{d["verdict"]}</b></div>')
    meters = "".join(meter(c, d["values"][i], d["texts"][i]) for i, c in enumerate(CLASSES))
    decider = (f'<article class="bb-tile bb-decider"><header><span class="bb-letter is-dark">{letter(S.best)}</span>'
               f'<div><h3>{METHOD_INFO[S.best]["short"].split(" (")[0]}</h3><p>{d["meta"]}</p></div>'
               f'<span class="bb-decides">{icon("check")}Decides</span></header>{call}<div class="bb-meters">{meters}</div></article>')
    also = ""
    for k in others():
        r = rows.get(k)
        verdict = r["verdict"] if r else "Not trained yet"
        bar = mini_bar(r["values"]) if r and r.get("bar", True) else ""
        also += (f'<div class="bb-row"><span class="bb-letter">{letter(k)}</span><div class="bb-row-name"><b>'
                 f'{METHOD_INFO[k]["short"].split(" (")[0]}</b><span>{r["meta"] if r else method_meta(k)}</span></div>'
                 f'<b class="bb-row-verdict">{verdict}</b>{bar}</div>')
    also = f'<article class="bb-tile bb-also"><header><h3>Also checked</h3><p>{hint}</p></header>{also}</article>'
    return f'<div class="bb bb-methods">{decider}{also}</div>'


def photo_rows(results: dict) -> dict:
    rows = {}
    for k, r in results.items():
        unsure = r["conf"] < UNSURE_THRESHOLD
        name = f"{SHORT_BIN[r['label']]} {r['conf'] * 100:.0f}%"
        rows[k] = {"verdict": f"Not sure · {name}" if unsure else f"{SHORT_BIN[r['label']]} · {r['conf'] * 100:.0f}%",
                   "meta": f"{method_meta(k)} · {r['ms']:.0f} ms", "bin": None if unsure else r["label"],
                   "values": list(r["probs"]), "texts": [f"{p * 100:.0f}%" for p in r["probs"]]}
    return rows


def idle_rows() -> dict:
    """Before any input: each method's real held-out test scores."""
    if not S.results:
        return {}
    rows = {}
    for k, r in S.results["results"].items():
        if k not in S.preds:
            continue
        pc = [r["test"]["per_class"][c]["f1"] / 100 for c in CLASSES]
        rows[k] = {"verdict": f"Ready · {r['test']['macro_f1']:.1f}% test macro-F1" if k == S.best else f"Ready · {r['test']['macro_f1']:.1f}%",
                   "meta": f"{method_meta(k)} · F1 per bin" if k == S.best else method_meta(k), "bin": None,
                   "values": pc, "texts": [f"{v * 100:.1f}" for v in pc], "bar": False}
    return rows


IDLE_HINT = "Held-out test macro-F1 · 1021 images"


# ---- handlers ----------------------------------------------------------------------------------
def on_image(img):
    """Returns (drawn image, answer, methods, input image, Original/Drawn switch)."""
    if img is None:
        return (gr.update(value=None, visible=False), idle_hero(*PHOTO_IDLE), methods_html(idle_rows(), IDLE_HINT),
                gr.update(visible=True), gr.update(value="Original", visible=False))
    if not S.preds:
        return gr.update(value=None, visible=False), verdict_html({}), "", gr.update(visible=True), gr.update(visible=False)
    results = run_methods(S.preds, img)
    main = draw_single(img, S.best, results[S.best], max_side=900)
    return (gr.update(value=main, visible=True), verdict_html(results, "image"),
            methods_html(photo_rows(results), "Not deciding; shown so you can see if they agree"),
            gr.update(visible=False), gr.update(value="Drawn", visible=True))


def on_switch(choice):
    """Original shows the input (to change the photo); Drawn shows the result drawn on it."""
    return gr.update(visible=choice == "Original"), gr.update(visible=choice == "Drawn")


def on_webcam(frame, smoother=None):
    """One streamed frame. `smoother` is per-browser-session state: predictions are averaged over the last frames."""
    if frame is None or not S.preds:
        return None, idle_hero(*LIVE_IDLE), methods_html(idle_rows(), IDLE_HINT), smoother
    smoother = smoother or Smoother()
    results = smoother.update(run_methods(S.preds, frame))
    return (draw_all(frame, results, S.best), verdict_html(results, "camera"),
            methods_html(photo_rows(results), "Below 55% every method says Not sure"), smoother)


def frames_html(seqs: dict) -> str:
    """One bar per analysed frame and method, coloured by its bin."""
    rows = ""
    for k in [S.best] + others():
        if not seqs.get(k):
            continue
        bars = "".join(f'<i style="background:{STRONG[c]}"></i>' for c in seqs[k])
        counts = Counter(seqs[k])
        label = ", ".join(f"{SHORT_BIN[c]} {counts[c]}" for c in CLASSES if counts[c])
        rows += (f'<div class="bb-frames-row"><b>{letter(k)} · {METHOD_INFO[k]["short"].split(" (")[0]}</b>'
                 f'<span class="bb-frames" role="img" aria-label="{label} frames">{bars}</span></div>')
    return (f'<section class="bb bb-tile bb-frames-tile"><header><h3>Frame by frame</h3>'
            f'<p>One bar per analysed frame, every 0.2 s</p></header>{rows}</section>')


def on_video(path, analyse_fps=5.0, max_seconds=60, progress=gr.Progress()):
    """Returns (drawn video, input video, Original/Drawn switch, answer, frames + methods)."""
    import cv2
    import imageio

    hidden = gr.update(value=None, visible=False)
    if not path or not S.preds:
        return (hidden, gr.update(visible=True), gr.update(visible=False),
                idle_hero("Upload a short video first", "Up to 60 seconds, one item in view at a time.", "video"), "")
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    stride = max(1, int(round(src_fps / analyse_fps)))
    out_path = Path(tempfile.mkdtemp()) / "annotated.mp4"
    writer = imageio.get_writer(out_path, fps=src_fps / stride, codec="libx264", pixelformat="yuv420p", macro_block_size=2)
    seqs = {k: [] for k in S.preds}
    times = {k: [] for k in S.preds}
    smoother = Smoother(alpha=0.5, reset_after_s=None)  # analysed frames are 0.2 s apart: average over ~3 of them
    i = 0
    max_frames = int(max_seconds * src_fps)
    while i < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            results = smoother.update(run_methods(S.preds, rgb))
            for k, r in results.items():
                seqs[k].append(r["label"])
                times[k].append(r["ms"])
            frame = draw_all(rgb, results, S.best)
            frame = frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]
            writer.append_data(frame)
            if n_total:
                progress(min(i / min(n_total, max_frames), 1.0), desc="analysing video")
        i += 1
    cap.release()
    writer.close()
    if not seqs.get(S.best):
        return (hidden, gr.update(visible=True), gr.update(visible=False),
                idle_hero("No frames could be read", "Try another file (MP4 works best).", "video"), "")
    rows, majority = {}, {}
    for k, seq in seqs.items():
        counts = Counter(seq)
        shares = [counts[c] / len(seq) for c in CLASSES]
        majority[k] = lab = counts.most_common(1)[0][0]
        rows[k] = {"verdict": f"Mostly {SHORT_BIN[lab].lower()} · {counts[lab] / len(seq) * 100:.0f}%",
                   "meta": f"{len(seq)} frames · {np.mean(times[k]):.0f} ms each", "bin": lab,
                   "values": shares, "texts": [f"{s * 100:.0f}%" for s in shares]}
    lab = majority[S.best]
    counts = Counter(seqs[S.best])
    n, tot = counts[lab], len(seqs[S.best])
    vhead = f'<h2 class="bb-verdict">{SHORT_BIN[lab]}</h2><p class="bb-lead">{n} of {tot} frames · {BIN_COLOR_NAME[lab]}</p>'
    body = f'<div class="bb-say">{buddy(lab)}<p class="bb-bubble">{ADVICE[lab]}</p></div>'
    agree = len(set(majority.values())) == 1
    foot = ((state_pill("agree", f"{len(majority)} of {len(majority)} agree") if agree else state_pill("disagree", "Methods disagree"))
            + f'<span class="bb-decided">Decided by {METHOD_INFO[S.best]["short"]} · a frame every 0.2 s</span>')
    head = hero(vhead, body, donut({c: counts[c] / tot for c in CLASSES}, f"{n / tot * 100:.0f}%", "of frames"), foot, "video",
                f"Answer: {SHORT_BIN[lab]}")
    detail = frames_html(seqs) + methods_html(rows, f"Share of the {tot} analysed frames")
    return gr.update(value=str(out_path), visible=True), gr.update(visible=False), gr.update(value="Drawn", visible=True), head, detail


def on_video_switch(choice):
    return gr.update(visible=choice == "Original"), gr.update(visible=choice == "Drawn")


def example_images(n_per_class=3) -> list[str]:
    try:
        df = pd.read_csv(SPLITS_DIR / "test.csv")
    except FileNotFoundError:
        return []
    root = Path(__file__).resolve().parent
    picks = df.sample(frac=1.0, random_state=3).groupby("label").head(n_per_class).sort_values("label")["path"].tolist()
    return [str(root / p) for p in picks if (root / p).exists()]


# ---- compare tab ----------------------------------------------------------------------------------
def line_chart(vals: list, color: str, w: int = 136, h: int = 64) -> str:
    pts = " ".join(f"{i / (len(vals) - 1) * (w - 6) + 3:.1f},{h - 3 - (v - 50) / 50 * (h - 6):.1f}" for i, v in enumerate(vals))
    lx, ly = pts.split(" ")[-1].split(",")
    return (f'<svg class="bb-curve" viewBox="0 0 {w} {h}" fill="none" aria-hidden="true"><line x1="0" y1="{h - 3}" x2="{w}" y2="{h - 3}" stroke="#D2D9E4"/>'
            f'<polyline points="{pts}" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
            f'<circle cx="{lx}" cy="{ly}" r="4" fill="{color}"/></svg>')


def compare_html() -> str:
    if not S.results:
        return idle_hero("No results yet", "Run python evaluate.py after training the three methods.", "chart")
    res, best = S.results["results"], S.best
    order = sorted(res, key=lambda k: -res[k]["test"]["macro_f1"])
    n_test = S.results.get("test_images", "?")
    b = res[best]
    bars = "".join(f'<div class="bb-f1{" is-best" if k == best else ""}"><span>{letter(k)} · {METHOD_INFO[k]["short"]}</span>'
                   f'<b>{res[k]["test"]["macro_f1"]:.1f}%</b><span class="bb-f1-track" aria-hidden="true"><i style="width:{res[k]["test"]["macro_f1"]:.1f}%"></i></span></div>'
                   for k in order)
    chead = (f'<h2 class="bb-verdict is-sentence">{METHOD_INFO[best]["short"]} decides</h2>'
             f'<p class="bb-lead">Highest macro-F1 on {n_test} held-out test images, and {b["runtime_cpu_ms"]:.0f} ms per photo on a CPU: '
             f'fast enough for the live camera.</p>')
    top = hero(chead, f'<div class="bb-f1s">{bars}</div>', "",
               '<p class="bb-note">Macro-F1 weighs the three bins equally · same images and splits for all three</p>', "chart",
               "Which method decides")
    # F1 by bin
    head = "".join(f'<th scope="col"><span class="bb-th">{icon(BIN_ICON[c])}{CAST_NAME[c]}</span></th>' for c in CLASSES)
    body = ""
    for k in order:
        f1 = [res[k]["test"]["per_class"][c]["f1"] for c in CLASSES]
        best_col = [max(res[m]["test"]["per_class"][c]["f1"] for m in res) for c in CLASSES]
        body += f'<tr><th scope="row">{letter(k)} · {METHOD_INFO[k]["short"].split(" (")[0]}</th>' + "".join(
            f'<td class="{"is-best" if v == best_col[i] else ""}">{v:.1f}</td>' for i, v in enumerate(f1)) + "</tr>"
    support = "".join(f'<td>{res[best]["test"]["per_class"][c]["support"]}</td>' for c in CLASSES)
    f1_tile = (f'<section class="bb-tile"><h3>F1 by bin</h3><table class="bb-table"><thead><tr><th></th>{head}</tr></thead>'
               f'<tbody>{body}<tr class="is-quiet"><th scope="row">Test images</th>{support}</tr></tbody></table>'
               '<p class="bb-hint">Best in each bin in heavier weight. F1 in %.</p></section>')
    # training curves
    meta_files = {"pretrained": "pretrained_meta.json", "scratch": "scratch_cnn_meta.json", "classical": "classical_meta.json"}
    curves = ""
    for k in order:
        hist = (load_json(MODELS_DIR / meta_files[k]) or {}).get("history", {}).get("val_f1") or []
        if len(hist) > 1:
            curves += (f'<div class="bb-curve-box"><b>{letter(k)} · {METHOD_INFO[k]["short"].split(" (")[0]}</b>'
                       f'{line_chart(hist[:: max(1, len(hist) // 40)], METHOD_COLOR[k])}'
                       f'<span>{len(hist)} epochs · {hist[0]:.1f} → {hist[-1]:.1f}</span></div>')
    curve_tile = (f'<section class="bb-tile"><header><h3>How training went</h3><p>Validation F1 per epoch, 50–100%</p></header>'
                  f'<div class="bb-curves">{curves}</div></section>') if curves else ""
    # speed and size
    def fmt(v, unit, d=1):
        return f"{v:.{d}f} {unit}" if v else "—"
    spec = [("CPU per photo", lambda r: fmt(r.get("runtime_cpu_ms"), "ms")), ("GPU per photo", lambda r: fmt(r.get("runtime_gpu_ms"), "ms")),
            ("Parameters", lambda r: fmt(r.get("params_millions"), "M", 2)), ("Compute (MACs)", lambda r: fmt(r.get("macs_millions"), "M", 0)),
            ("Training time", lambda r: fmt(r.get("train_time_min"), "min"))]
    srows = "".join(f'<tr><th scope="row">{lab}</th>' + "".join(f"<td>{f(res[k])}</td>" for k in order) + "</tr>" for lab, f in spec)
    speed = (f'<section class="bb-tile"><h3>Speed and size</h3><table class="bb-table"><thead><tr><th></th>'
             + "".join(f'<th scope="col">{letter(k)}</th>' for k in order) + f'</tr></thead><tbody>{srows}</tbody></table></section>')
    # robustness
    rob = S.results.get("robustness_macro_f1") or {}
    rob_tile = ""
    if rob:
        conds = list(next(iter(rob.values())))
        groups = "".join(f'<div class="bb-group"><div class="bb-group-bars">' + "".join(
            f'<span style="--h:{rob[k][c]:.0f}%; --c:{METHOD_COLOR[k]}"><b>{rob[k][c]:.0f}</b></span>' for k in order if k in rob)
            + f'</div><span class="bb-group-label">{c.capitalize()}</span></div>' for c in conds)
        key = "".join(f'<span><i style="background:{METHOD_COLOR[k]}"></i>{letter(k)}</span>' for k in order)
        rob_tile = (f'<section class="bb-tile"><header><h3>Under pressure</h3><p class="bb-key">{key}</p></header>'
                    f'<p class="bb-hint">Macro-F1 on a random sample of test images when photos are blurred, darkened, brightened, noisy or rotated</p>'
                    f'<div class="bb-groups" role="img" aria-label="Robustness macro-F1 per condition">{groups}</div></section>')
    # confusion matrices
    grids = ""
    for k in order:
        cm = res[k]["test"].get("confusion_matrix")
        if not cm:
            continue
        cols = "".join(f'<span class="bb-cm-head">{icon(BIN_ICON[c])}<span class="bb-sr">predicted {SHORT_BIN[c]}</span></span>' for c in CLASSES)
        rows = ""
        for ri, c in enumerate(CLASSES):
            tot = sum(cm[ri]) or 1
            cells = ""
            for v in cm[ri]:
                s = v / tot
                bg = tuple(round(a + (b2 - a) * max(.04, s)) for a, b2 in zip((238, 242, 246), (21, 26, 69)))
                cells += (f'<span class="bb-cm-cell{" is-dark" if s >= .45 else ""}" style="background:rgb{bg}">'
                          f'<b>{v}</b><small>{s * 100:.0f}%</small></span>')
            rows += f'<span class="bb-cm-row-label" style="color:{STRONG[c]}">{icon(BIN_ICON[c])}<span>{CAST_NAME[c]}</span></span>{cells}'
        grids += (f'<div class="bb-cm"><p><b>{letter(k)} · {METHOD_INFO[k]["short"].split(" (")[0]}</b> accuracy {res[k]["test"]["accuracy"]:.1f}%</p>'
                  f'<div class="bb-cm-grid"><span></span>{cols}{rows}</div></div>')
    cm_tile = (f'<section class="bb-tile"><header><h3>Where each method goes wrong</h3><p>Rows: true bin · columns: predicted bin · '
               f'counts and row share</p></header><div class="bb-cms">{grids}</div></section>')
    return (f'<div class="bb bb-compare"><div class="bb-bento is-hero">{top}<div class="bb-stack">{f1_tile}{curve_tile}</div></div>'
            f'<div class="bb-bento is-split">{speed}{rob_tile}</div>{cm_tile}</div>')


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


# ---- about tab ------------------------------------------------------------------------------------
def about_html() -> str:
    top = hero('<h2 class="bb-verdict is-sentence">Sort it at the source.</h2><p class="bb-lead">RA 9003, the Ecological Solid Waste '
               'Management Act of 2000, asks every household to separate its waste into three bins. BasuraBin looks at a photo of '
               'one item and names its bin.</p>', "", cast_html(), "", "book", "About BasuraBin", idle=True)
    rules = "".join(f'<li><span class="bb-rule-ico{" is-warn" if warn else ""}">{icon(ic)}</span><div><b>{t}</b><p>{d}</p></div></li>' for ic, t, d, warn in [
        ("scan", "One item per photo", "The models look at the centre square. Fill it with the item, on a plain background with good light.", False),
        ("alert", "“Not sure” below 55%", "Below 55% every method says “Not sure”, and the app's answer follows the deciding method, "
                                          "so it says so instead of guessing.", True),
        ("alert", "Not covered: hazardous waste", "Batteries and other special or hazardous waste are not one of the three bins.", True)])
    rules_tile = f'<section class="bb-tile"><h3>Before you start</h3><ul class="bb-rules">{rules}</ul></section>'
    where = {"biodegradable": "Green bin · compost pit", "recyclable": "Blue bin · junk shop", "residual": "Black bin · disposal"}
    bins = "".join(f'<section class="bb-tile bb-bin" style="background:{TINT[c]}; color:{INK[c]}"><header>{buddy(c, 64)}<div><h3>{DISPLAY[c]}</h3>'
                   f'<p>{where[c]}</p></div></header><ul class="bb-examples">' + "".join(f"<li>{e}</li>" for e in EXAMPLES_TEXT[c])
                   + "</ul></section>" for c in CLASSES)
    steps = "".join(f'<li><span class="bb-letter{" is-dark" if k == S.best else ""}">{letter(k)}</span><div><b>{t}</b><p>{d}</p></div></li>' for k, t, d in [
        ("classical", "Classical", "Hand-designed features (HOG edges, HSV colour and LBP texture per image region) and an SVM, written in NumPy."),
        ("scratch", "Trained from scratch", "Our own small CNN, learned only from about 5,000 training photos."),
        ("pretrained", "Pretrained", "MobileNetV3-Large learned general visual features from 1.2 million ImageNet photos, then was fine-tuned on ours.")])
    how = (f'<section class="bb-tile"><h3>How it decides</h3><p>Three methods judge every item. All three were trained and tested on the same '
           f'images and splits; the one with the highest held-out macro-F1 makes the call.</p><ul class="bb-steps">{steps}</ul></section>')
    data = ('<section class="bb-tile"><h3>Data</h3><p>Garbage Classification (Kaggle, mostafaabla) and RealWaste (UCI #908, CC BY 4.0), '
            'relabelled into the three bins.</p><p>1021 held-out test images: 275 biodegradable, 372 recyclable, 374 residual.</p></section>')
    return (f'<div class="bb bb-about"><div class="bb-bento is-hero">{top}{rules_tile}</div><div class="bb-bento is-three">{bins}</div>'
            f'<div class="bb-bento is-wide">{how}{data}</div></div>')


PHOTO_IDLE = ("Which bin does it go in?", "Add a photo of one item. Three computer-vision methods judge it, and the most accurate one "
              "makes the call.", "image", "Below 55% it says “Not sure” instead of guessing.")
LIVE_IDLE = ("Hold one item to the camera", "Start the camera on the right. The answer updates live and is averaged over the last "
             "few frames, so it does not flicker.", "camera", "Below 55% it says “Not sure” instead of guessing.")
VIDEO_IDLE = ("Upload a short clip", "Up to 60 seconds, one item in view at a time. A frame is classified every 0.2 s.", "video",
              "The answer is the bin for most of the clip.")


# ---- styles --------------------------------------------------------------------------------------
CSS = """
/* Bin Control Room (DESIGN.md). Gradio wraps gr.HTML in .prose, whose "*" rule recolours everything: inside .bb, colour
   inherits instead, and every coloured element sets its own colour with a second class. */
.bb * {color: inherit}
.bb h1, .bb h2, .bb h3, .bb p, .bb ul, .bb li, .bb figure, .bb table {margin: 0; padding: 0}
.bb ul {list-style: none}
.bb {font-variant-numeric: tabular-nums; color: #121633}
.bb-ico {width: 1.15em; height: 1.15em; flex: none}
.bb-sr {position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap}
::selection {background: #DDF79A; color: #0E1232}
:focus-visible {outline: 3px solid #2F6BFF !important; outline-offset: 2px}

/* shell. HTML blocks have no visible box, so no rounded clipping: it would cut text and shadows at their edges */
.block:has(> .html-container) {overflow: visible !important; border-radius: 0 !important}
/* fluid, not media queries: Gradio scopes media-query rules under .contain, so they never reach these two ancestors */
.gradio-container {max-width: 1488px !important; padding: clamp(12px, 3vw, 24px) !important}
main.fillable.app {padding-left: clamp(0px, calc(8vw - 30px), 32px) !important; padding-right: clamp(0px, calc(8vw - 30px), 32px) !important}
.bb-shell {gap: 24px !important; align-items: stretch !important}
.bb-rail-col {flex: 0 0 88px !important; min-width: 88px !important}
.bb.bb-rail {position: sticky; top: 24px; height: calc(100vh - 48px); min-height: 520px; display: flex; flex-direction: column;
  align-items: center; gap: 14px; padding: 20px 0; border-radius: 28px; background: #fff;
  box-shadow: 0 1px 2px rgba(18,22,51,.04), 0 8px 24px -8px rgba(14,18,50,.08)}
.bb-logo {display: grid; place-items: center; width: 52px; height: 52px; border-radius: 16px; background: #0E1232; color: #C6F16D}
.bb-logo .bb-ico {width: 26px; height: 26px}
.bb .bb-rail-label {margin-top: 18px; font-size: 13px; font-weight: 600; color: #525A75}
.bb-rail-methods {display: flex; flex-direction: column; gap: 14px; align-items: center}
.bb-rail-methods li {display: flex; flex-direction: column; align-items: center; gap: 4px; font-size: 11px; font-weight: 600; color: #525A75;
  text-align: center; max-width: 80px}
.bb-rail-methods .bb-letter {position: relative}
.bb-rail-methods li.is-ready .bb-letter::after, .bb-rail-methods li.is-missing .bb-letter::after {content: ""; position: absolute; top: 0; right: 0;
  width: 10px; height: 10px; border-radius: 50%; border: 2px solid #fff; background: #15924B}
.bb-rail-methods li.is-missing .bb-letter::after {background: #8A90A7}
.bb .bb-rail-device {margin-top: auto; display: flex; flex-direction: column; align-items: center; gap: 4px; font-size: 13px; font-weight: 600; color: #525A75}
.bb-rail-device .bb-ico {width: 22px; height: 22px}
.bb.bb-top {display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; min-height: 60px; padding-right: 560px}
.bb .bb-wordmark {font-size: 30px; font-weight: 700; letter-spacing: -0.02em; line-height: 1.1; color: #121633}
.bb .bb-sub {font-size: 15px; font-weight: 500; color: #525A75; margin-top: 2px}
.bb-status-pill {display: none; white-space: nowrap; align-items: center; gap: 6px; padding: 6px 12px; border-radius: 999px; background: #fff; font-size: 13px; font-weight: 600}
.bb-dot {width: 8px; height: 8px; border-radius: 50%; background: #15924B}
.bb-dot.is-warn {background: #F2A516}

/* main navigation: Gradio's tabs as the dark pill bar, beside the wordmark; use | learn groups */
.bb-tabs {border: none !important; background: none !important; padding: 0 !important}
.bb-tabs > .tab-wrapper {justify-content: flex-end; margin-top: -84px; margin-bottom: 24px; border: none !important; height: auto !important;
  padding: 0 !important; pointer-events: none}
.bb-tabs > .tab-wrapper > .tab-container {pointer-events: auto; gap: 2px; padding: 6px; border-radius: 999px; background: #0E1232;
  border: none !important; height: auto !important; flex: 0 0 auto !important; width: auto !important}
.bb-tabs > .tab-wrapper button[role=tab] {border-radius: 999px; padding: 11px 20px !important; height: auto !important; color: #CDD2EC !important;
  font-weight: 600; font-size: 15px; border: none !important; background: transparent; transition: background-color .15s ease-out, color .15s ease-out}
.bb-tabs > .tab-wrapper button[role=tab]:hover {color: #fff !important; background: #1F2766}
.bb-tabs > .tab-wrapper button[role=tab].selected {background: #C6F16D; color: #0E1232 !important}
.bb-tabs > .tab-wrapper button[role=tab].selected::after {display: none}
.bb-tabs > .tab-wrapper button[role=tab]:nth-child(4) {position: relative; margin-left: 21px}
.bb-tabs > .tab-wrapper button[role=tab]:nth-child(4)::before {content: ""; position: absolute; left: -11px; top: 25%; height: 50%;
  border-left: 1px solid #2B3A8C}
.bb-tabs > .tabitem {padding: 0 !important; border: none !important; background: none !important}

/* the navy answer tile with its notch; the drop shadow follows the notch because it sits on the unmasked wrapper */
.bb.bb-hero {position: relative; height: 100%; min-height: 540px}
.bb-hero-body {height: 100%; min-height: inherit; filter: drop-shadow(0 14px 15px rgba(14,18,50,.14))}
.bb-hero.is-idle .bb-hero-surface {grid-template-areas: "head head" "foot side"; grid-template-rows: auto 1fr}
.bb-hero.no-side .bb-hero-surface {grid-template-columns: minmax(0, 1fr); grid-template-areas: "head" "main" "foot"; grid-template-rows: auto 1fr auto}
.bb-hero.is-idle .bb-hero-side {width: auto; align-self: end}
.bb-hero.is-idle .bb-hero-foot {align-self: end}
.bb-hero.is-idle .bb .bb-verdict.is-sentence, .bb-hero.is-idle .bb-verdict.is-sentence {max-width: 16ch}
.bb-hero-surface {height: 100%; min-height: inherit; box-sizing: border-box; display: grid; grid-template-columns: minmax(0, 1fr) auto;
  grid-template-areas: "head head" "main side" "foot side"; grid-template-rows: auto 1fr auto; container-type: inline-size; gap: 16px 24px; padding: 36px; border-radius: 28px;
  color: #fff; background: linear-gradient(180deg, #151A45 30%, #2E40A1 130%);
  -webkit-mask: radial-gradient(circle 40px at calc(100% - 34px) 34px, #0000 39.5px, #000 40.5px);
  mask: radial-gradient(circle 40px at calc(100% - 34px) 34px, #0000 39.5px, #000 40.5px)}
.bb-notch {position: absolute; top: 8px; right: 8px; z-index: 1; display: grid; place-items: center; width: 52px; height: 52px; border-radius: 50%;
  background: #0E1232; color: #fff}
.bb-hero-head {grid-area: head; display: flex; flex-direction: column; gap: 10px; min-width: 0; padding-right: 48px}
.bb-hero-main {grid-area: main; display: flex; flex-direction: column; justify-content: flex-end; gap: 12px; min-width: 0}
.bb-hero-side {grid-area: side; align-self: center; width: 320px}
.bb-hero-foot {grid-area: foot; display: flex; flex-wrap: wrap; align-items: center; gap: 12px}
.bb .bb-verdict {font-size: clamp(32px, 10.5cqi, 76px); font-weight: 700; letter-spacing: -0.035em; line-height: 0.92; color: #fff; overflow-wrap: anywhere}
.bb .bb-verdict.is-sentence {font-size: clamp(32px, 8.5cqi, 60px); line-height: 1; max-width: 13ch; text-wrap: balance}
.bb .bb-lead {font-size: 18px; font-weight: 500; line-height: 1.45; color: #CDD2EC; max-width: 46ch}
.bb .bb-lead.is-unsure {color: #FFC552}
.bb-say {display: flex; align-items: flex-start; gap: 16px}
.bb-say .bb-buddy {flex: none}
.bb .bb-bubble {background: #1F2766; color: #fff; font-size: 15px; font-weight: 500; line-height: 1.5; padding: 14px 18px;
  border-radius: 20px 20px 20px 6px; max-width: 32ch}
.bb .bb-bubble.is-solo {margin-top: 12px; border-radius: 20px}
.bb .bb-note {font-size: 13px; font-weight: 600; color: #CDD2EC}
.bb .bb-decided {font-size: 14px; font-weight: 600; color: #CDD2EC}
.bb-donut {width: 100%; height: auto; display: block}
.bb-donut text {font-family: inherit}
.bb-donut .bb-donut-val {font-size: 56px; font-weight: 700; letter-spacing: -0.03em}
.bb-donut .bb-donut-cap {font-size: 14px; font-weight: 600; fill: #CDD2EC}
.bb-donut .bb-chip-txt {font-size: 11px; font-weight: 700; fill: #fff}
.bb-cast {display: flex; gap: 14px; align-items: flex-end}
.bb-cast.is-dim {opacity: .5; margin-top: auto}
.bb-cast-item {display: flex; flex-direction: column; align-items: center; gap: 6px; text-align: center; min-width: 104px; white-space: nowrap}
.bb-cast-item figcaption {display: flex; flex-direction: column; gap: 2px}
.bb .bb-cast-item b {font-size: 14px; font-weight: 600; color: #fff}
.bb .bb-cast-item span {font-size: 13px; font-weight: 500; color: #CDD2EC}
.bb-cast.is-dim .bb-cast-item span {display: none}
.bb-state {display: inline-flex; align-items: center; gap: 8px; padding: 8px 14px 8px 12px; border-radius: 999px; font-size: 14px; font-weight: 700}
.bb .bb-state.is-agree {background: #C6F16D; color: #0E1232}
.bb .bb-state.is-unsure {background: #FFC552; color: #6E4300}
.bb .bb-state.is-disagree {color: #FFC552; box-shadow: inset 0 0 0 1.5px #FFC552}

/* light tiles */
.bb-tile {display: flex; flex-direction: column; gap: 14px; padding: 22px 24px; border-radius: 28px; background: #fff; color: #121633;
  box-shadow: 0 1px 2px rgba(18,22,51,.04), 0 8px 24px -8px rgba(14,18,50,.08)}
.bb-tile > header {display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 6px 12px}
.bb .bb-tile h3 {font-size: 20px; font-weight: 700; letter-spacing: -0.01em; line-height: 1.25}
.bb .bb-tile > header p, .bb .bb-hint {font-size: 13px; font-weight: 600; color: #525A75}
.bb .bb-tile > p {font-size: 15px; font-weight: 500; line-height: 1.5; color: #525A75}
.bb-letter {display: grid; place-items: center; flex: none; width: 40px; height: 40px; border-radius: 50%; background: #EEF2F6; font-weight: 700; font-size: 14px}
.bb .bb-letter.is-dark {background: #0E1232; color: #C6F16D}
.bb-methods {display: grid; grid-template-columns: minmax(0, 520fr) minmax(0, 736fr); gap: 24px; align-items: stretch; margin-top: 24px}
.bb-decider > header {flex-wrap: nowrap; align-items: center; justify-content: flex-start}
.bb-decider > header > div {flex: 1; min-width: 0}
.bb .bb-decider header p, .bb .bb-row-name span {font-size: 13px; font-weight: 600; color: #525A75}
.bb .bb-decides {display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; border-radius: 999px; background: #C6F16D; color: #0E1232;
  font-size: 13px; font-weight: 700}
.bb-call {display: flex; align-items: center; gap: 12px; font-size: 20px}
.bb-call .bb-buddy {width: 44px; height: auto}
.bb-meters {display: flex; flex-direction: column; gap: 10px}
.bb-meter {display: grid; grid-template-columns: 26px 104px 1fr 48px; align-items: center; gap: 10px; font-size: 14px; font-weight: 600}
.bb-meter > b {text-align: right}
.bb-chip {display: grid; place-items: center; width: 26px; height: 26px; border-radius: 50%}
.bb-chip .bb-ico {width: 15px; height: 15px}
.bb-bar {height: 10px; border-radius: 5px; background: #EEF2F6; overflow: hidden}
.bb-bar i {display: block; height: 100%; border-radius: 5px}
.bb-row {display: grid; grid-template-columns: 36px minmax(0, 1fr) auto 120px; align-items: center; gap: 14px; padding: 12px 16px;
  border-radius: 18px; background: #EEF2F6}
.bb-row .bb-letter {width: 36px; height: 36px; background: #fff}
.bb-row-name {display: flex; flex-direction: column; gap: 2px; min-width: 0}
.bb .bb-row-name b, .bb .bb-row-verdict {font-size: 14px; font-weight: 700}
.bb-mini {height: 10px; border-radius: 5px}

/* video: frame by frame */
.bb-frames-tile {margin-top: 24px}
.bb-frames-row {display: grid; grid-template-columns: 150px 1fr; align-items: center; gap: 16px; font-size: 14px}
.bb-frames {display: flex; gap: 3px; height: 30px}
.bb-frames i {flex: 1; border-radius: 3px}

/* compare */
.bb-bento {display: grid; gap: 24px; margin-top: 24px}
.bb-bento:first-child {margin-top: 0}
.bb-bento.is-hero {grid-template-columns: minmax(0, 760fr) minmax(0, 496fr)}
.bb-bento.is-split {grid-template-columns: minmax(0, 520fr) minmax(0, 736fr)}
.bb-bento.is-three {grid-template-columns: repeat(3, minmax(0, 1fr))}
.bb-bento.is-wide {grid-template-columns: minmax(0, 836fr) minmax(0, 420fr)}
.bb-stack {display: flex; flex-direction: column; gap: 24px}
.bb-compare .bb.bb-hero, .bb-about .bb.bb-hero {min-height: 440px}
.bb-f1s {display: flex; flex-direction: column; gap: 18px; margin-top: 18px}
.bb-f1 {display: grid; grid-template-columns: 1fr auto; gap: 8px; font-size: 14px; font-weight: 600; color: #CDD2EC}
.bb .bb-f1 b {font-size: 18px; font-weight: 700; color: #fff}
.bb .bb-f1.is-best span:first-child {color: #fff}
.bb .bb-f1.is-best b {color: #C6F16D}
.bb-f1-track {grid-column: 1 / -1; height: 14px; border-radius: 7px; background: #1F2766; overflow: hidden}
.bb-f1-track i {display: block; height: 100%; border-radius: 7px; background: rgba(205,210,236,.55)}
.bb-f1.is-best .bb-f1-track i {background: #C6F16D}
.bb-table {width: 100%; border-collapse: collapse; font-size: 14px}
.bb .bb-table, .bb .bb-table tr, .bb .bb-table th, .bb .bb-table td {border: 0; background: none}
.bb .bb-table th, .bb .bb-table td {padding: 10px 6px; border-top: 1px solid #D2D9E4; text-align: right; font-weight: 600; white-space: nowrap}
.bb .bb-table thead th {border-top: 0; font-size: 13px; color: #525A75}
.bb .bb-table th[scope=row], .bb .bb-table thead th:first-child {text-align: left}
.bb .bb-table td.is-best {font-weight: 800}
.bb .bb-table tr.is-quiet th, .bb .bb-table tr.is-quiet td {font-weight: 500; color: #525A75; font-size: 13px}
.bb-th {display: inline-flex; align-items: center; gap: 6px}
.bb-curves {display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px}
.bb-curve-box {display: flex; flex-direction: column; gap: 4px; font-size: 13px}
.bb .bb-curve-box span {color: #525A75; font-weight: 500}
.bb-curve {width: 100%; height: auto}
.bb-key {display: flex; gap: 14px}
.bb-key span {display: inline-flex; align-items: center; gap: 6px; font-weight: 700}
.bb-key i {width: 12px; height: 12px; border-radius: 3px}
.bb-groups {display: flex; justify-content: space-between; gap: 12px; height: 200px; align-items: flex-end}
.bb-group {display: flex; flex-direction: column; align-items: center; gap: 8px; height: 100%; justify-content: flex-end}
.bb-group-bars {display: flex; align-items: flex-end; gap: 4px; height: 170px}
.bb-group-bars span {position: relative; width: 22px; height: var(--h); background: var(--c); border-radius: 6px 6px 0 0}
.bb-group-bars b {position: absolute; bottom: 100%; left: 50%; transform: translateX(-50%); font-size: 11px; padding-bottom: 2px}
.bb .bb-group-label {font-size: 13px; font-weight: 600; color: #525A75}
.bb-cms {display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 32px}
.bb-cm > p {font-size: 13px; color: #525A75; margin-bottom: 10px}
.bb .bb-cm > p b {font-size: 14px; color: #121633; margin-right: 6px}
.bb-cm-grid {display: grid; grid-template-columns: 112px repeat(3, minmax(0, 1fr)); gap: 4px; align-items: center}
.bb-cm-head {display: grid; place-items: center; color: #525A75; padding-bottom: 4px}
.bb-cm-row-label {display: flex; align-items: center; gap: 6px; font-size: 13px; font-weight: 600}
.bb .bb-cm-row-label span {color: #121633}
.bb-cm-cell {display: flex; flex-direction: column; align-items: center; justify-content: center; height: 58px; border-radius: 10px; color: #121633}
.bb .bb-cm-cell.is-dark {color: #fff}
.bb-cm-cell b {font-size: 15px}
.bb-cm-cell small {font-size: 13px; opacity: .85}

/* about */
.bb-rules {display: flex; flex-direction: column; gap: 18px}
.bb-rules li, .bb-steps li {display: flex; gap: 14px; align-items: flex-start}
.bb-rule-ico {display: grid; place-items: center; flex: none; width: 44px; height: 44px; border-radius: 50%; background: #EEF2F6}
.bb .bb-rule-ico.is-warn {background: #FFF0D2; color: #6E4300}
.bb .bb-rules p, .bb .bb-steps p {font-size: 15px; font-weight: 500; line-height: 1.5; color: #525A75}
.bb-steps {display: flex; flex-direction: column; gap: 14px}
.bb-bin > header {flex-wrap: nowrap; align-items: center; justify-content: flex-start; gap: 16px}
.bb .bb-bin h3 {font-size: 20px}
.bb .bb-bin header p {font-size: 14px; font-weight: 600; color: inherit}
.bb-examples {display: flex; flex-wrap: wrap; gap: 8px}
.bb-examples li {padding: 7px 12px; border-radius: 999px; background: rgba(255,255,255,.75); font-size: 14px; font-weight: 600}

/* Gradio widgets in the same language */
.bb-switch {background: none !important; box-shadow: none !important; padding: 0 !important}
.bb-switch .wrap {display: inline-flex !important; gap: 2px !important; padding: 4px; border-radius: 999px; background: #0E1232; width: auto !important}
.bb-switch label {border: none !important; border-radius: 999px !important; background: transparent !important; color: #CDD2EC !important;
  padding: 7px 14px !important; font-size: 13px !important; font-weight: 700 !important; box-shadow: none !important; cursor: pointer}
.bb-switch label.selected {background: #C6F16D !important; color: #0E1232 !important}
.bb-switch input[type=radio] {position: absolute; opacity: 0; width: 1px; height: 1px}
.bb-input, .bb-output {border-radius: 28px !important; overflow: hidden}
.bb-caption {font-size: 13px; font-weight: 600; color: #525A75; padding: 0 4px}
.bb-details {border-radius: 28px !important}
.bb-details-title {font-size: 20px; font-weight: 700; margin: 8px 4px 0}

@media (prefers-reduced-motion: reduce) {* {transition: none !important; animation: none !important}}
@media (max-width: 1100px) {
  .bb.bb-top {padding-right: 0}
  .bb-tabs > .tab-wrapper {margin-top: 0; justify-content: flex-start; overflow-x: auto; scrollbar-width: none}
  .bb-tabs > .tab-wrapper::-webkit-scrollbar {display: none}
  .bb-bento.is-hero, .bb-bento.is-split, .bb-bento.is-wide, .bb-methods {grid-template-columns: minmax(0, 1fr)}
}
@media (max-width: 760px) {
  .bb-rail-col {display: none !important}
  .bb-status-pill {display: inline-flex}
  .bb .bb-wordmark {font-size: 24px}
  .bb .bb-sub {display: none}
  .bb-tabs > .tab-wrapper button[role=tab] {padding: 10px 14px !important; font-size: 14px}
  .bb.bb-hero {min-height: 0}
  .bb-hero-surface, .bb-hero.is-idle .bb-hero-surface, .bb-hero.no-side .bb-hero-surface {
    grid-template-columns: minmax(0, 1fr); grid-template-areas: "head" "main" "side" "foot"; grid-template-rows: auto; padding: 24px 20px}
  .bb-hero-side {width: 100%; max-width: 300px; justify-self: center}
  .bb-cast {gap: 8px; justify-content: space-between}
  .bb-cast-item {flex: 1; min-width: 0; white-space: normal}
  .bb-cast .bb-buddy {width: 64px; height: auto}
  .bb-bento.is-three, .bb-cms {grid-template-columns: minmax(0, 1fr)}
  .bb-groups {gap: 4px}
  .bb-group {min-width: 0}
  .bb-group-bars {gap: 2px}
  .bb-group-bars span {width: 12px}
  .bb-group-bars b {font-size: 10px}
  .bb .bb-group-label {font-size: 11px; text-align: center}
  .bb-row {grid-template-columns: 36px minmax(0, 1fr) auto}
  .bb-mini {display: none}
  .bb-meter {grid-template-columns: 26px 96px 1fr 44px}
  .bb-frames-row {grid-template-columns: 1fr}
}
"""


# ---- UI ------------------------------------------------------------------------------------------
# Light-only design: drop dark mode. Then the buddies' pupils follow the pointer. Buddies are looked up on every frame
# because Gradio replaces the HTML with each answer; pupils travel at most 3.5 of the eye's 9 units, so they stay inside.
LOAD_JS = """() => {
  document.body.classList.remove('dark');
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  let x = null, y = null, queued = false;
  const look = () => {
    queued = false;
    for (const svg of document.querySelectorAll('.bb-buddy')) {
      const m = svg.getScreenCTM();
      if (!m || !svg.getClientRects().length) continue;
      const toSvg = x === null ? null : new DOMPoint(x, y).matrixTransform(m.inverse());
      for (const eye of svg.querySelectorAll('.bb-eye')) {
        let dx = 1.5, dy = 1.5;
        if (toSvg) {
          const c = eye.firstElementChild, vx = toSvg.x - c.cx.baseVal.value, vy = toSvg.y - c.cy.baseVal.value;
          const d = Math.hypot(vx, vy) || 1, r = Math.min(3.5, d / 12);
          dx = vx / d * r; dy = vy / d * r;
        }
        eye.setAttribute('transform', `translate(${dx.toFixed(2)} ${dy.toFixed(2)})`);
      }
    }
  };
  const queue = () => { if (!queued) { queued = true; requestAnimationFrame(look); } };
  addEventListener('pointermove', e => { x = e.clientX; y = e.clientY; queue(); }, {passive: true});
  document.documentElement.addEventListener('pointerleave', () => { x = null; queue(); });
}"""


def build_ui() -> gr.Blocks:
    # one GoogleFont only (Gradio 5.50 crashes comparing a second one with its built-in themes); offline it falls back to
    # the system sans-serif. The design is light-only, so dark mode is switched off on load.
    theme = gr.themes.Base(primary_hue="lime", neutral_hue="slate",
                           font=[gr.themes.GoogleFont("Urbanist", weights=(500, 600, 700, 800)), "ui-sans-serif", "system-ui",
                                 "sans-serif"]).set(
        body_background_fill="#E3E8EF", body_text_color="#121633", body_text_color_subdued="#525A75",
        background_fill_primary="#FFFFFF", background_fill_secondary="#EEF2F6", border_color_primary="#D2D9E4",
        block_background_fill="#FFFFFF", block_border_width="0px", block_radius="28px", block_padding="16px",
        block_shadow="0 1px 2px rgba(18,22,51,.04), 0 8px 24px -8px rgba(14,18,50,.08)",
        block_label_background_fill="#EEF2F6", block_label_text_color="#121633", block_label_radius="999px",
        block_title_text_weight="700", layout_gap="24px", color_accent="#2F6BFF", color_accent_soft="#E1EAFF",
        button_large_radius="999px", button_large_padding="15px 24px", button_large_text_size="16px", button_large_text_weight="700",
        button_primary_background_fill="#C6F16D", button_primary_background_fill_hover="#B4E553", button_primary_text_color="#0E1232",
        button_primary_border_color="#C6F16D", button_secondary_background_fill="#0E1232", button_secondary_text_color="#FFFFFF",
    )
    with gr.Blocks(title="BasuraBin - which bin does it go in?", theme=theme, css=CSS,
                   js=LOAD_JS) as demo:
        with gr.Row(elem_classes="bb-shell", equal_height=False):
            with gr.Column(scale=0, min_width=88, elem_classes="bb-rail-col"):
                gr.HTML(rail_html(), padding=False)
            with gr.Column(scale=1):
                gr.HTML(topbar_html(), padding=False)
                # every use tab has the same shape: the answer on the left (navy), the input/evidence on the right, then
                # how each method saw it
                with gr.Tabs(elem_classes="bb-tabs"):
                    with gr.Tab("Photo"):
                        verdict = gr.HTML(idle_hero(*PHOTO_IDLE), padding=False, render=False)
                        main_img = gr.Image(label="Drawn on your photo", interactive=False, height=430, visible=False, render=False,
                                            show_label=False, elem_classes="bb-output")
                        methods = gr.HTML(methods_html(idle_rows(), IDLE_HINT), padding=False, render=False)
                        inp = gr.Image(type="numpy", sources=["upload", "webcam", "clipboard"], label="Photo of one waste item",
                                       show_label=False, height=430, render=False, elem_classes="bb-input")
                        switch = gr.Radio(["Original", "Drawn"], value="Original", show_label=False, visible=False, render=False,
                                          container=False, elem_classes="bb-switch", label="Show")
                        outs = [main_img, verdict, methods, inp, switch]
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=7):
                                verdict.render()
                            with gr.Column(scale=5):
                                switch.render()
                                inp.render()
                                main_img.render()
                                gr.HTML('<p class="bb bb-caption">One item per photo · plain background · good light</p>', padding=False)
                                btn = gr.Button("Which bin?", variant="primary", size="lg")
                        methods.render()
                        ex = example_images()
                        if ex:  # clicking an example classifies it straight away
                            gr.Examples(ex, inputs=inp, outputs=outs, fn=on_image, run_on_click=True, cache_examples=False,
                                        examples_per_page=12, label="No photo? Try a test-set example (never seen in training)")
                        btn.click(on_image, inp, outs)
                        inp.upload(on_image, inp, outs)
                        inp.clear(on_image, inp, outs)  # back to the idle state, so no stale answer sits next to an empty input
                        switch.input(on_switch, switch, [inp, main_img])

                    with gr.Tab("Live camera"):
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=7):
                                live_verdict = gr.HTML(idle_hero(*LIVE_IDLE), padding=False)
                            with gr.Column(scale=5):
                                cam = gr.Image(sources=["webcam"], type="numpy", streaming=True, label="Camera", show_label=False,
                                               elem_classes="bb-input")
                                live = gr.Image(label="Drawn on the camera feed", interactive=False, show_label=False, height=260,
                                                elem_classes="bb-output")
                        live_methods = gr.HTML(methods_html(idle_rows(), IDLE_HINT), padding=False)
                        smooth_state = gr.State(None)  # one Smoother per browser session
                        cam.stream(on_webcam, [cam, smooth_state], [live, live_verdict, live_methods, smooth_state], stream_every=0.15,
                                   time_limit=900, concurrency_limit=1)

                    with gr.Tab("Video"):
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=7):
                                vverdict = gr.HTML(idle_hero(*VIDEO_IDLE), padding=False)
                            with gr.Column(scale=5):
                                vswitch = gr.Radio(["Original", "Drawn"], value="Original", show_label=False, visible=False,
                                                   container=False, elem_classes="bb-switch", label="Show")
                                vin = gr.Video(label="Input video", show_label=False, sources=["upload"], height=430, elem_classes="bb-input")
                                vout = gr.Video(label="Drawn on your video", show_label=False, visible=False, height=430,
                                                elem_classes="bb-output")
                                vbtn = gr.Button("Analyse video", variant="primary", size="lg")
                        vdetail = gr.HTML(padding=False)
                        vbtn.click(on_video, vin, [vout, vin, vswitch, vverdict, vdetail])
                        vswitch.input(on_video_switch, vswitch, [vin, vout])

                    # conclusion first, then the evidence, then the raw details
                    with gr.Tab("Compare methods"):
                        gr.HTML(compare_html(), padding=False)
                        gr.HTML('<h2 class="bb bb-details-title">Details</h2>', padding=False)
                        with gr.Accordion("Why this method", open=False, elem_classes="bb-details"):
                            gr.Markdown(S.explanation)
                        mistakes = RESULTS_DIR / f"misclassified_{S.best}.png"
                        if mistakes.exists():
                            with gr.Accordion(f"Test photos {METHOD_INFO[S.best]['short']} got wrong", open=False, elem_classes="bb-details"):
                                gr.Image(str(mistakes), show_label=False, interactive=False)
                        with gr.Accordion("Full results table", open=False, elem_classes="bb-details"):
                            md = RESULTS_DIR / "results_table.md"
                            gr.Markdown(md.read_text() if md.exists() else "No results yet.")
                        with gr.Accordion("Experiments", open=False, elem_classes="bb-details"):
                            gr.Markdown(experiments_md())

                    with gr.Tab("About"):
                        gr.HTML(about_html(), padding=False)
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
