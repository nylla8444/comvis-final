"""Draw predictions on images (shared by the Gradio app and the OpenCV live demo)."""
from __future__ import annotations

import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import CACHE_SIZE, CLASSES, COLORS, DISPLAY, METHOD_INFO, center_square, preprocess_array

UNSURE_THRESHOLD = 0.55  # below this top-class confidence the app says "not sure"


def _font(size: int):
    for name in ["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def run_methods(predictors: dict, frame: np.ndarray) -> dict:
    """frame: HxWx3 uint8 RGB (any size). Returns {key: {"probs", "label", "conf", "ms"}}."""
    x = preprocess_array(frame, CACHE_SIZE)  # centre square -> 256x256, exactly like the training images
    out = {}
    for key, p in predictors.items():
        t0 = time.perf_counter()
        probs = p.predict_proba([x])[0]
        if hasattr(p, "sync"):
            p.sync()
        ms = (time.perf_counter() - t0) * 1000
        i = int(probs.argmax())
        out[key] = {"probs": probs, "label": CLASSES[i], "conf": float(probs[i]), "ms": ms}
    return out


def guide_box(w: int, h: int) -> tuple[int, int, int, int]:
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return left, top, left + s, top + s


def draw_single(frame: np.ndarray, key: str, r: dict, max_side: int = 640) -> np.ndarray:
    """One method's result drawn on the input: coloured guide box + label banner."""
    img = Image.fromarray(frame).convert("RGB")
    scale = max_side / max(img.size)
    if scale < 1:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.BILINEAR)
    d = ImageDraw.Draw(img)
    unsure = r["conf"] < UNSURE_THRESHOLD
    color = (230, 160, 0) if unsure else COLORS[r["label"]]
    box = guide_box(*img.size)
    bw = max(3, img.width // 120)
    d.rectangle(box, outline=color, width=bw)
    d.rectangle((box[0] + bw, box[1] + bw, box[2] - bw, box[3] - bw), outline="white", width=1)
    text = f"{'Not sure - ' if unsure else ''}{DISPLAY[r['label']]}  {r['conf'] * 100:.0f}%"
    sub = f"{METHOD_INFO[key]['short']}  |  {r['ms']:.1f} ms"
    f1, f2 = _font(max(14, img.width // 28)), _font(max(11, img.width // 42))
    pad = 6
    th = f1.getbbox(text)[3] + f2.getbbox(sub)[3] + 3 * pad
    d.rectangle((0, 0, img.width, th), fill=color)
    d.text((pad, pad), text, fill="white", font=f1)
    d.text((pad, pad * 2 + f1.getbbox(text)[3]), sub, fill="white", font=f2)
    return np.asarray(img)


def draw_all(frame: np.ndarray, results: dict, best: str | None, max_side: int = 720, fps: float | None = None) -> np.ndarray:
    """All methods on one frame (live webcam / video): guide box coloured by the recommended method,
    and one line per method at the bottom (recommended one marked with a star)."""
    img = Image.fromarray(frame).convert("RGB")
    scale = max_side / max(img.size)
    if scale < 1:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.BILINEAR)
    d = ImageDraw.Draw(img)
    lead = best if best in results else next(iter(results), None)
    if lead:
        r = results[lead]
        color = (230, 160, 0) if r["conf"] < UNSURE_THRESHOLD else COLORS[r["label"]]
        box, bw = guide_box(*img.size), max(3, img.width // 110)
        d.rectangle(box, outline=color, width=bw)
        d.rectangle((box[0] + bw, box[1] + bw, box[2] - bw, box[3] - bw), outline="white", width=1)
    f = _font(max(13, img.width // 40))
    line_h = f.getbbox("Ag")[3] + 8
    n = len(results) + (1 if fps else 0)
    y0 = img.height - n * line_h - 6
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle((0, y0 - 4, img.width, img.height), fill=(0, 0, 0, 150))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    d = ImageDraw.Draw(img)
    y = y0
    for key, r in results.items():
        unsure = r["conf"] < UNSURE_THRESHOLD
        star = "* " if key == best else "  "
        label = "not sure" if unsure else DISPLAY[r["label"]]
        d.rectangle((6, y + 3, 6 + line_h - 10, y + line_h - 7), fill=(230, 160, 0) if unsure else COLORS[r["label"]])
        d.text((line_h, y), f"{star}{METHOD_INFO[key]['short']}: {label} {r['conf'] * 100:.0f}%  ({r['ms']:.0f} ms)",
               fill="white", font=f)
        y += line_h
    if fps:
        d.text((line_h, y), f"{fps:.1f} FPS   (* = recommended method)", fill=(200, 200, 200), font=f)
    return np.asarray(img)


__all__ = ["run_methods", "draw_single", "draw_all", "UNSURE_THRESHOLD", "center_square"]
