"""Real-time webcam demo in a plain OpenCV window (most reliable option for the class demo).

    python live.py                   # default camera
    python live.py --camera 1        # another camera (e.g. USB webcam)
    python live.py --cpu             # run the networks on the CPU
    python live.py --save-to train   # keys 1/2/3 save photos for TRAINING instead of the local test set

Keys:  q / Esc = quit,  s = save annotated snapshot to results/snapshots/,  m = mirror on/off
       1 / 2 / 3 = save the raw camera frame as a biodegradable / recyclable / residual photo
                   (data/raw/local/<bin>/ by default, data/raw/local_train/<bin>/ with --save-to train;
                    then run prepare_data.py again so the photos are used)
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2

from annotate import Smoother, draw_all, run_methods
from common import RAW_DIR, RESULTS_DIR, load_json
from methods import load_predictors

SAVE_KEYS = {ord("1"): "biodegradable", ord("2"): "recyclable", ord("3"): "residual"}


def save_photo(frame_bgr, bin_: str, root: Path) -> tuple[Path, int]:
    """Save one raw camera frame (no drawings) into root/<bin>/; returns the path and the number of photos there."""
    folder = root / bin_
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"cam_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.jpg"
    cv2.imwrite(str(path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path, len(list(folder.glob("*.jpg")))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--save-to", choices=["test", "train"], default="test",
                    help="where keys 1/2/3 save photos: data/raw/local (extra test set) or data/raw/local_train")
    ap.add_argument("--no-smooth", action="store_true", help="show the raw per-frame prediction (no averaging)")
    a = ap.parse_args(argv)

    preds, missing = load_predictors("cpu" if a.cpu else None)
    if not preds:
        raise SystemExit("No trained models in models/. Train the methods first.")
    res = load_json(RESULTS_DIR / "results.json")
    best = res["recommendation"]["best"] if res and res["recommendation"]["best"] in preds else \
        ("mobilenet_v3_large" if "mobilenet_v3_large" in preds else next(iter(preds)))

    cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {a.camera}")
    smoother = None if a.no_smooth else Smoother()
    mirror, fps, last = True, 0.0, time.perf_counter()
    snap_dir = RESULTS_DIR / "snapshots"
    photo_root = RAW_DIR / ("local" if a.save_to == "test" else "local_train")
    notice, notice_until = "", 0.0
    print("q = quit, s = snapshot, m = mirror, 1/2/3 = save photo as biodegradable/recyclable/residual"
          f" (into {photo_root})")
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            print("camera frame not received")
            break
        bgr = cv2.flip(frame_bgr, 1) if mirror else frame_bgr
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = run_methods(preds, rgb)
        if smoother:
            results = smoother.update(results)
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6)) if fps else 1.0 / max(now - last, 1e-6)
        last = now
        out = cv2.cvtColor(draw_all(rgb, results, best, max_side=1100, fps=fps), cv2.COLOR_RGB2BGR)
        if now < notice_until:  # short confirmation after saving a photo
            for color, thick in [((0, 0, 0), 5), ((255, 255, 255), 2)]:
                cv2.putText(out, notice, (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thick, cv2.LINE_AA)
        cv2.imshow("BasuraBin live (q = quit)", out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("m"):
            mirror = not mirror
        if key == ord("s"):
            snap_dir.mkdir(parents=True, exist_ok=True)
            path = snap_dir / f"snap_{int(time.time())}.jpg"
            cv2.imwrite(str(path), out)
            print("saved", path)
        if key in SAVE_KEYS:
            path, n = save_photo(frame_bgr, SAVE_KEYS[key], photo_root)
            notice, notice_until = f"saved: {SAVE_KEYS[key]} ({n} photos in {photo_root.name})", now + 1.5
            print("saved", path)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
