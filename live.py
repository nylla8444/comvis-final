"""Real-time webcam demo in a plain OpenCV window (most reliable option for the class demo).

    python live.py               # default camera
    python live.py --camera 1    # another camera (e.g. USB webcam)
    python live.py --cpu         # run the networks on the CPU

Keys:  q / Esc = quit,  s = save snapshot to results/snapshots/,  m = mirror on/off
"""
from __future__ import annotations

import argparse
import os
import time

import cv2

from annotate import draw_all, run_methods
from common import RESULTS_DIR, load_json
from methods import load_predictors


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args(argv)

    preds, missing = load_predictors("cpu" if a.cpu else None)
    if not preds:
        raise SystemExit("No trained models in models/. Train the methods first.")
    res = load_json(RESULTS_DIR / "results.json")
    best = res["recommendation"]["best"] if res and res["recommendation"]["best"] in preds else \
        ("pretrained" if "pretrained" in preds else next(iter(preds)))

    cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {a.camera}")
    mirror, fps, last = True, 0.0, time.perf_counter()
    snap_dir = RESULTS_DIR / "snapshots"
    print("press q to quit, s to save a snapshot, m to toggle mirror")
    while True:
        ok, bgr = cap.read()
        if not ok:
            print("camera frame not received")
            break
        if mirror:
            bgr = cv2.flip(bgr, 1)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = run_methods(preds, rgb)
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6)) if fps else 1.0 / max(now - last, 1e-6)
        last = now
        out = draw_all(rgb, results, best, max_side=1100, fps=fps)
        cv2.imshow("BasuraBin live (q = quit)", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("m"):
            mirror = not mirror
        if key == ord("s"):
            snap_dir.mkdir(parents=True, exist_ok=True)
            path = snap_dir / f"snap_{int(time.time())}.jpg"
            cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            print("saved", path)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
