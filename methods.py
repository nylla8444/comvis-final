"""Load whichever of the three trained methods exist, behind one common interface:

    predictor.predict_proba(list_of_256x256_uint8_rgb_arrays) -> (N, 3) array of class probabilities
"""
from __future__ import annotations

from common import METHOD_KEYS, MODELS_DIR

MODEL_FILES = {
    "classical": MODELS_DIR / "classical.npz",
    "scratch": MODELS_DIR / "scratch_cnn.pt",
    "pretrained": MODELS_DIR / "pretrained.pt",
}


def load_predictors(device=None, keys=METHOD_KEYS, verbose=True):
    """Returns ({key: predictor}, [missing keys])."""
    preds, missing = {}, []
    for k in keys:
        if not MODEL_FILES[k].exists():
            missing.append(k)
            continue
        if k == "classical":
            from classical import ClassicalPredictor

            preds[k] = ClassicalPredictor(MODEL_FILES[k])
        elif k == "scratch":
            from scratch_cnn import load_predictor

            preds[k] = load_predictor(device)
        else:
            from pretrained import load_predictor

            preds[k] = load_predictor(device)
    if verbose and missing:
        print(f"not trained yet (skipped): {missing}")
    return preds, missing
