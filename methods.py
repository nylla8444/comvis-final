"""Load whichever of the three fine-tuned models exist, behind one common interface:

    predictor.predict_proba(list_of_256x256_uint8_rgb_arrays) -> (N, 3) array of class probabilities
"""
from __future__ import annotations

from importlib import import_module

from common import METHOD_INFO, METHOD_KEYS, MODELS_DIR

MODEL_FILES = {k: MODELS_DIR / f"{k}.pt" for k in METHOD_KEYS}


def load_predictors(device=None, keys=METHOD_KEYS, verbose=True):
    """Returns ({key: predictor}, [missing keys])."""
    preds, missing = {}, []
    for k in keys:
        if not MODEL_FILES[k].exists():
            missing.append(k)
            continue
        try:
            preds[k] = import_module(METHOD_INFO[k]["module"]).load_predictor(device)
        except Exception as e:  # e.g. a model file saved by an older version of the code: retrain that model
            print(f"could not load {MODEL_FILES[k].name} ({type(e).__name__}: {e}) - retrain this model")
            missing.append(k)
    if verbose and missing:
        print(f"not trained yet or not loadable (skipped): {missing}")
    return preds, missing
