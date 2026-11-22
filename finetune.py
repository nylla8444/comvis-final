"""Shared fine-tuning recipe for the three pretrained models, so the comparison is fair.

Every model (mobilenet_v3.py, efficientnet_b0.py, resnet18.py) gets exactly the same data, splits, input size,
preprocessing, augmentation, loss, optimiser, schedule and early-stopping rule. A model module only says how to load
its ImageNet weights and which layer is its classification head.

Preprocessing: centre square crop -> 224x224 -> ImageNet mean/std normalisation (what the ImageNet weights expect).
Fine-tuning in two stages:
  1) freeze the backbone and train only the new 3-class head (fast; keeps the ImageNet features intact)
  2) unfreeze everything and train with a 10x smaller learning rate on the backbone than on the head
Loss: class-weighted cross-entropy with label smoothing 0.1; AdamW + cosine schedule; early stopping on validation
macro-F1 (the best weights are kept).

    python finetune.py          # fine-tune all three models one after another
"""
from __future__ import annotations

import argparse
import os
from importlib import import_module

import torch
import torch.nn as nn

from common import METHOD_INFO, METHOD_KEYS, MODELS_DIR, SEED, save_json, set_seed
from train_utils import TorchPredictor, count_macs, count_params, fit, get_device, make_loader, val_metrics_of

IMG_SIZE = 224


def imagenet(weights):
    """ImageNet weights to load. COMVIS_RANDOM_INIT=1 skips the download (only for quick pipeline tests)."""
    return None if os.environ.get("COMVIS_RANDOM_INIT") == "1" else weights


def model_path(key: str):
    return MODELS_DIR / f"{key}.pt"


def finetune(key: str, model: nn.Module, head: nn.Module, head_epochs=3, epochs=15, batch_size=64, lr=1e-3,
             weight_decay=1e-4, patience=5, num_workers=None, device=None) -> dict:
    """Two-stage fine-tuning; saves models/<key>.pt and models/<key>_meta.json."""
    set_seed(SEED)
    device = get_device(device)
    name = METHOD_INFO[key]["name"]
    print(f"fine-tuning {name} (ImageNet weights) on {device}")
    head_params = list(head.parameters())
    head_ids = {id(p) for p in head_params}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    tr = make_loader("train", IMG_SIZE, batch_size, train=True, num_workers=num_workers)
    va = make_loader("val", IMG_SIZE, batch_size, train=False, num_workers=num_workers)

    # stage 1: head only
    for p in backbone:
        p.requires_grad_(False)
    s1 = fit(model, tr, va, device, epochs=head_epochs, lr=lr, weight_decay=weight_decay, patience=head_epochs,
             param_groups=[{"params": head_params}], log_prefix=f"[{key}/head] ")
    # stage 2: whole network, smaller learning rate on the backbone
    for p in backbone:
        p.requires_grad_(True)
    s2 = fit(model, tr, va, device, epochs=epochs, lr=lr, weight_decay=weight_decay, patience=patience,
             param_groups=[{"params": backbone, "lr": lr / 10}, {"params": head_params, "lr": lr}],
             log_prefix=f"[{key}/full] ")

    path = model_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "key": key, "img_size": IMG_SIZE}, path)
    history = {k: s1["history"][k] + s2["history"][k] for k in s1["history"]}
    meta = {
        "method": key,
        "architecture": name,
        "display_name": name,
        "weights": "ImageNet-1k (torchvision)",
        "image_size": IMG_SIZE,
        "params_millions": count_params(model),
        "macs_millions": count_macs(model, IMG_SIZE),
        "hyperparameters": {"head_epochs": head_epochs, "epochs": epochs, "batch_size": batch_size, "lr_head": lr,
                            "lr_backbone": lr / 10, "weight_decay": weight_decay, "patience": patience,
                            "optimizer": "AdamW", "schedule": "cosine", "label_smoothing": 0.1},
        "stage1_epochs_run": len(s1["history"]["train_loss"]),
        "stage1_best_val_f1": s1["best_val_f1"],  # frozen ImageNet features + new head only
        "best_epoch": len(s1["history"]["train_loss"]) + s2["best_epoch"],
        "train_time_min": s1["train_time_min"] + s2["train_time_min"],
        "device": str(device) + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
        "val": val_metrics_of(model, IMG_SIZE, device),
        "history": history,
    }
    save_json(meta, MODELS_DIR / f"{key}_meta.json")
    print(f"saved {path} | params {meta['params_millions']:.2f} M | {meta['macs_millions']:.0f} M MACs/img"
          f" | val F1 head only {meta['stage1_best_val_f1']:.2f}% -> fine-tuned {meta['val']['macro_f1']:.2f}%"
          f" | {meta['train_time_min']:.1f} min")
    return meta


def load_predictor(key: str, model: nn.Module, device=None) -> TorchPredictor:
    """`model` is the untrained architecture (no ImageNet download needed); the fine-tuned weights are loaded into it."""
    ckpt = torch.load(model_path(key), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    return TorchPredictor(key, model, ckpt.get("img_size", IMG_SIZE), device)


def add_train_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--head-epochs", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")


def train_kwargs(a) -> dict:
    return dict(head_epochs=a.head_epochs, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, num_workers=a.workers,
                device=a.device)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_train_args(ap)
    a = ap.parse_args(argv)
    for key in METHOD_KEYS:
        import_module(METHOD_INFO[key]["module"]).train(**train_kwargs(a))


if __name__ == "__main__":
    main()
