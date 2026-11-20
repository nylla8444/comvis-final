"""Method C - Pretrained, modern: MobileNetV3-Large (Howard et al., 2019) with ImageNet weights from
torchvision, fine-tuned on our 3 bins with the same splits, augmentation and loss as the scratch CNN.

Why MobileNetV3: it was designed for phones (depthwise-separable convs, squeeze-excite, h-swish), so it
runs in real time on a laptop CPU for the live webcam demo. `--arch efficientnet_b0` or `resnet18`
are drop-in alternatives from the same method menu.

Fine-tuning in two stages:
  1) freeze the backbone, train only the new 3-class head (fast, protects the ImageNet features)
  2) unfreeze everything, train with a 10x smaller LR on the backbone than on the head

    python pretrained.py --epochs 15
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torchvision import models

from common import CLASSES, MODELS_DIR, SEED, save_json, set_seed
from train_utils import TorchPredictor, count_params, fit, get_device, make_loader, val_metrics_of

IMG_SIZE = 224
MODEL_PATH = MODELS_DIR / "pretrained.pt"
ARCHS = {
    "mobilenet_v3_large": "MobileNetV3-Large",
    "efficientnet_b0": "EfficientNet-B0",
    "resnet18": "ResNet-18",
}


def build_model(arch: str = "mobilenet_v3_large", pretrained: bool = True, n_classes: int = len(CLASSES)) -> nn.Module:
    """Load the published architecture (+ ImageNet weights) and swap the last layer for a 3-class one."""
    if arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    elif arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    else:
        raise ValueError(f"unknown arch {arch}")
    return m


def head_and_backbone(model: nn.Module, arch: str):
    head = model.fc if arch == "resnet18" else model.classifier
    head_ids = {id(p) for p in head.parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return list(head.parameters()), backbone


def train(arch="mobilenet_v3_large", head_epochs=3, epochs=15, batch_size=64, lr=1e-3, weight_decay=1e-4, patience=5,
          num_workers=None, device=None) -> dict:
    set_seed(SEED)
    device = get_device(device)
    print(f"fine-tuning {ARCHS[arch]} (ImageNet weights) on {device}")
    model = build_model(arch)
    head, backbone = head_and_backbone(model, arch)
    tr = make_loader("train", IMG_SIZE, batch_size, train=True, num_workers=num_workers)
    va = make_loader("val", IMG_SIZE, batch_size, train=False, num_workers=num_workers)

    # stage 1: head only
    for p in backbone:
        p.requires_grad_(False)
    s1 = fit(model, tr, va, device, epochs=head_epochs, lr=lr, weight_decay=weight_decay, patience=head_epochs,
             param_groups=[{"params": head}], log_prefix="[pretrained/head] ")
    # stage 2: whole network, smaller LR on the backbone
    for p in backbone:
        p.requires_grad_(True)
    s2 = fit(model, tr, va, device, epochs=epochs, lr=lr, weight_decay=weight_decay, patience=patience,
             param_groups=[{"params": backbone, "lr": lr / 10}, {"params": head, "lr": lr}],
             log_prefix="[pretrained/full] ")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "arch": arch, "img_size": IMG_SIZE}, MODEL_PATH)
    history = {k: s1["history"][k] + s2["history"][k] for k in s1["history"]}
    meta = {
        "method": "pretrained",
        "architecture": ARCHS[arch],
        "arch": arch,
        "weights": "ImageNet-1k (torchvision)",
        "image_size": IMG_SIZE,
        "params_millions": count_params(model),
        "hyperparameters": {"head_epochs": head_epochs, "epochs": epochs, "batch_size": batch_size, "lr_head": lr,
                            "lr_backbone": lr / 10, "weight_decay": weight_decay, "patience": patience},
        "stage1_epochs_run": len(s1["history"]["train_loss"]),
        "best_epoch": len(s1["history"]["train_loss"]) + s2["best_epoch"],
        "train_time_min": s1["train_time_min"] + s2["train_time_min"],
        "device": str(device) + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
        "val": val_metrics_of(model, IMG_SIZE, device),
        "history": history,
    }
    save_json(meta, MODELS_DIR / "pretrained_meta.json")
    print(f"saved {MODEL_PATH} | params {meta['params_millions']:.2f} M | val F1 {meta['val']['macro_f1']:.2f}%"
          f" | {meta['train_time_min']:.1f} min")
    return meta


def load_predictor(device=None, path=MODEL_PATH) -> TorchPredictor:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ckpt.get("arch", "mobilenet_v3_large"), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return TorchPredictor("pretrained", model, ckpt.get("img_size", IMG_SIZE), device)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="mobilenet_v3_large", choices=list(ARCHS))
    ap.add_argument("--head-epochs", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    train(a.arch, a.head_epochs, a.epochs, a.batch_size, a.lr, num_workers=a.workers, device=a.device)


if __name__ == "__main__":
    main()
