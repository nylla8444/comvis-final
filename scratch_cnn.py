"""Method B - Trained from scratch: WasteNet, our own 4-block CNN written in PyTorch, random init.

Architecture (input 3x128x128):
    block 1:  [conv3x3(32)  - BN - ReLU] x2 - maxpool2   -> 32 x 64 x 64
    block 2:  [conv3x3(64)  - BN - ReLU] x2 - maxpool2   -> 64 x 32 x 32
    block 3:  [conv3x3(128) - BN - ReLU] x2 - maxpool2   -> 128 x 16 x 16
    block 4:  [conv3x3(256) - BN - ReLU] x2 - maxpool2   -> 256 x 8 x 8
    head:     global average pool - dropout(0.3) - linear(256 -> 3)
About 1.2 M parameters. Overfitting is controlled with data augmentation, batch norm, dropout,
weight decay (AdamW), label smoothing and early stopping on validation macro-F1.

    python scratch_cnn.py --epochs 100
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from common import CLASSES, MODELS_DIR, SEED, save_json, set_seed
from train_utils import TorchPredictor, count_params, fit, get_device, make_loader, val_metrics_of

IMG_SIZE = 128
MODEL_PATH = MODELS_DIR / "scratch_cnn.pt"


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class WasteNet(nn.Module):
    def __init__(self, n_classes: int = len(CLASSES), widths=(32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        chans = (3,) + tuple(widths)
        self.features = nn.Sequential(*[conv_block(chans[i], chans[i + 1]) for i in range(len(widths))])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(widths[-1], n_classes))
        for m in self.modules():  # He initialisation for ReLU networks
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def train(epochs=100, batch_size=64, lr=2e-3, weight_decay=5e-4, patience=10, num_workers=None, device=None) -> dict:
    set_seed(SEED)
    device = get_device(device)
    print(f"training WasteNet from scratch on {device}")
    model = WasteNet()
    tr = make_loader("train", IMG_SIZE, batch_size, train=True, num_workers=num_workers)
    va = make_loader("val", IMG_SIZE, batch_size, train=False, num_workers=num_workers)
    out = fit(model, tr, va, device, epochs=epochs, lr=lr, weight_decay=weight_decay, patience=patience, log_prefix="[scratch] ")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "img_size": IMG_SIZE}, MODEL_PATH)
    meta = {
        "method": "scratch",
        "architecture": "WasteNet 4-block CNN",
        "image_size": IMG_SIZE,
        "params_millions": count_params(model),
        "hyperparameters": {"epochs": epochs, "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
                             "patience": patience, "optimizer": "AdamW", "schedule": "cosine", "label_smoothing": 0.1},
        "best_epoch": out["best_epoch"],
        "train_time_min": out["train_time_min"],
        "device": str(device) + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
        "val": val_metrics_of(model, IMG_SIZE, device),
        "history": out["history"],
    }
    save_json(meta, MODELS_DIR / "scratch_cnn_meta.json")
    print(f"saved {MODEL_PATH} | params {meta['params_millions']:.2f} M | best epoch {out['best_epoch']}"
          f" | val F1 {meta['val']['macro_f1']:.2f}% | {out['train_time_min']:.1f} min")
    return meta


def load_predictor(device=None, path=MODEL_PATH) -> TorchPredictor:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = WasteNet()
    model.load_state_dict(ckpt["state_dict"])
    return TorchPredictor("scratch", model, ckpt.get("img_size", IMG_SIZE), device)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    a = ap.parse_args(argv)
    train(a.epochs, a.batch_size, a.lr, num_workers=a.workers, device=a.device)


if __name__ == "__main__":
    main()
