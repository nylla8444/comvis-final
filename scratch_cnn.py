"""Method B - Trained from scratch: our own CNN written in PyTorch, random initialisation.

Default: WasteNet-R, a small residual network (input 3x160x160, ~1.6 M parameters):
    stem:     conv3x3(32, stride 2) - BN - ReLU                  -> 32 x 80 x 80
    stage 1:  1 residual block (32 channels)                     -> 32 x 80 x 80
    stage 2:  2 residual blocks (64), the first with stride 2    -> 64 x 40 x 40
    stage 3:  2 residual blocks (128), the first with stride 2   -> 128 x 20 x 20
    stage 4:  1 residual block (256), stride 2                   -> 256 x 10 x 10
    head:     global average pool - dropout(0.3) - linear(256 -> 3)
    residual block = conv3x3-BN-ReLU-conv3x3-BN, added to its input (1x1 conv if the shape changes), then ReLU

Why residual: the skip connections (He et al. 2016) let the gradient reach the early layers directly, so a deeper
network trains as easily as a shallow one. The stride-2 stem shrinks the image once at the start, so WasteNet-R
needs fewer multiply-adds per image than the first version even though it sees larger 160x160 inputs.

The first version, `--arch plain` (WasteNet: 4 plain blocks [conv3x3-BN-ReLU]x2 + max-pool, 32/64/128/256 channels,
input 128x128, ~1.2 M parameters), is kept so the two designs can be compared.

Overfitting is controlled with data augmentation, batch norm, dropout, weight decay (AdamW), label smoothing and
early stopping on validation macro-F1.

    python scratch_cnn.py                  # WasteNet-R, up to 150 epochs
    python scratch_cnn.py --arch plain     # the first version
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from common import CLASSES, MODELS_DIR, SEED, save_json, set_seed
from train_utils import TorchPredictor, count_macs, count_params, fit, get_device, make_loader, val_metrics_of

MODEL_PATH = MODELS_DIR / "scratch_cnn.pt"
ARCHS = {  # name -> (display name, input size)
    "resnet": ("WasteNet-R: residual CNN (ours)", 160),
    "plain": ("WasteNet: plain 4-block CNN (ours)", 128),
}
IMG_SIZE = ARCHS["resnet"][1]


def _init_weights(model: nn.Module) -> None:
    for m in model.modules():  # He initialisation for ReLU networks
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)


# ---- first version: plain CNN --------------------------------------------------------------
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
        _init_weights(self)

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


# ---- default: residual CNN ---------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity() if stride == 1 and c_in == c_out else nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class WasteResNet(nn.Module):
    def __init__(self, n_classes: int = len(CLASSES), widths=(32, 64, 128, 256), blocks=(1, 2, 2, 1),
                 dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        layers, c_in = [], widths[0]
        for i, (w, n) in enumerate(zip(widths, blocks)):
            for j in range(n):
                layers.append(ResidualBlock(c_in, w, stride=2 if (j == 0 and i > 0) else 1))
                c_in = w
        self.stages = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(widths[-1], n_classes))
        _init_weights(self)
        for m in self.modules():  # start every residual branch at zero, so each block begins as an identity
            if isinstance(m, ResidualBlock):
                nn.init.zeros_(m.bn2.weight)

    def forward(self, x):
        return self.head(self.pool(self.stages(self.stem(x))))


def build_model(arch: str = "resnet") -> nn.Module:
    return WasteResNet() if arch == "resnet" else WasteNet()


def train(arch="resnet", epochs=150, batch_size=64, lr=2e-3, weight_decay=5e-4, patience=25, num_workers=None,
          device=None) -> dict:
    set_seed(SEED)
    device = get_device(device)
    name, img_size = ARCHS[arch]
    print(f"training {name} from scratch on {device}")
    model = build_model(arch)
    tr = make_loader("train", img_size, batch_size, train=True, num_workers=num_workers)
    va = make_loader("val", img_size, batch_size, train=False, num_workers=num_workers)
    out = fit(model, tr, va, device, epochs=epochs, lr=lr, weight_decay=weight_decay, patience=patience, log_prefix="[scratch] ")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "arch": arch, "img_size": img_size}, MODEL_PATH)
    meta = {
        "method": "scratch",
        "arch": arch,
        "architecture": name,
        "display_name": name,
        "image_size": img_size,
        "params_millions": count_params(model),
        "macs_millions": count_macs(model, img_size),
        "hyperparameters": {"epochs": epochs, "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
                             "patience": patience, "optimizer": "AdamW", "schedule": "cosine", "label_smoothing": 0.1},
        "best_epoch": out["best_epoch"],
        "train_time_min": out["train_time_min"],
        "device": str(device) + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
        "val": val_metrics_of(model, img_size, device),
        "history": out["history"],
    }
    save_json(meta, MODELS_DIR / "scratch_cnn_meta.json")
    print(f"saved {MODEL_PATH} | params {meta['params_millions']:.2f} M | {meta['macs_millions']:.0f} M MACs/img"
          f" | best epoch {out['best_epoch']} | val F1 {meta['val']['macro_f1']:.2f}% | {out['train_time_min']:.1f} min")
    return meta


def load_predictor(device=None, path=MODEL_PATH) -> TorchPredictor:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    arch = ckpt.get("arch", "plain")  # checkpoints from the first version have no "arch" field
    model = build_model(arch)
    model.load_state_dict(ckpt["state_dict"])
    return TorchPredictor("scratch", model, ckpt.get("img_size", ARCHS[arch][1]), device)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=list(ARCHS), default="resnet")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    a = ap.parse_args(argv)
    train(a.arch, a.epochs, a.batch_size, a.lr, patience=a.patience, num_workers=a.workers, device=a.device)


if __name__ == "__main__":
    main()
