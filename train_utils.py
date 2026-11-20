"""PyTorch helpers shared by the two neural methods (scratch CNN and pretrained model).

Both networks get exactly the same data, augmentation, loss, early-stopping rule and metrics,
so the only thing that differs between them is the model itself.
"""
from __future__ import annotations

import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from common import CLASSES, class_weights, classification_metrics, load_split, softmax

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def get_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_transform(size: int):
    """Augmentation against overfitting: random crop/zoom, flips, rotation and lighting changes
    (a webcam in a classroom never sees an item at exactly the same scale, angle or brightness)."""
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(20),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(size: int):
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class WasteDataset(Dataset):
    def __init__(self, df, transform):
        self.paths = df["abs_path"].tolist()
        self.labels = df["label_idx"].to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = self.transform(im.convert("RGB"))
        return x, int(self.labels[i])


def default_workers() -> int:
    """DataLoader worker processes (override with the COMVIS_WORKERS environment variable)."""
    if os.environ.get("COMVIS_WORKERS"):
        return int(os.environ["COMVIS_WORKERS"])
    n = os.cpu_count() or 2
    return min(4, max(0, n - 1))


def make_loader(split: str, size: int, batch_size: int, train: bool, num_workers: int | None = None) -> DataLoader:
    df = load_split(split)
    ds = WasteDataset(df, train_transform(size) if train else eval_transform(size))
    nw = default_workers() if num_workers is None else num_workers
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=nw, pin_memory=torch.cuda.is_available(),
                      persistent_workers=nw > 0, drop_last=False)


def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    ys, probs, losses = [], [], []
    crit = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = model(x)
        losses.append(crit(out.float(), y).item())
        probs.append(torch.softmax(out.float(), 1).cpu().numpy())
        ys.append(y.cpu().numpy())
    y_true, p = np.concatenate(ys), np.concatenate(probs)
    return y_true, p, sum(losses) / len(y_true)


def fit(model, train_loader, val_loader, device, epochs=40, lr=1e-3, weight_decay=1e-4, patience=8,
        label_smoothing=0.1, param_groups=None, log_prefix="") -> dict:
    """AdamW + cosine LR, class-weighted cross-entropy with label smoothing, mixed precision on GPU,
    early stopping on validation macro-F1 (the best weights are restored at the end)."""
    model.to(device)
    w = torch.tensor(class_weights(train_loader.dataset.labels), device=device)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)
    opt = torch.optim.AdamW(param_groups or model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "lr"]}
    best_f1, best_state, best_epoch, bad = -1.0, None, 0, 0
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        tot, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                out = model(x)
                loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            tot += len(y)
        sched.step()
        y_true, probs, val_loss = predict_loader(model, val_loader, device)
        m = classification_metrics(y_true, probs.argmax(1))
        hist["train_loss"].append(loss_sum / tot)
        hist["train_acc"].append(correct / tot * 100)
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(m["accuracy"])
        hist["val_f1"].append(m["macro_f1"])
        hist["lr"].append(opt.param_groups[0]["lr"])
        flag = ""
        if m["macro_f1"] > best_f1:
            best_f1, best_state, best_epoch, bad, flag = m["macro_f1"], copy.deepcopy(model.state_dict()), ep, 0, " *"
        else:
            bad += 1
        print(f"{log_prefix}epoch {ep:3d}/{epochs}  train loss {hist['train_loss'][-1]:.3f} acc {hist['train_acc'][-1]:5.1f}"
              f" | val loss {val_loss:.3f} acc {m['accuracy']:5.1f} F1 {m['macro_f1']:5.1f}{flag}", flush=True)
        if bad >= patience:
            print(f"{log_prefix}early stop: no val F1 improvement for {patience} epochs")
            break
    model.load_state_dict(best_state)
    return {"history": hist, "best_epoch": best_epoch, "best_val_f1": best_f1, "train_time_min": (time.perf_counter() - t0) / 60}


class TorchPredictor:
    """Wraps a trained network for the app / evaluation: list of 256x256 uint8 RGB arrays -> class probabilities."""

    def __init__(self, key: str, model: nn.Module, img_size: int, device=None):
        self.key, self.model, self.img_size = key, model.eval(), img_size
        self.tf = eval_transform(img_size)
        self.n_params = count_params(model)
        self.to(device or get_device())

    def to(self, device):
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    @torch.no_grad()
    def predict_proba(self, images: list[np.ndarray]) -> np.ndarray:
        x = torch.stack([self.tf(Image.fromarray(im)) for im in images]).to(self.device)
        return softmax(self.model(x).float().cpu().numpy())


def val_metrics_of(model, img_size, device, batch_size=64) -> dict:
    loader = make_loader("val", img_size, batch_size, train=False, num_workers=0)
    y, p, _ = predict_loader(model.to(device), loader, device)
    return classification_metrics(y, p.argmax(1))


__all__ = ["CLASSES", "get_device", "make_loader", "fit", "count_params", "TorchPredictor", "predict_loader",
           "train_transform", "eval_transform", "val_metrics_of"]
