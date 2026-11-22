"""Model C - ResNet-18 (He et al., "Deep Residual Learning for Image Recognition", CVPR 2016), ImageNet-1k weights
from torchvision, fine-tuned on our three bins with the shared recipe in finetune.py.

Why this model: the classic residual network. Plain 3x3 convolutions in pairs, each pair added to its own input
(skip connection), which lets the gradient reach the early layers and made very deep networks trainable. It uses
full (not depthwise) convolutions throughout: about 1.8 G multiply-adds per 224x224 image and about 11 M parameters,
the largest and oldest of the three - the baseline the two mobile designs try to beat on speed.

Head: `fc` (512 -> 1000 ImageNet classes) is replaced by 512 -> 3 bins.

    python resnet18.py
"""
from __future__ import annotations

import argparse

import torch.nn as nn
from torchvision import models

import finetune
from common import CLASSES

KEY = "resnet18"


def build_model(pretrained: bool = True) -> nn.Module:
    weights = finetune.imagenet(models.ResNet18_Weights.IMAGENET1K_V1) if pretrained else None
    m = models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    return m


def head(model: nn.Module) -> nn.Module:
    return model.fc


def train(**kwargs) -> dict:
    model = build_model()
    return finetune.finetune(KEY, model, head(model), **kwargs)


def load_predictor(device=None):
    return finetune.load_predictor(KEY, build_model(pretrained=False), device)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    finetune.add_train_args(ap)
    train(**finetune.train_kwargs(ap.parse_args(argv)))


if __name__ == "__main__":
    main()
