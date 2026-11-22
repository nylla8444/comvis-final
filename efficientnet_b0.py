"""Model B - EfficientNet-B0 (Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
ICML 2019), ImageNet-1k weights from torchvision, fine-tuned on our three bins with the shared recipe in finetune.py.

Why this model: its building block (MBConv: an inverted residual with a depthwise convolution, squeeze-and-excitation
and the swish activation) was found by neural architecture search, and the family is grown by "compound scaling":
depth, width and input resolution are scaled together instead of one at a time. B0 is the smallest member.
About 0.39 G multiply-adds per 224x224 image, about 4 M parameters with our 3-class head: the best ImageNet accuracy
per parameter of the three, but its many depthwise and squeeze-and-excitation layers are memory-bound on a CPU.

Head: `classifier` = dropout + linear (1280 -> 1000 ImageNet classes); the linear layer is replaced by 1280 -> 3 bins.

    python efficientnet_b0.py
"""
from __future__ import annotations

import argparse

import torch.nn as nn
from torchvision import models

import finetune
from common import CLASSES

KEY = "efficientnet_b0"


def build_model(pretrained: bool = True) -> nn.Module:
    weights = finetune.imagenet(models.EfficientNet_B0_Weights.IMAGENET1K_V1) if pretrained else None
    m = models.efficientnet_b0(weights=weights)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASSES))
    return m


def head(model: nn.Module) -> nn.Module:
    return model.classifier


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
