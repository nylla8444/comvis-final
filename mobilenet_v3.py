"""Model A - MobileNetV3-Large (Howard et al., "Searching for MobileNetV3", ICCV 2019), ImageNet-1k weights from
torchvision, fine-tuned on our three bins with the shared recipe in finetune.py.

Why this model: it was designed for phones. It uses depthwise-separable convolutions (one 3x3 filter per channel,
then a 1x1 convolution that mixes the channels), squeeze-and-excitation (the network re-weights its channels for each
image) and the cheap h-swish activation; the block layout itself was found by neural architecture search.
About 0.22 G multiply-adds per 224x224 image: the lightest of the three, the natural candidate for a live webcam app.

Head: the `classifier` block (960 -> 1280 -> 1000 ImageNet classes); its last layer is replaced by 1280 -> 3 bins.

    python mobilenet_v3.py
"""
from __future__ import annotations

import argparse

import torch.nn as nn
from torchvision import models

import finetune
from common import CLASSES

KEY = "mobilenet_v3_large"


def build_model(pretrained: bool = True) -> nn.Module:
    weights = finetune.imagenet(models.MobileNet_V3_Large_Weights.IMAGENET1K_V2) if pretrained else None
    m = models.mobilenet_v3_large(weights=weights)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(CLASSES))
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
