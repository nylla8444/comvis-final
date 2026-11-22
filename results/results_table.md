| Method | Slot | Top-1 acc (%) | Macro F1 (%) | Runtime CPU (ms/img) | Runtime GPU (ms/img) | Params (M) | MACs (M/img) | Train time (min) |
|---|---|---|---|---|---|---|---|---|
| MobileNetV3-Large (ImageNet, fine-tuned) | A. MobileNetV3-Large | 97.45 | 97.52 | 29.15 | 8.58 | 4.206 | 215 | 4.69 |
| EfficientNet-B0 (ImageNet, fine-tuned) | B. EfficientNet-B0 | 96.87 | 96.96 | 40.81 | 10.48 | 4.011 | 385 | 4.87 |
| ResNet-18 (ImageNet, fine-tuned) | C. ResNet-18 | 96.57 | 96.7 | 50.31 | 5.27 | 11.178 | 1814 | 4.73 |

Runtime = median time to classify one 256x256 image (batch 1), the models timed in 5 alternating rounds on the same images; CPU: Intel64 Family 6 Model 186 Stepping 2, GenuineIntel, GPU: NVIDIA GeForce RTX 4050 Laptop GPU. MACs = multiply-accumulate operations per image in the network's conv/linear layers (the compute that decides CPU speed; more parameters does not always mean slower).
