| Method | Slot | Top-1 acc (%) | Macro F1 (%) | Runtime CPU (ms/img) | Runtime GPU (ms/img) | Params (M) | MACs (M/img) | Train time (min) |
|---|---|---|---|---|---|---|---|---|
| HOG + colour + LBP (spatial pyramid) -> RBF-kernel SVM (random Fourier features) | A. Classical | 79.82 | 80.33 | 10.63 | - | 0.0 | - | 3.98 |
| WasteNet-R: residual CNN (ours) | B. Trained from scratch | 86.39 | 86.96 | 14.25 | 4.73 | 1.597 | 635 | 19.51 |
| MobileNetV3-Large (ImageNet) fine-tuned | C. Pretrained, modern | 97.55 | 97.59 | 27.13 | 8.37 | 4.206 | 215 | 4.71 |

Runtime = median time to classify one 256x256 image (batch 1), CPU: Intel64 Family 6 Model 186 Stepping 2, GenuineIntel, GPU: NVIDIA GeForce RTX 4050 Laptop GPU. Params = 0 for the classical method (rubric convention). MACs = multiply-accumulate operations per image in the network's conv/linear layers (the compute that decides CPU speed; more parameters does not always mean slower).
