| Method | Slot | Top-1 acc (%) | Macro F1 (%) | Runtime CPU (ms/img) | Runtime GPU (ms/img) | Params (M) | Train time (min) |
|---|---|---|---|---|---|---|---|
| HOG + HSV colour hist + LBP -> linear SVM | A. Classical | 77.08 | 77.67 | 5.67 | - | 0.0 | 2.83 |
| WasteNet: 4-block CNN (ours) | B. Trained from scratch | 88.64 | 88.9 | 40.76 | 5.87 | 1.174 | 9.75 |
| MobileNetV3-Large (ImageNet) fine-tuned | C. Pretrained, modern | 97.36 | 97.43 | 22.53 | 8.4 | 4.206 | 3.56 |

Runtime = median time to classify one 256x256 image (batch 1), CPU: Intel64 Family 6 Model 186 Stepping 2, GenuineIntel, GPU: NVIDIA GeForce RTX 4050 Laptop GPU. Params = 0 for the classical method (rubric convention).
