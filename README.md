# BasuraBin: waste segregation with three pretrained CNNs

WVSU Computer Vision Comparison Project. One task, three pretrained models, same data, same splits, same fine-tuning recipe,
same metrics, plus an app that runs all three on a photo, a video or the live webcam.

**Task:** classify a photo of one household waste item into the three RA 9003 bins:

| Bin | Examples | Colour in the app |
|---|---|---|
| **Biodegradable** (*nabubulok*) | food scraps, peelings, leaves | green |
| **Recyclable** | paper, cardboard, cans, glass, plastic bottles | blue |
| **Non-biodegradable / Residual** (*di-nabubulok*) | sachets, wrappers, styrofoam, old textiles | black/grey |

"Non-biodegradable" here means **residual**: neither recyclable nor compostable. A plastic bottle is non-biodegradable but goes to Recyclable.

**Research question:** for a real-time camera app, which ImageNet-pretrained architecture gives the best trade-off between accuracy,
speed and size?

| Model | Year | Design idea | Size | File |
|---|---|---|---|---|
| A. MobileNetV3-Large | 2019 | built for phones: depthwise-separable convolutions, squeeze-and-excitation | ~4 M params, 0.22 G MACs | `mobilenet_v3.py` |
| B. EfficientNet-B0 | 2019 | compound scaling of depth, width and resolution | ~4 M params, 0.39 G MACs | `efficientnet_b0.py` |
| C. ResNet-18 | 2016 | classic residual network, full 3×3 convolutions | ~11 M params, 1.8 G MACs | `resnet18.py` |

All three load ImageNet-1k weights from torchvision and are fine-tuned by the same recipe in `finetune.py`; only the architecture differs.

## Results (final run, held-out test set of 1,021 images)

| Model | Macro-F1 | Accuracy | CPU ms/img | GPU ms/img | Params | MACs | Train time |
|---|---|---|---|---|---|---|---|
| **A. MobileNetV3-Large** (recommended) | **97.5%** | **97.5%** | **29.2** | 8.6 | 4.2 M | 215 M | 4.7 min |
| B. EfficientNet-B0 | 97.0% | 96.9% | 40.8 | 10.5 | 4.0 M | 385 M | 4.9 min |
| C. ResNet-18 | 96.7% | 96.6% | 50.3 | **5.3** | 11.2 M | 1,814 M | 4.7 min |

Runtime: one 256×256 photo (batch 1), median of 5 rounds in which the three models alternate, on the training laptop
(Intel laptop CPU, NVIDIA RTX 4050 Laptop GPU). Training time on that GPU.

* **MobileNetV3 is recommended:** the most accurate, the fastest on a CPU and the most robust, although it is the smallest model.
* **Fine-tuning matters most for the larger models.** Validation macro-F1 with only the new head trained on frozen ImageNet features
  vs after fine-tuning the whole network: MobileNetV3 94.4% → 98.0%, EfficientNet-B0 89.4% → 97.2%, ResNet-18 87.6% → 96.4%.
* **Robustness** (test macro-F1 under corruptions): all three stay at 94-97% under blur, darkness and a 30° rotation, but under
  sensor noise EfficientNet-B0 drops to 73% (MobileNetV3 91%, ResNet-18 89%), and ResNet-18 drops to 91% on over-bright photos.
* **Photo source:** the real landfill photos (RealWaste) are about 3-4 points harder than the web photos for all three
  (MobileNetV3: 95.5% vs 98.7% accuracy).
* **CPU vs GPU:** ResNet-18 needs 8.4× the multiply-adds of MobileNetV3 and is the slowest on a CPU, yet the fastest on a GPU:
  its full convolutions suit a GPU, while the depthwise layers of the two mobile designs are limited by memory traffic.

Full numbers and figures: `results/results_table.md`, `results/results.json`, `results/*.png` and the executed `CV_Project.ipynb`.

---

## Quick start: just run the app (no GPU, no dataset, no training)

The three fine-tuned models are in `models\`, so a fresh clone can run the app on any Windows laptop with **Python 3.10-3.12** and **Git**.
In PowerShell:

```powershell
git clone https://github.com/nylla8444/comvis-final.git
```
```powershell
cd comvis-final
```
```powershell
python -m venv .venv
```
```powershell
.venv\Scripts\python -m pip install -r requirements.txt
```
```powershell
.venv\Scripts\python app.py
```

The install downloads a few hundred MB (PyTorch, Gradio, OpenCV ...) the first time. The app then opens in the browser at
http://127.0.0.1:7860 and runs on the CPU when there is no GPU.

* **Photo** tab: upload a photo or take one with the webcam. (The test-set example thumbnails only appear on the laptop that has the dataset.)
* **Live camera** tab: allow camera access, hold one item inside the square.
* `.venv\Scripts\python live.py` opens the full-window webcam demo instead.
* To get the newest version later: `git pull`.
* macOS / Linux: use `.venv/bin/python` instead of `.venv\Scripts\python`.

Everything below is only needed to rebuild the dataset and retrain the models.

## 1. Setup on the GPU laptop (Windows, PowerShell)

```powershell
git clone https://github.com/nylla8444/comvis-final.git
cd comvis-final
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install PyTorch **with CUDA** first. Pick the command for the laptop's driver at https://pytorch.org/get-started/locally/, for example:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

```powershell
pip install -r requirements.txt
```

Check that the GPU is visible (must print `True` and the GPU name):

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. Get the data (about 940 MB, no Kaggle login needed)

```powershell
python download_data.py
```

Or download in a browser and unzip yourself:

| Dataset | Direct download | Unzip into |
|---|---|---|
| Garbage Classification (Kaggle, 15,515 images, 251 MB) | https://www.kaggle.com/api/v1/datasets/download/mostafaabla/garbage-classification | `data\raw\garbage_classification\` |
| RealWaste (UCI #908, CC BY 4.0, 4,752 images, 689 MB) | https://www.kaggle.com/api/v1/datasets/download/joebeachcapital/realwaste | `data\raw\realwaste\` |

(Dataset pages for citing: https://www.kaggle.com/datasets/mostafaabla/garbage-classification and https://archive.ics.uci.edu/dataset/908/realwaste.
If you download by hand, you can also drop the zips into `data\downloads\` as `garbage-classification.zip` / `realwaste.zip` and run `python download_data.py` to unzip.)

### Optional: our own photos

The datasets are mostly clean web photos and landfill photos; a classroom webcam looks different (hand, background, lighting).
Our own photos show how well the models handle that, and can also teach them. Two folders, one item per photo:

```
data\raw\local\<bin>\         extra TEST set, never trained on: measures real-world accuracy
data\raw\local_train\<bin>\   added to the TRAINING split for all three models
<bin> = biodegradable (banana peel, rice, leaves ...), recyclable (bottles, cans, cardboard ...), residual (sachets, wrappers, styrofoam ...)
```

Use **different objects** in the two folders (an identical photo found in both is dropped from the test set). Aim for 30+ photos per
bin in each, with the classroom background and light. The webcam window saves them for you (needs trained models):

```powershell
python live.py                   # keys 1 / 2 / 3 save the frame into data\raw\local\<bin>\ (test photos)
```
```powershell
python live.py --save-to train   # keys 1 / 2 / 3 save into data\raw\local_train\<bin>\ (training photos)
```

Phone photos can be copied into the same folders. Then run `prepare_data.py` and retrain (or rerun the notebook).

## 3. Build the dataset and splits

```powershell
python prepare_data.py
```

* maps the source classes to the 3 bins (batteries dropped: special/hazardous waste)
* removes duplicate images, caps each bin at 2,500 images sampled evenly across source classes
* writes `splits\train.csv`, `val.csv`, `test.csv` (70/15/15, seed 42) and `splits\summary.json`
* stores every image as a 256×256 centre crop in `data\processed\`
* adds `data\raw\local_train\` photos to the train split and writes `data\raw\local\` photos to `splits\local_test.csv` (if present)

Expected total: about 6,800 images (above the 5,000 minimum). The exact numbers are printed and saved in `splits\summary.json`.

## 4. Fine-tune the three models (same splits and recipe for all)

```powershell
python finetune.py              # all three, one after another (~15 min on an RTX 4050)
```

or one at a time: `python mobilenet_v3.py`, `python efficientnet_b0.py`, `python resnet18.py`.

The shared recipe (`finetune.py`): centre crop → 224×224 → ImageNet normalisation; augmentation (crop/zoom, flips, rotation, colour
jitter, blur); stage 1 trains only the new 3-class head on frozen ImageNet features (3 epochs), stage 2 fine-tunes the whole network
with a 10× smaller learning rate on the backbone (up to 15 epochs); class-weighted cross-entropy with label smoothing, AdamW, cosine
schedule, early stopping on validation macro-F1.

Each model is saved as `models\<model>.pt` plus `models\<model>_meta.json` (training curves, params, MACs, training time, validation
metrics, and the head-only validation score).

## 5. Evaluate (fills the results table)

```powershell
python evaluate.py
```

Writes to `results\`:

| File | For the report |
|---|---|
| `results_table.md` / `.csv` | the comparison table: accuracy, macro-F1, runtime (CPU & GPU), params, MACs (compute per image), train time |
| `confusion_matrices.png` | confusion matrix per model |
| `training_curves.png` | train vs validation loss/accuracy for every model (dotted line = backbone unfrozen) |
| `robustness.png` | macro-F1 under blur / dark / bright / noise / rotation |
| `misclassified_*.png` | failure cases for the error analysis |
| `results.json` | all numbers + the recommendation the app shows |

Experiments: (1) main comparison, (2) efficiency, (3) accuracy by photo source (web vs landfill), (4) robustness to corruptions,
(5) frozen ImageNet features vs full fine-tuning, (6) our own photos if provided.

The app recommends the **fastest model on the CPU among those within 1 macro-F1 point of the most accurate one**: with about
1,000 test images a 1-point gap is about the size of the test-set noise, while speed decides how smooth the live camera feels.

## 6. The notebook (deliverable)

`CV_Project.ipynb` runs everything above from top to bottom (download → prepare → fine-tune A, B, C → evaluate → figures).
It works in Google Colab (it clones this repo itself; choose *Runtime → Change runtime type → GPU*) and locally.

The deliverable must be the **executed** notebook (saved with its outputs). The simplest way on the laptop runs every cell without
opening a browser and writes the outputs into the file (about 30 min on an RTX 4050; nothing is printed until it finishes with
`Writing ... bytes to CV_Project.ipynb`):

```powershell
python -m pip install notebook
```
```powershell
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 CV_Project.ipynb
```

(`python -m` avoids the "jupyter is not recognized" error when pip's Scripts folder is not on PATH. Or open it with
`python -m jupyter notebook CV_Project.ipynb`, choose *Run → Run All Cells*, and save when it finishes.)

The notebook retrains all three models, so its numbers replace the ones in `results\`. Commit the notebook, the outputs and the
models together (`models` holds the three fine-tuned models the app loads, about 80 MB, plus their training logs):

```powershell
git add CV_Project.ipynb results splits models
```
```powershell
git commit -m "chore(results): add run of the three pretrained models"
```
```powershell
git push
```

## 7. Run the app

```powershell
python app.py          # browser app: Photo / Live camera / Video / Compare methods / About tabs
python live.py         # full-window OpenCV webcam demo (most reliable for the class demo)
```

* `python app.py --share` prints a temporary public link (handy when controlling the laptop remotely).
* `python app.py --cpu` / `python live.py --cpu` run on the CPU to show real-time speed without a GPU.
* `live.py` keys: `q` quit, `s` snapshot, `m` mirror, `1`/`2`/`3` save the frame as a biodegradable/recyclable/residual photo (see step 2). `--camera 1` for a USB webcam.
* In the live webcam modes each model's answer is averaged over the last few frames, so the label does not flicker (`live.py --no-smooth` shows the raw per-frame answer).

**Demo tips:** hold **one** item inside the square, fill most of the square, use a plain background (bond paper or a table) and good light.
Below 55% confidence the app says *Not sure* instead of guessing.

---

## Files

```
common.py          shared constants: bins, colours, the three models, preprocessing, metrics, timing
download_data.py   downloads + unzips both datasets
prepare_data.py    3-bin mapping, de-duplication, balancing, splits, 256x256 cache, own photos
train_utils.py     shared PyTorch training loop, augmentation, MAC counter, predictor wrapper
finetune.py        the shared two-stage fine-tuning recipe (python finetune.py trains all three)
mobilenet_v3.py    Model A: MobileNetV3-Large
efficientnet_b0.py Model B: EfficientNet-B0
resnet18.py        Model C: ResNet-18
methods.py         loads the trained models behind one predict_proba() interface
evaluate.py        experiments, results table, plots, recommendation
annotate.py        draws results on images/frames, smooths live predictions
app.py             Gradio app (photo, live camera, video, comparison, about)
live.py            OpenCV real-time webcam window + photo capture
CV_Project.ipynb   the end-to-end notebook
models/            the three fine-tuned models the app loads + training logs (*_meta.json)
results/           results table, figures and results.json (the recommendation the app shows)
splits/            the fixed train/val/test lists used by all three models
```

## Troubleshooting

* **`torch.cuda.is_available()` is False:** you installed the CPU build of PyTorch. Uninstall (`pip uninstall torch torchvision`) and reinstall with the CUDA command from step 1.
* **Out of GPU memory:** add `--batch-size 32` to `finetune.py` (or to one model's script).
* **DataLoader errors or slow start on Windows:** set `$env:COMVIS_WORKERS=0` before training.
* **Webcam not found in `live.py`:** try `--camera 1`. Close other apps using the camera (Zoom, Teams, the browser tab of `app.py`).
* **Webcam tab in the browser asks for permission:** allow it. It only works on `http://127.0.0.1` / `localhost` or the `--share` https link.

## Credits

* Garbage Classification (12 classes), Kaggle user mostafaabla, https://www.kaggle.com/datasets/mostafaabla/garbage-classification
* S. Single, S. Iranmanesh, R. Raad, "RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning", *Information* 14(12):633, 2023. UCI ML Repository #908, CC BY 4.0.
* A. Howard et al., "Searching for MobileNetV3", ICCV 2019.
* M. Tan, Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML 2019.
* K. He, X. Zhang, S. Ren, J. Sun, "Deep Residual Learning for Image Recognition", CVPR 2016.
* Pretrained ImageNet-1k weights from torchvision (https://pytorch.org/vision/stable/models.html).

## AI-assistance disclosure

Parts of this code were written with an AI coding assistant (Claude). Per the project rules, the contribution statement must say how it
was used, and every member must be able to explain every line in the Q&A.
