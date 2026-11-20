# BasuraBin: waste segregation with 3 computer-vision methods

WVSU Computer Vision Comparison Project. One task, three methods, same data, same splits, same metrics,
plus an app that runs all three on a photo, a video or the live webcam.

**Task:** classify a photo of one household waste item into the three RA 9003 bins:

| Bin | Examples | Colour in the app |
|---|---|---|
| **Biodegradable** (*nabubulok*) | food scraps, peelings, leaves | green |
| **Recyclable** | paper, cardboard, cans, glass, plastic bottles | blue |
| **Non-biodegradable / Residual** (*di-nabubulok*) | sachets, wrappers, styrofoam, old textiles | black/grey |

"Non-biodegradable" here means **residual**: neither recyclable nor compostable. A plastic bottle is non-biodegradable but goes to Recyclable.

| Slot | Method | File |
|---|---|---|
| A. Classical | HOG + HSV colour histogram + uniform LBP, linear SVM, **all in NumPy** | `classical.py` |
| B. Trained from scratch | WasteNet, our 4-block CNN (~1.2 M params), PyTorch | `scratch_cnn.py` |
| C. Pretrained, modern | MobileNetV3-Large, ImageNet weights (torchvision), fine-tuned | `pretrained.py` |

---

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

Optional but recommended: **our own local photos** as an extra test set (never used for training). Take phone photos of one item each and put them in:

```
data\raw\local\biodegradable\   (banana peel, rice, leaves, ...)
data\raw\local\recyclable\      (bottles, cans, cardboard, paper, ...)
data\raw\local\residual\        (sachets, candy wrappers, styrofoam, ...)
```

## 3. Build the dataset and splits

```powershell
python prepare_data.py
```

* maps the source classes to the 3 bins (batteries dropped: special/hazardous waste)
* removes duplicate images, caps each bin at 2,500 images sampled evenly across source classes
* writes `splits\train.csv`, `val.csv`, `test.csv` (70/15/15, seed 42) and `splits\summary.json`
* stores every image as a 256×256 centre crop in `data\processed\`

Expected total: about 6,800 images (above the 5,000 minimum). The exact numbers are printed and saved in `splits\summary.json`.

## 4. Train the three methods (same splits for all)

```powershell
python classical.py sanity      # sanity checks on synthetic images (rubric requirement)
python classical.py train       # ~4-7 min, CPU (grid of 10 regularisation / learning-rate settings)
python classical.py ablation    # optional: HOG vs colour vs texture
python scratch_cnn.py           # WasteNet, up to 100 epochs with early stopping (~10 min on GPU)
python pretrained.py            # MobileNetV3, 3 head epochs + up to 15 fine-tuning epochs
```

Each saves its model in `models\` plus a `*_meta.json` (training curves, params, training time, validation metrics).

## 5. Evaluate (fills the results table)

```powershell
python evaluate.py
```

Writes to `results\`:

| File | For the report |
|---|---|
| `results_table.md` / `.csv` | the comparison table: accuracy, macro-F1, runtime (CPU & GPU), params, train time |
| `confusion_matrices.png` | confusion matrix per method |
| `training_curves.png` | train vs validation loss/accuracy for every method |
| `robustness.png` | macro-F1 under blur / dark / bright / noise / rotation |
| `misclassified_*.png` | failure cases for the error analysis |
| `results.json` | all numbers + the recommendation the app shows |

## 6. The notebook (deliverable)

`CV_Project.ipynb` runs everything above from top to bottom (download → prepare → sanity check → train A, B, C → feature ablation → evaluate → figures).
It works in Google Colab (it clones this repo itself; choose *Runtime → Change runtime type → GPU*) and locally.

The deliverable must be the **executed** notebook (saved with its outputs). The simplest way on the laptop runs every cell without opening a browser and writes the outputs into the file (about 30 min on an RTX 4050):

```powershell
pip install notebook
```
```powershell
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 CV_Project.ipynb
```

(Or open it with `jupyter notebook CV_Project.ipynb`, choose *Run → Run All Cells*, and save when it finishes.)

The notebook retrains all three methods, so its numbers replace the ones in `results\`. Commit the notebook and the outputs together; `models` only adds the small `*_meta.json` training logs, the weights stay ignored:

```powershell
git add CV_Project.ipynb results splits models
```
```powershell
git commit -m "chore(results): add final run and executed notebook"
```
```powershell
git push
```

## 7. Run the app

```powershell
python app.py          # browser app: Image / Live webcam / Video / Model comparison tabs
python live.py         # full-window OpenCV webcam demo (most reliable for the class demo)
```

* `python app.py --share` prints a temporary public link (handy when controlling the laptop remotely).
* `python app.py --cpu` / `python live.py --cpu` run on the CPU to show real-time speed without a GPU.
* `live.py` keys: `q` quit, `s` snapshot, `m` mirror. `--camera 1` for a USB webcam.

**Demo tips:** hold **one** item inside the square, fill most of the square, use a plain background (bond paper or a table) and good light.
Below 55% confidence the app says *Not sure* instead of guessing.

---

## Files

```
common.py         shared constants: bins, colours, preprocessing, metrics, timing
download_data.py  downloads + unzips both datasets
prepare_data.py   3-bin mapping, de-duplication, balancing, splits, 256x256 cache
classical.py      Method A (NumPy HOG/HSV/LBP + SVM), sanity checks, ablation, feature figure
train_utils.py    shared PyTorch training loop (same augmentation/loss/early stopping for B and C)
scratch_cnn.py    Method B (WasteNet)
pretrained.py     Method C (MobileNetV3-Large; --arch efficientnet_b0 / resnet18 also work)
methods.py        loads the trained methods behind one predict_proba() interface
evaluate.py       experiments, results table, plots, recommendation
annotate.py       draws results on images/frames
app.py            Gradio app (upload image, upload video, live webcam)
live.py           OpenCV real-time webcam window
CV_Project.ipynb  the end-to-end notebook
```

## Troubleshooting

* **`torch.cuda.is_available()` is False:** you installed the CPU build of PyTorch. Uninstall (`pip uninstall torch torchvision`) and reinstall with the CUDA command from step 1.
* **Out of GPU memory:** add `--batch-size 32` to `scratch_cnn.py` / `pretrained.py`.
* **DataLoader errors or slow start on Windows:** set `$env:COMVIS_WORKERS=0` before training.
* **Webcam not found in `live.py`:** try `--camera 1`. Close other apps using the camera (Zoom, Teams, the browser tab of `app.py`).
* **Webcam tab in the browser asks for permission:** allow it. It only works on `http://127.0.0.1` / `localhost` or the `--share` https link.

## Data credits

* Garbage Classification (12 classes), Kaggle user mostafaabla, https://www.kaggle.com/datasets/mostafaabla/garbage-classification
* S. Single, S. Iranmanesh, R. Raad, "RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning", *Information* 14(12):633, 2023. UCI ML Repository #908, CC BY 4.0.
* MobileNetV3: A. Howard et al., "Searching for MobileNetV3", ICCV 2019. Weights from torchvision.

## AI-assistance disclosure

Parts of this code were written with an AI coding assistant (Claude). Per the project rules, the contribution statement must say how it
was used, and every member must be able to explain every line in the Q&A.
