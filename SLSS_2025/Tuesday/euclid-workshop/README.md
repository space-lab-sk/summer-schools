# Euclid Galaxy Segmentation — Summer School Workshop

Welcome to the workshop! 🚀
In this hands-on lab you will:

* Create **classical masks** of galaxies from Euclid cutouts (threshold + morphology).
* Train a tiny **U-Net** (or similar) to predict masks.
* Use **AI coding assistance** (Windsurf / Codeium plugin) to generate most of the code in `.py` files.

> ⚠️ **Important:** The repository contains only **scaffolds** (skeleton code with TODOs).
> Your task is to **use AI to complete them**.

---

## 1. Environment Setup (Python 3.12)

We will use **Conda** or **venv** for a clean environment.

### Option A — Conda (recommended)

```bash
conda create -n euclid-seg python=3.12 -y
conda activate euclid-seg
pip install -r requirements.txt
```

### Option B — venv (built-in)

```bash
python3.12 -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Verify installation

```bash
python -c "import tensorflow as tf, cv2, numpy; print('TF version:', tf.__version__)"
```

---

## 2. AI Coding Assistant

We will use **Windsurf (by Codeium)** or the **Codeium AI plugin** in your favorite editor.

### Supported editors

* **VS Code** → Codeium extension
* **PyCharm / IntelliJ** → Codeium plugin
* **Neovim** → Codeium extension
* **Windsurf IDE** → stand-alone AI editor

> You may also use any other AI code assistant if you prefer.
> The important part is: work in `.py` files, not in notebooks.

### Tips

* Highlight a TODO block and **ask AI to implement it**.
* Keep prompts short and precise. Example:

  > “Implement Otsu thresholding with OpenCV to create a binary mask from grayscale input.”
* Iterate: if code doesn’t work, refine the prompt or ask for fixes.

---

## 3. Project Structure

```
euclid-summer-workshop/
├── data/
│   ├── images/                 # galaxy cutouts (sample provided)
│   └── masks/                  # generated masks
├── notebooks/
│   └── 00_driver.ipynb         # viseualize, run scripts
├── src/
│   ├── traditional_segmentation.py   # classical masks (OpenCV)
│   ├── unet_model.py                  # tiny U-Net (Keras)
│   ├── train_unet.py                  # training loop
│   └── infer_unet.py                  # inference script
└── utils/
    └── dataset.py                     # tf.data helpers
```

---

## 4. Scaffold Files & Example Prompts

### `src/traditional_segmentation.py`

Goal: Create masks with **OpenCV**.

* Grayscale → threshold (Otsu/adaptive/fixed) → morphology cleanup
* Save `<name>_mask.png`

💡 Prompt:

> “Write an OpenCV CLI tool that thresholds galaxy cutouts, applies morphology, and saves binary masks.”

---

### `src/unet_model.py`

Goal: Tiny **U-Net** in Keras.

* Input: 256×256×1
* Conv2D+BN+ReLU blocks, MaxPooling downs, UpSampling ups
* Output: 1-channel logits

💡 Prompt:

> “Build a small U-Net in Keras with skip connections, returning logits.”

---

### `src/train_unet.py`

Goal: Training loop in Keras.

* Use `utils.dataset.make_dataset`
* Loss: `BinaryCrossentropy(from_logits=True)`
* Metric: Dice
* Save checkpoints to `runs/`

💡 Prompt:

> “Create a Keras training script for U-Net with BCE loss and Dice metric, saving best model to runs/best.keras.”

---

### `src/infer_unet.py`

Goal: Inference on images.

* Load model
* Run prediction
* Apply sigmoid+0.5 threshold
* Save `<name>_pred.png`

💡 Prompt:

> “Implement batch inference for U-Net: load model, predict, apply threshold, save PNGs.”

---

### `utils/dataset.py`

Goal: tf.data pipeline.

* Pair images with \*\_mask.png
* Load grayscale float32 \[0,1]
* Resize to 256×256
* Return (image, mask)

💡 Prompt:

> “Write a tf.data pipeline that loads grayscale images and masks, resizes them to 256×256, and batches them.”

---

## 5. Workflow

1. **Classical masks**

   ```bash
   python src/traditional_segmentation.py --images_dir data/images --out_dir data/masks
   ```

2. **Train U-Net**

   ```bash
   python src/train_unet.py --images_dir data/images --masks_dir data/masks --epochs 3
   ```

3. **Inference**

   ```bash
   python src/infer_unet.py --images_dir data/images --checkpoint runs/best.keras --out_dir runs/preds
   ```

---
