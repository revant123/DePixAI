<div align="center">

# 🔍 DePix AI

**Detect pixelation in images — instantly and accurately.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-CNN-orange?style=flat-square&logo=tensorflow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow?style=flat-square&logo=scikit-learn)
![Cython](https://img.shields.io/badge/Cython-optimized-green?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square)

*Two detection engines. One goal: know whether your image is pixelated.*

</div>

---

## ✨ What is DePix AI?

DePix AI analyzes images and tells you whether they are pixelated — using either a **classical computer vision pipeline** or a **deep learning CNN**. Whether you need speed, explainability, or raw neural power, DePix AI has you covered.

---

## 🧠 Two Detection Approaches

| | Classical (`main.py`) | Deep Learning (`with_tensorflow/`) |
|---|---|---|
| **Engine** | scikit-learn + Canny edges | Keras CNN |
| **Speed** | ⚡ Fast | 🐢 Slower (GPU-friendly) |
| **Model file** | `model/pixelation_model.pkl` | `model/pixelation_cnn_model.h5` |
| **Input size** | Any | Resized to 128×128 |
| **Output** | Pixelated / Not pixelated | Pixelated / Not pixelated |

---

## 📁 Project Structure

```
DePix AI/
├── main.py                          # ✅ Recommended: classical ML detector
├── slow.py                          # Baseline heuristic detector
├── blocky_score.pyx                 # Cython-optimized blocky score function
├── blocky_score.c                   # Auto-generated C output from Cython
├── setup.py                         # Build script for Cython extension
├── requirements.txt                 # Python dependencies
│
├── input_images/                    # Sample test images
│   ├── pixelated-cloud.webp
│   ├── non.jpg
│   └── ...
│
├── model/
│   └── pixelation_model.pkl         # Trained scikit-learn model
│
└── with_tensorflow/
    ├── train_cnn_model.py           # Train the CNN from scratch
    ├── predict_with_cnn.py          # Run inference with trained CNN
    ├── model/
    │   └── pixelation_cnn_model.h5  # Saved Keras model
    └── pixel/
        ├── pixel/                   # Pixelated training samples (7 images)
        └── depixel/                 # Non-pixelated training samples (6 images)
```

---

## 🔬 How It Works

### 🖼️ Classical Pipeline — `main.py`

```
Image → Downscale → Canny Edge Detection → Block Feature Extraction → ML Prediction
                                                  ↓
                                     (variance + edge density per block)
                                                  ↓
                                    pixelation_model.pkl → verdict
                                                  ↓
                                  Heuristic fallback if confidence is low
```

### 🤖 CNN Pipeline — `with_tensorflow/`

```
Image → Resize to 128×128 → Normalize [0, 1] → CNN → sigmoid output → threshold 0.5 → verdict
```

### ⚙️ Baseline Heuristic — `slow.py`

```
Image → Nearest-neighbor zoom → Canny edges → Block score (variance × edge density) → threshold
```

### 🚀 Cython Extension — `blocky_score.pyx`

An optimized implementation of the blocky-score function compiled to native C for performance.
> Currently built separately — intended for future integration into the main pipeline.

---

## ⚙️ Setup

> **Requires Python 3.8+**

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. For CNN inference and training

```powershell
pip install tensorflow pillow
```

### 4. (Optional) Build the Cython extension

```powershell
pip install cython
python setup.py build_ext --inplace
```

---

## 🚀 Run

### Classical detector *(recommended)*
```powershell
python main.py
```

### Baseline heuristic
```powershell
python slow.py
```

### CNN inference
```powershell
python with_tensorflow/predict_with_cnn.py
```

### Train the CNN from scratch
```powershell
python with_tensorflow/train_cnn_model.py
```

---

## ⚠️ Known Issues & Limitations

| Issue | Status |
|---|---|
| `temp.txt` is a development scratch file | 🗑️ To be cleaned up |
| Classical and CNN pipelines are not yet unified | 🔜 Planned |

---

## 🗺️ Roadmap

- [ ] Add unified evaluation script with accuracy, precision, and recall metrics
- [ ] Integrate Cython blocky-score extension into `main.py`
- [ ] Add unit tests for preprocessing and block-scoring functions

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

<div align="center">
  Made with 🧠 and 🐍 &nbsp;|&nbsp; DePix AI
</div>
