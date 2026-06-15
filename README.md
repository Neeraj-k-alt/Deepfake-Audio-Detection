# Deepfake-Audio-Detection
# 🎙️ Deepfake Audio Detection

A machine learning-based system for detecting AI-generated and manipulated speech using handcrafted acoustic features and an ensemble of classical machine learning models.

This project was developed as part of **Problem Statement 02 — Machine Learning / Deep Learning / Audio Processing**.

---

## 📌 Project Overview

Recent advances in generative AI have enabled the creation of highly realistic synthetic voices, creating significant challenges for digital trust, media authenticity, and cybersecurity.

This project aims to automatically distinguish between **genuine human speech** and **deepfake audio** using a robust feature extraction pipeline and an ensemble classification approach.

The complete workflow includes:

* Dataset acquisition using KaggleHub
* Audio preprocessing
* Feature extraction
* Model training and evaluation
* Deepfake prediction on unseen audio samples

---

## 📂 Dataset

**Dataset:** *Fake or Real Audio Dataset*

* **Source:** Kaggle
* **Link:** https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset
* **Classes:**

  * Genuine (Human Speech)
  * Deepfake (Synthetic Speech)

The dataset contains audio recordings in multiple formats, including:

* `.wav`
* `.flac`
* `.mp3`
* `.ogg`

---

## 🏗️ Project Pipeline

```text
Dataset Download
        ↓
Dataset Scanning
        ↓
Audio Preprocessing
        ↓
Feature Extraction
        ↓
Train/Validation Split
        ↓
Feature Scaling
        ↓
Model Training
        ↓
Performance Evaluation
        ↓
Model Export
        ↓
Inference on New Audio
```

---

## ⚙️ Audio Preprocessing

Each audio file undergoes the following preprocessing steps:

* Audio loading using Librosa
* Resampling to a fixed sampling rate
* Mono channel conversion
* Signal normalization
* Handling variable-length recordings
* Invalid/corrupted file filtering

---

## 🔍 Feature Extraction

A **274-dimensional feature vector** is extracted from each audio sample.

Extracted features include:

### Spectral Features

* Spectral centroid
* Spectral bandwidth
* Spectral roll-off
* Spectral contrast

### Temporal Features

* Zero-crossing rate (ZCR)
* Root mean square (RMS) energy

### Frequency Features

* Fundamental frequency (F0)

### Cepstral Features

* MFCC coefficients
* Delta MFCCs
* Delta-Delta MFCCs

### Harmonic Features

* Chroma features

Feature extraction is parallelized using multi-threading for faster processing.

---

## 🤖 Model Architecture

The project evaluates multiple machine learning models:

1. Random Forest
2. XGBoost
3. Support Vector Machine (RBF Kernel)

A soft voting ensemble combines predictions from all models.

### Ensemble Configuration

```text
VotingClassifier(
    estimators = [Random Forest, XGBoost, SVM],
    voting = "soft",
    weights = [2, 2, 1]
)
```

### Best Performing Model

🏆 **Random Forest Classifier**

Hyperparameters:

* Number of trees: 300
* Minimum samples per leaf: 2
* Feature selection: sqrt
* Class weighting: balanced

---

## 📊 Performance Metrics

The following metrics are used for evaluation:

* Accuracy
* Precision
* Recall
* F1 Score
* Equal Error Rate (EER)
* ROC-AUC
* Confusion Matrix
* Cross-validation accuracy

---

## 📈 Confusion Matrix

| Actual \ Predicted | Genuine | Deepfake |
| ------------------ | ------- | -------- |
| Genuine            | 109     | 0        |
| Deepfake           | 0       | 109      |

---

## 🖥️ Hardware Configuration

* GPU: NVIDIA Tesla T4
* Framework: Scikit-learn
* Optional GPU Acceleration: XGBoost CUDA

---

## 📁 Project Structure

```text
deepfake-audio-detection/

├── notebooks/
│   └── deepfake_audio_detection.ipynb
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── X_features.npy
│
├── src/
│   ├── preprocess.py
│   ├── feature_extraction.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── performance_report.pdf
│
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone <repository-url>
cd deepfake-audio-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---


## 🧪 Training

Training is performed directly within the notebook.

The workflow automatically:

* Downloads the dataset
* Extracts features
* Trains models
* Evaluates performance
* Saves the best model

---

## 🎯 Inference on New Audio

Run the inference script:

```bash
python src/inference.py --audio path/to/audio.wav
```

Example output:

```text
Prediction: Deepfake
Confidence: 98.7%
```

---

## 📚 Libraries Used

* Python
* NumPy
* Pandas
* Librosa
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Joblib
* PyTorch
* KaggleHub
* TQDM

---

## 🔮 Future Improvements

* End-to-end deep learning models (CNN, CRNN, Transformer)
* Real-time inference API
* Streamlit web application
* Explainable AI techniques
* Robustness evaluation against unseen attacks
* Cross-dataset generalization

---

## 👨‍💻 Author

**Neeraj**

BS-MS (Chemistry) | Machine Learning & Data Science Enthusiast

---

## 📄 License

This project is intended for academic and research purposes only.
