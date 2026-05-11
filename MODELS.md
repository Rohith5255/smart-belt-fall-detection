# 📦 Pre-trained Model Weights

The trained Vision Transformer model (\it_model.pth\) is hosted on Google Drive due to its file size (~305 MB).

## 📥 Download Instructions

### Option 1: Automated (Recommended)

\\\ash
python download_models.py
\\\

This handles installation of \gdown\ automatically and downloads the model to the correct location.

### Option 2: Manual Download
1. Download from Google Drive: [vit_model.pth](https://drive.google.com/file/d/10_57V1f1EgN6tzsiWD5vDB7deeDJyyrl/view?usp=sharing)
2. Place it inside the \saved_models/\ directory in the project root.

### Option 3: Direct gdown command

\\\ash
pip install gdown
python -m gdown https://drive.google.com/uc?id=10_57V1f1EgN6tzsiWD5vDB7deeDJyyrl -O saved_models/vit_model.pth
\\\

> **Note:** Use \python -m gdown\ instead of just \gdown\ to avoid PATH issues on Windows.

## 📂 Expected File Structure After Download

\\\
CAPSTONE/
├── saved_models/
│   └── vit_model.pth   ← Downloaded file goes here (~305 MB)
├── inference/
│   └── realtime_inference.py
└── ...
\\\

## 🧠 Model Details

| Attribute | Value |
|---|---|
| **Architecture** | Vision Transformer (ViT) |
| **Training Dataset** | Le2i Fall Detection Dataset |
| **Task** | Binary classification (Fall / No-Fall) |
| **File Size** | ~305 MB |
| **Used in** | \inference/realtime_inference.py\ |
| **Fusion partner** | SVM model on IMU + pressure sensor data |
