# 📦 Pre-trained Model Weights

The trained Vision Transformer model (\it_model.pth\) is hosted on Google Drive due to its file size.

## 📥 Download Instructions

### Option 1: Manual Download
1. Download from Google Drive: [vit_model.pth](https://drive.google.com/file/d/10_57V1f1EgN6tzsiWD5vDB7deeDJyyrl/view?usp=sharing)
2. Place it inside the \saved_models/\ directory in the project root.

### Option 2: Automated Download with gdown (Recommended)

\\\ash
pip install gdown
gdown https://drive.google.com/uc?id=10_57V1f1EgN6tzsiWD5vDB7deeDJyyrl -O saved_models/vit_model.pth
\\\

### Option 3: One-Liner via Helper Script

\\\ash
python download_models.py
\\\

## 📂 Expected File Structure After Download

\\\
CAPSTONE/
├── saved_models/
│   └── vit_model.pth   ← Downloaded file goes here (~XXX MB)
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
| **Used in** | \inference/realtime_inference.py\ |
| **Fusion partner** | SVM model on IMU + pressure sensor data |
