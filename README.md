# Smart Belt Fall Detection System
Final Year Capstone Project

---

## Folder Structure (place everything inside CAPSTONE/)

```
CAPSTONE/
│
├── data/
│   ├── upfall/                              ← UP-Fall dataset CSVs
│   ├── urfd/
│   │   └── UR_fall_detection_dataset_cam0_rgb/
│   │       ├── fall-01-cam0-rgb/
│   │       ├── fall-02-cam0-rgb/  ...
│   │       ├── adl-01-cam0-rgb/
│   │       └── adl-02-cam0-rgb/  ...
│   ├── le2i/                                ← raw Le2i videos (as downloaded)
│   │   ├── Coffee_room_01/Coffee_room_01/
│   │   │   ├── Annotation_files/
│   │   │   └── Videos/
│   │   ├── Coffee_room_02/ ...
│   │   ├── Home_01/ ...
│   │   ├── Home_02/ ...
│   │   ├── Lecture_room/ ...
│   │   └── Office/ ...
│   └── le2i_frames/                         ← AUTO-CREATED by extract_frames
│       ├── fall/
│       └── no_fall/
│
├── preprocessing/
│   ├── extract_le2i_frames.py
│   ├── video_preprocessing.py
│   └── sensor_preprocessing.py
│
├── models/
│   ├── svm_model.py
│   └── vit_model.py
│
├── fusion/
│   └── fusion_model.py
│
├── training/
│   ├── train_svm.py
│   └── train_vit.py
│
├── inference/
│   └── realtime_inference.py
│
├── iot/
│   ├── iot_receiver.py
│   └── esp32_belt_firmware.ino
│
├── alerts/
│   └── alert_system.py
│
├── saved_models/                            ← AUTO-CREATED during training
│   ├── svm_model.pkl
│   ├── vit_model.pth
│   ├── svm_confusion_matrix.png
│   ├── vit_confusion_matrix.png
│   └── vit_training_curves.png
│
├── config.py
├── main.py
└── requirements.txt
```

---

## Setup

```bash
# From inside CAPSTONE/ folder:
pip install -r requirements.txt
```

---

## Run Order

### Step 1 — Extract Le2i frames (run once)
```bash
python main.py --mode extract_frames
```

### Step 2 — Train SVM (needs UP-Fall dataset in data/upfall/)
```bash
python main.py --mode train_svm
```

### Step 3 — Train ViT (needs URFD + Le2i frames)
```bash
python main.py --mode train_vit
```

### Step 4 — Run live system
```bash
# With real ESP32 belt + camera:
python main.py --mode run

# Test without any hardware:
python main.py --mode run --simulate
```

---

## Config

Edit `config.py` to change:
- `SERIAL_PORT` — your ESP32 COM port (default: COM3)
- `CAMERA_INDEX` — webcam number (default: 0)
- `FALL_THRESHOLD` — sensitivity (default: 0.50)
- `VIT_EPOCHS` — training epochs (default: 20)
