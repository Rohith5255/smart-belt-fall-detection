"""
preprocessing/video_preprocessing.py
Loads extracted frames from URFD and Le2i for ViT training.
"""

import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import random

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VIT_IMG_SIZE, VIT_BATCH_SIZE

_FALL_KEYWORDS   = ["fall", "falls", "chute"]
_NOFALL_KEYWORDS = ["adl", "notfall", "not_fall", "no fall",
                     "no_fall", "nofall", "activities", "normal"]

def _resolve_label(folder_name):
    name = folder_name.lower().strip()
    for kw in _FALL_KEYWORDS:
        if name.startswith(kw):
            return 1
    for kw in _NOFALL_KEYWORDS:
        if kw in name:
            return 0
    return None


def _collect_samples(root_dir, dataset_name=""):
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(
            f"[{dataset_name}] Not found: {os.path.abspath(root_dir)}")

    subdirs = sorted([d for d in os.listdir(root_dir)
                       if os.path.isdir(os.path.join(root_dir, d))])
    samples = []
    fall_count = no_fall_count = skipped = 0

    for subdir in subdirs:
        label = _resolve_label(subdir)
        if label is None:
            skipped += 1
            continue
        subdir_path = os.path.join(root_dir, subdir)
        imgs = (glob.glob(os.path.join(subdir_path, "*.jpg"))
              + glob.glob(os.path.join(subdir_path, "*.jpeg"))
              + glob.glob(os.path.join(subdir_path, "*.png")))
        if not imgs:
            imgs = (glob.glob(os.path.join(subdir_path, "**", "*.jpg"), recursive=True)
                  + glob.glob(os.path.join(subdir_path, "**", "*.png"), recursive=True))
        samples.extend([(p, label) for p in imgs])
        if label == 1: fall_count    += len(imgs)
        else:          no_fall_count += len(imgs)

    print(f"  [{dataset_name}] Subfolders: {len(subdirs)} (skipped {skipped})")
    print(f"  [{dataset_name}] Fall      : {fall_count}")
    print(f"  [{dataset_name}] No-fall   : {no_fall_count}")
    print(f"  [{dataset_name}] Total     : {len(samples)}")

    if len(samples) == 0:
        raise ValueError(f"[{dataset_name}] No images found in {root_dir}.")
    return samples


def _build_transform(split):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((VIT_IMG_SIZE + 20, VIT_IMG_SIZE + 20)),
            transforms.RandomCrop(VIT_IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((VIT_IMG_SIZE, VIT_IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


class FallFrameDataset(Dataset):
    def __init__(self, samples, split="train"):
        self.samples   = samples
        self.transform = _build_transform(split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (VIT_IMG_SIZE, VIT_IMG_SIZE), color=0)
        return self.transform(image), label


def _split_samples(samples, val_ratio=0.15, test_ratio=0.10, seed=42):
    random.seed(seed)
    s = samples.copy()
    random.shuffle(s)
    n       = len(s)
    n_test  = max(1, int(n * test_ratio))
    n_val   = max(1, int(n * val_ratio))
    n_train = n - n_val - n_test
    return s[:n_train], s[n_train:n_train + n_val], s[n_train + n_val:]


def get_combined_dataloaders(urfd_dir=None, le2i_dir=None,
                              batch_size=VIT_BATCH_SIZE, num_workers=2):
    if urfd_dir is None and le2i_dir is None:
        raise ValueError("Provide at least one of: urfd_dir, le2i_dir")

    print("\n[VideoPrep] Scanning datasets …")
    all_samples = []

    if urfd_dir and os.path.isdir(urfd_dir):
        all_samples.extend(_collect_samples(urfd_dir, "URFD"))
    elif urfd_dir:
        print(f"  [URFD] ⚠️  Not found at {urfd_dir} — skipping.")

    if le2i_dir and os.path.isdir(le2i_dir):
        all_samples.extend(_collect_samples(le2i_dir, "Le2i"))
    elif le2i_dir:
        print(f"  [Le2i] ⚠️  Not found at {le2i_dir}")
        print(f"         Run: python preprocessing/extract_le2i_frames.py first.")

    if not all_samples:
        raise ValueError("No samples loaded. Check dataset paths in config.py.")

    print(f"\n[VideoPrep] Combined : {len(all_samples)} frames")
    print(f"            Falls    : {sum(1 for _,l in all_samples if l==1)}")
    print(f"            No-falls : {sum(1 for _,l in all_samples if l==0)}\n")

    train_s, val_s, test_s = _split_samples(all_samples)
    print(f"[VideoPrep] Train:{len(train_s)}  Val:{len(val_s)}  Test:{len(test_s)}")

    kw = dict(batch_size=batch_size, num_workers=num_workers)
    return (DataLoader(FallFrameDataset(train_s, "train"), shuffle=True, pin_memory=True, **kw),
            DataLoader(FallFrameDataset(val_s,   "val"),   **kw),
            DataLoader(FallFrameDataset(test_s,  "test"),  **kw))


# ── Inference helper ───────────────────────────────────────────────────
_INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((VIT_IMG_SIZE, VIT_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def preprocess_frame_for_inference(frame_bgr):
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = _INFER_TRANSFORM(Image.fromarray(rgb))
    return tensor.unsqueeze(0)
