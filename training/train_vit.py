"""
training/train_vit.py — Fine-tunes ViT-Base/16 on URFD + Le2i.

Run from CAPSTONE folder:
    python preprocessing/extract_le2i_frames.py   ← run this FIRST
    python training/train_vit.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from preprocessing.video_preprocessing import get_combined_dataloaders
from models.vit_model import FallViT, save_vit, load_vit
from config import VIT_EPOCHS, VIT_LR, VIT_BATCH_SIZE, URFD_DATA_DIR, LE2I_FRAMES_DIR


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in tqdm(loader, desc="  Train", leave=False):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    all_labels, all_proba = [], []
    for imgs, labels in tqdm(loader, desc="  Eval ", leave=False):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(imgs)
        proba  = torch.softmax(logits, dim=1)[:, 1]
        total_loss += criterion(logits, labels).item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_proba.extend(proba.cpu().numpy())
    all_labels = np.array(all_labels)
    all_proba  = np.array(all_proba)
    auc = roc_auc_score(all_labels, all_proba) if len(set(all_labels)) > 1 else 0.0
    return total_loss / total, correct / total, auc, all_labels, all_proba


def main():
    print("=" * 60)
    print("  ViT Fine-tuning  |  URFD + Le2i Combined")
    print("=" * 60)

    # ── Check Le2i frames ────────────────────────────────────────────
    if not os.path.isdir(LE2I_FRAMES_DIR):
        print(f"\n⚠️  Le2i frames not found at: {os.path.abspath(LE2I_FRAMES_DIR)}")
        print("   Run this first:")
        print("     python preprocessing/extract_le2i_frames.py\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[ViT] Device : {device}")
    if device.type == "cuda":
        print(f"      GPU    : {torch.cuda.get_device_name(0)}")
    else:
        print("      ⚠️  No GPU — use Google Colab for faster training.")

    # ── Data ─────────────────────────────────────────────────────────
    urfd = URFD_DATA_DIR if os.path.isdir(URFD_DATA_DIR) else None
    le2i = LE2I_FRAMES_DIR if os.path.isdir(LE2I_FRAMES_DIR) else None

    if not urfd:
        print(f"[ViT] URFD not found: {os.path.abspath(URFD_DATA_DIR)}")
    if not le2i:
        print(f"[ViT] Le2i frames not found: {os.path.abspath(LE2I_FRAMES_DIR)}")
    if not urfd and not le2i:
        print("\n❌  No datasets found. Check config.py paths."); return

    train_loader, val_loader, test_loader = get_combined_dataloaders(
        urfd_dir=urfd, le2i_dir=le2i, batch_size=VIT_BATCH_SIZE, num_workers=2)

    # ── Model ────────────────────────────────────────────────────────
    model = FallViT(pretrained=True).to(device)
    for p in model.backbone.parameters():
        p.requires_grad = False
    print("\n[ViT] Backbone frozen for warm-up (epochs 1–5).")

    criterion     = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer     = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=VIT_LR, weight_decay=1e-4)
    scheduler     = CosineAnnealingLR(optimizer, T_max=VIT_EPOCHS)
    scaler        = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    history       = {k: [] for k in ["train_loss","val_loss","train_acc","val_acc","val_auc"]}
    best_val_auc  = 0.0
    unfreeze_done = False

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(1, VIT_EPOCHS + 1):
        if epoch == 6 and not unfreeze_done:
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer     = AdamW(model.parameters(), lr=VIT_LR * 0.1, weight_decay=1e-4)
            scheduler     = CosineAnnealingLR(optimizer, T_max=VIT_EPOCHS - 5)
            print("\n[ViT] Backbone unfrozen — full fine-tuning.")
            unfreeze_done = True

        tr_loss, tr_acc               = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        vl_loss, vl_acc, vl_auc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        for k, v in zip(["train_loss","val_loss","train_acc","val_acc","val_auc"],
                         [tr_loss, vl_loss, tr_acc, vl_acc, vl_auc]):
            history[k].append(v)

        saved = ""
        if vl_auc > best_val_auc:
            best_val_auc = vl_auc
            save_vit(model)
            saved = "  ✅ saved"

        print(f"Epoch {epoch:3d}/{VIT_EPOCHS}  "
              f"loss={tr_loss:.4f}/{vl_loss:.4f}  "
              f"acc={tr_acc:.4f}/{vl_acc:.4f}  "
              f"AUC={vl_auc:.4f}{saved}")

    # ── Test evaluation ───────────────────────────────────────────────
    print("\n── Test Evaluation ──")
    best = load_vit(device=device)
    _, test_acc, test_auc, y_true, y_proba = evaluate(best, test_loader, criterion, device)
    y_pred = (y_proba >= 0.5).astype(int)
    print(f"\nTest Accuracy : {test_acc:.4f}")
    print(f"Test AUC-ROC  : {test_auc:.4f}")
    print("\n" + classification_report(y_true, y_pred, target_names=["No-Fall","Fall"]))

    # ── Plots ─────────────────────────────────────────────────────────
    os.makedirs("saved_models", exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"], label="Train"); axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")
    axes[1].plot(history["train_acc"], label="Train"); axes[1].plot(history["val_acc"], label="Val")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].set_xlabel("Epoch")
    axes[2].plot(history["val_auc"], color="green")
    axes[2].set_title("Val AUC-ROC"); axes[2].set_xlabel("Epoch")
    plt.suptitle(f"ViT Training — URFD + Le2i  (Test AUC={test_auc:.3f})", fontsize=12)
    plt.tight_layout()
    fig.savefig("saved_models/vit_training_curves.png", dpi=150)

    fig2, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues",
                ax=ax, xticklabels=["No-Fall","Fall"], yticklabels=["No-Fall","Fall"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"ViT Confusion Matrix  (AUC={test_auc:.3f})")
    plt.tight_layout()
    fig2.savefig("saved_models/vit_confusion_matrix.png", dpi=150)

    print("\n[ViT] Plots → saved_models/")
    print("[ViT] Training complete! ✅")


if __name__ == "__main__":
    main()
