"""
models/vit_model.py — ViT-Base/16 for visual fall detection pipeline.
"""

import os
import torch
import torch.nn as nn
import timm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VIT_MODEL_NAME, VIT_NUM_CLASSES, VIT_PRETRAINED, VIT_MODEL_PATH


class FallViT(nn.Module):
    def __init__(self, num_classes=VIT_NUM_CLASSES, pretrained=VIT_PRETRAINED):
        super().__init__()
        self.backbone = timm.create_model(
            VIT_MODEL_NAME, pretrained=pretrained, num_classes=0)
        embed_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))

    def get_embedding(self, x):
        return self.backbone(x)


def save_vit(model, path=VIT_MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[ViT] Saved → {path}")


def load_vit(path=VIT_MODEL_PATH, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ViT model not found at {path}.\n"
            "Run: python training/train_vit.py")
    model = FallViT(pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    print(f"[ViT] Loaded ← {path}  (device={device})")
    return model


@torch.no_grad()
def vit_predict_proba(model, tensor, device=None):
    if device is None:
        device = next(model.parameters()).device
    logits = model(tensor.to(device))
    return torch.softmax(logits, dim=1)[0, 1].item()
