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


_LOCAL_FALLBACK = r"C:\Temp\vit_model_local.pth"


def _is_cloud_only(path):
    """Returns True if the file is an OneDrive cloud-only placeholder."""
    try:
        import ctypes
        FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x400000
        attrs = ctypes.windll.kernel32.GetFileAttributesW(path)
        return bool(attrs & FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS)
    except Exception:
        return False


def load_vit(path=VIT_MODEL_PATH, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prefer a fully-local copy if the OneDrive version is still cloud-only
    load_path = path
    if _is_cloud_only(path):
        if os.path.exists(_LOCAL_FALLBACK) and not _is_cloud_only(_LOCAL_FALLBACK):
            print(f"[ViT] OneDrive file still syncing — using local copy: {_LOCAL_FALLBACK}")
            load_path = _LOCAL_FALLBACK
        else:
            raise OSError(
                f"vit_model.pth is still a OneDrive cloud placeholder and no local "
                f"copy exists at {_LOCAL_FALLBACK}.\n"
                "Right-click the file in Explorer → 'Always keep on this device', "
                "or wait for OneDrive to finish syncing.")

    if not os.path.exists(load_path):
        raise FileNotFoundError(
            f"ViT model not found at {load_path}.\n"
            "Run: python training/train_vit.py")

    # Quick read-test: detect OneDrive cloud files that pass the attribute
    # check but hang indefinitely when actually read (mid-download timeout).
    try:
        import signal as _signal

        def _timeout_handler(signum, frame):
            raise OSError("Read timed out — file is still downloading from OneDrive.")

        # Windows doesn't support SIGALRM, so use a threading-based timeout.
        import threading as _threading
        _read_ok = []
        _read_err = []

        def _probe():
            try:
                with open(load_path, "rb") as _f:
                    magic = _f.read(4)
                if magic[:2] != b"PK":
                    _read_err.append(f"Unexpected magic bytes {magic.hex()} — file may be corrupt.")
                else:
                    _read_ok.append(True)
            except OSError as exc:
                _read_err.append(str(exc))

        _t = _threading.Thread(target=_probe, daemon=True)
        _t.start()
        _t.join(timeout=5.0)          # give OneDrive 5 s to return the first 4 bytes
        if not _read_ok:
            err = _read_err[0] if _read_err else "Read timed out — OneDrive has not synced this file yet."
            raise OSError(err)
    except OSError:
        raise

    model = FallViT(pretrained=False)
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.to(device).eval()
    print(f"[ViT] Loaded from {load_path}  (device={device})")
    return model


@torch.no_grad()
def vit_predict_proba(model, tensor, device=None):
    if device is None:
        device = next(model.parameters()).device
    logits = model(tensor.to(device))
    return torch.softmax(logits, dim=1)[0, 1].item()
