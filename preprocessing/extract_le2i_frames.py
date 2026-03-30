"""
preprocessing/extract_le2i_frames.py
─────────────────────────────────────────────────────────────────────
Extracts labeled frames from Le2i raw videos using annotation files.

Le2i annotation format:
  Line 1: fall_start_frame  (single integer, e.g. 48)
  Line 2: fall_end_frame    (single integer, e.g. 80)
  Line 3+: per-frame data   (ignored)

Run from CAPSTONE folder:
    python preprocessing/extract_le2i_frames.py
─────────────────────────────────────────────────────────────────────
"""

import os
import cv2
import glob
import re
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LE2I_RAW_DIR, LE2I_FRAMES_DIR

FRAME_SKIP = 3   # save 1 frame every 3 frames (~8fps from 25fps video)


def parse_annotation_file(ann_path):
    """
    Le2i annotation format:
        Line 1: fall_start_frame  (single integer)
        Line 2: fall_end_frame    (single integer)
        Line 3+: per-frame data   (ignored)

    Returns: [(start_frame, end_frame)]
    """
    fall_intervals = []
    try:
        with open(ann_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        if len(lines) >= 2:
            # First number on line 1 = start, first number on line 2 = end
            start = int(re.findall(r"\d+", lines[0])[0])
            end   = int(re.findall(r"\d+", lines[1])[0])
            fall_intervals.append((start, end))
    except Exception as e:
        print(f"  [Warning] Could not parse {ann_path}: {e}")
    return fall_intervals


def is_fall_frame(frame_idx, fall_intervals):
    for start, end in fall_intervals:
        if start <= frame_idx <= end:
            return True
    return False


def extract_frames_from_scene(videos_dir, annotations_dir,
                               fall_out, nofall_out,
                               scene_name, counters):
    video_files = (glob.glob(os.path.join(videos_dir, "*.avi"))
                 + glob.glob(os.path.join(videos_dir, "*.mp4")))
    if not video_files:
        print(f"  [Skip] No videos in {videos_dir}")
        return

    for vpath in video_files:
        vname = os.path.splitext(os.path.basename(vpath))[0]

        # Match annotation by exact filename (handles spaces + parentheses)
        ann_path = os.path.join(annotations_dir, vname + ".txt")
        if not os.path.exists(ann_path):
            # Try other extensions
            for ext in [".csv", ".ann"]:
                candidate = os.path.join(annotations_dir, vname + ext)
                if os.path.exists(candidate):
                    ann_path = candidate
                    break
            else:
                ann_path = None

        fall_intervals = parse_annotation_file(ann_path) if ann_path else []
        if not ann_path:
            print(f"  [NoAnn] {vname} → treating all frames as no_fall")

        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"  [Error] Cannot open {vpath}")
            continue

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % FRAME_SKIP == 0:
                # Sanitize filename (remove spaces and parentheses)
                safe_name = vname.replace(" ", "_").replace("(", "").replace(")", "")
                fname = f"{scene_name}_{safe_name}_f{frame_idx:06d}.jpg"
                if is_fall_frame(frame_idx, fall_intervals):
                    cv2.imwrite(os.path.join(fall_out, fname), frame)
                    counters["fall"] += 1
                else:
                    cv2.imwrite(os.path.join(nofall_out, fname), frame)
                    counters["no_fall"] += 1
            frame_idx += 1
        cap.release()


def main():
    print("=" * 60)
    print("  Le2i Frame Extractor")
    print("=" * 60)

    if not os.path.isdir(LE2I_RAW_DIR):
        print(f"\n❌  Le2i raw directory not found:")
        print(f"    {os.path.abspath(LE2I_RAW_DIR)}")
        return

    # Clear old frames and start fresh
    fall_out   = os.path.join(LE2I_FRAMES_DIR, "fall")
    nofall_out = os.path.join(LE2I_FRAMES_DIR, "no_fall")
    os.makedirs(fall_out,   exist_ok=True)
    os.makedirs(nofall_out, exist_ok=True)

    counters = {"fall": 0, "no_fall": 0}

    # Find all scene folders with Videos + Annotation_files
    scene_folders = []
    for root, dirs, files in os.walk(LE2I_RAW_DIR):
        if "Videos" in dirs and ("Annotation_files" in dirs or
                                   "Annotations_files" in dirs):
            scene_folders.append(root)

    if not scene_folders:
        print(f"\n❌  No scene folders found under {os.path.abspath(LE2I_RAW_DIR)}")
        return

    print(f"\nFound {len(scene_folders)} scene(s):\n")

    for scene_path in tqdm(scene_folders, desc="Scenes"):
        scene_name = os.path.basename(scene_path).replace(" ", "_")
        videos_dir = os.path.join(scene_path, "Videos")
        ann_dir    = os.path.join(scene_path, "Annotation_files")
        if not os.path.isdir(ann_dir):
            ann_dir = os.path.join(scene_path, "Annotations_files")

        extract_frames_from_scene(
            videos_dir, ann_dir,
            fall_out, nofall_out,
            scene_name, counters
        )

    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"  Fall frames    : {counters['fall']}")
    print(f"  No-fall frames : {counters['no_fall']}")
    print(f"  Saved to       : {os.path.abspath(LE2I_FRAMES_DIR)}")
    print(f"{'='*60}")
    print("\nNext → run:  python training/train_vit.py")


if __name__ == "__main__":
    main()