"""
main.py — Master entry point for the Fall Detection System.

Run from CAPSTONE folder:
    python main.py --mode extract_frames
    python main.py --mode train_svm
    python main.py --mode train_vit
    python main.py --mode run
    python main.py --mode run --simulate
"""

import argparse
import sys
import os

# Force UTF-8 output so Unicode arrows/emoji print correctly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser(
        description="Smart Belt Fall Detection System",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
RECOMMENDED ORDER:
  1. python main.py --mode extract_frames   ← extract Le2i video frames
  2. python main.py --mode train_svm        ← train sensor model
  3. python main.py --mode train_vit        ← train visual model
  4. python main.py --mode run              ← launch live system
  5. python main.py --mode run --simulate   ← test without hardware
"""
    )
    parser.add_argument("--mode", required=True,
                        choices=["extract_frames","train_svm","train_vit","run"])
    parser.add_argument("--simulate", action="store_true",
                        help="(run mode) use simulated sensor/camera data")
    args = parser.parse_args()

    if args.mode == "extract_frames":
        from preprocessing.extract_le2i_frames import main as run
        run()

    elif args.mode == "train_svm":
        from training.train_svm import main as run
        run()

    elif args.mode == "train_vit":
        from training.train_vit import main as run
        run()

    elif args.mode == "run":
        from inference.realtime_inference import run_realtime
        run_realtime(simulate=args.simulate)


if __name__ == "__main__":
    main()
