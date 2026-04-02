#!/usr/bin/env python
"""
Kaggle-native training script.

Run this directly inside a Kaggle Notebook kernel.
The State Farm dataset is already mounted at:
    /kaggle/input/state-farm-distracted-driver-detection/

Steps:
1. Open https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/
2. Go to Code → New Notebook
3. Copy-paste this file OR upload it as a notebook
4. Runtime: GPU (P100 or T4)
5. Run all cells — weights saved to /kaggle/working/
"""

import os
import sys
import json

# ── Add project root to path ──────────────────────────────────────────────────
# On Kaggle: upload this repo as a dataset or use !git clone
KAGGLE_INPUT = "/kaggle/input/state-farm-distracted-driver-detection"
PROJECT_ROOT = "/kaggle/working/project"

print("Cloning project repo...")
os.system(f"git clone https://github.com/YOUR_USERNAME/distracted-driver-detection-edge.git {PROJECT_ROOT}")
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

print("Installing dependencies...")
os.system("pip install -q -r requirements.txt")

# ── Symlink Kaggle dataset into expected location ─────────────────────────────
os.makedirs("data/raw", exist_ok=True)
if not os.path.exists("data/raw/imgs"):
    os.symlink(f"{KAGGLE_INPUT}/imgs", "data/raw/imgs")

# Verify
print("\nDataset classes:")
train_dir = "data/raw/imgs/train"
for cls in sorted(os.listdir(train_dir)):
    n = len(os.listdir(os.path.join(train_dir, cls)))
    print(f"  {cls}: {n} images")

# ── Train ─────────────────────────────────────────────────────────────────────
print("\nStarting training...")
os.system(
    "python scripts/train.py "
    "--epochs 20 "
    "--warmup 5 "
    "--batch-size 64 "
    "--device cuda "
    "--workers 4"
)

# ── Export ────────────────────────────────────────────────────────────────────
print("\nExporting to ONNX...")
os.system(
    "python scripts/export_model.py "
    "--weights models/weights/best_model.pt "
    "--output  /kaggle/working/driver_classifier.onnx "
    "--verify"
)

import shutil
shutil.copy("models/weights/best_model.pt",         "/kaggle/working/best_model.pt")
shutil.copy("models/weights/training_history.json", "/kaggle/working/training_history.json")

print("\nDone! Download from /kaggle/working/:")
for f in os.listdir("/kaggle/working/"):
    size = os.path.getsize(f"/kaggle/working/{f}") / 1e6
    print(f"  {f}  ({size:.1f} MB)")
