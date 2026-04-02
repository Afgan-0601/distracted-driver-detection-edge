"""Dataset and DataLoader utilities for the State Farm Distracted Driver dataset.

Expected raw data layout (mirrors Kaggle download):
    data/raw/
        imgs/
            train/
                c0/  *.jpg
                c1/  *.jpg
                ...
                c9/  *.jpg
            test/    *.jpg   (no labels — Kaggle competition format)
        driver_imgs_list.csv  (optional metadata: subject, class, image)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image

from src.config import (
    BATCH_SIZE,
    CLASS_INDEX,
    CLASS_NAMES,
    NUM_CLASSES,
    RAW_DATA_DIR,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
)
from src.data.transforms import get_train_transforms, get_val_transforms


# ─── Dataset ─────────────────────────────────────────────────────────────────

class DriverDataset(Dataset):
    """
    PyTorch Dataset for inside-cabin driver images.

    Supports two loading modes:
    * ``mode="folder"`` — scans ``root_dir`` for ImageFolder-style subfolders
      named c0-c9.
    * ``mode="csv"``  — reads a ``driver_imgs_list.csv`` that maps
      image filenames to class labels (useful for subject-level splits).
    """

    def __init__(
        self,
        root_dir: Path,
        transform: Optional[Callable] = None,
        mode: str = "folder",  # "folder" | "csv"
        csv_path: Optional[Path] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        if mode == "folder":
            self._load_from_folders()
        elif mode == "csv":
            if csv_path is None:
                raise ValueError("csv_path must be provided when mode='csv'")
            self._load_from_csv(csv_path)
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'folder' or 'csv'.")

    def _load_from_folders(self) -> None:
        for class_key, class_idx in CLASS_INDEX.items():
            class_dir = self.root_dir / class_key
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((img_path, class_idx))

    def _load_from_csv(self, csv_path: Path) -> None:
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                class_key = row["classname"]
                img_name  = row["img"]
                class_idx = CLASS_INDEX.get(class_key)
                if class_idx is None:
                    continue
                img_path = self.root_dir / class_key / img_name
                self.samples.append((img_path, class_idx))

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def class_names(self) -> dict[str, str]:
        return CLASS_NAMES

    @property
    def num_classes(self) -> int:
        return NUM_CLASSES


# ─── DataLoaders factory ──────────────────────────────────────────────────────

def build_dataloaders(
    train_dir: Optional[Path] = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders from the raw data directory.

    The training folder is split deterministically using a fixed seed so
    results are reproducible across runs.

    Returns
    -------
    dict with keys "train", "val", "test".
    """
    if train_dir is None:
        train_dir = RAW_DATA_DIR / "imgs" / "train"

    full_dataset = DriverDataset(
        root_dir=train_dir,
        transform=None,  # transforms applied per split below
    )

    total = len(full_dataset)
    n_train = int(total * TRAIN_SPLIT)
    n_val   = int(total * VAL_SPLIT)
    n_test  = total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    # Assign per-split transforms (augment only train)
    train_ds.dataset = _TransformWrapper(full_dataset, get_train_transforms())
    val_ds.dataset   = _TransformWrapper(full_dataset, get_val_transforms())
    test_ds.dataset  = _TransformWrapper(full_dataset, get_val_transforms())

    # pin_memory only works with CUDA, not MPS or CPU
    import platform
    effective_pin = pin_memory and torch.cuda.is_available()

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=effective_pin,
    )

    return {
        "train": DataLoader(train_ds, shuffle=True,  **loader_kwargs),
        "val":   DataLoader(val_ds,   shuffle=False, **loader_kwargs),
        "test":  DataLoader(test_ds,  shuffle=False, **loader_kwargs),
    }


class _TransformWrapper(Dataset):
    """Thin wrapper that applies a given transform to an underlying dataset."""

    def __init__(self, base_dataset: Dataset, transform: Callable) -> None:
        self._base = base_dataset
        self._transform = transform

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int):
        img_path, label = self._base.samples[idx]
        image = Image.open(img_path).convert("RGB")
        return self._transform(image), label
