#!/usr/bin/env python
"""
train.py — Run the full training pipeline.

Usage
-----
    python scripts/train.py
    python scripts/train.py --epochs 30 --batch-size 64 --device cuda
    python scripts/train.py --resume models/weights/best_model.pt
"""

import argparse
import sys
from pathlib import Path

# Make project root importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import BATCH_SIZE, MODELS_DIR, NUM_EPOCHS, RAW_DATA_DIR
from src.data.dataset import build_dataloaders
from src.models.classifier import build_model
from src.training.evaluate import evaluate_model, print_report
from src.training.trainer import TrainConfig, Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the distracted driver classifier")
    p.add_argument("--epochs",      type=int,   default=NUM_EPOCHS)
    p.add_argument("--warmup",      type=int,   default=5,
                   help="Epochs to train only the classifier head")
    p.add_argument("--batch-size",  type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--device",      type=str,   default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--train-dir",   type=Path,  default=None,
                   help="Override default train image directory")
    p.add_argument("--resume",      type=Path,  default=None,
                   help="Resume from checkpoint path")
    p.add_argument("--workers",     type=int,   default=0,
                   help="DataLoader workers (0=main thread, safest on macOS)")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Train from scratch without ImageNet weights")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("[train] Building data loaders …")
    dataloaders = build_dataloaders(
        train_dir=args.train_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    print(
        f"  train={len(dataloaders['train'].dataset)}  "
        f"val={len(dataloaders['val'].dataset)}  "
        f"test={len(dataloaders['test'].dataset)}"
    )

    print("[train] Building model …")
    model = build_model(
        pretrained=not args.no_pretrained,
        freeze_backbone=True,
    )

    train_cfg = TrainConfig(
        num_epochs=args.epochs,
        warmup_epochs=args.warmup,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_dir=MODELS_DIR,
    )

    trainer = Trainer(model, train_cfg)
    print(f"[train] Starting training on device={trainer.device} …\n")
    trainer.fit(dataloaders, resume_from=args.resume)

    print("\n[train] Evaluating best model on test set …")
    from src.models.classifier import DriverClassifier
    best_model = DriverClassifier.load(MODELS_DIR / "best_model.pt")
    metrics = evaluate_model(best_model, dataloaders["test"], device=str(trainer.device))
    print_report(metrics)

    print("[train] Done. Weights saved to:", MODELS_DIR)


if __name__ == "__main__":
    main()
