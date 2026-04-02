"""
Trainer — full training loop with:
* Two-phase strategy: warm-up (frozen backbone) → full fine-tuning
* Learning rate scheduling (ReduceLROnPlateau)
* Best-model checkpointing
* Per-epoch console logging (no external logging lib required for MVP)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.config import LEARNING_RATE, MODELS_DIR, NUM_EPOCHS, WEIGHT_DECAY
from src.models.classifier import DriverClassifier


@dataclass
class TrainConfig:
    num_epochs: int = NUM_EPOCHS
    warmup_epochs: int = 5          # epochs with frozen backbone
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    device: str = "auto"            # "auto" | "cpu" | "cuda" | "mps"
    checkpoint_dir: Path = field(default_factory=lambda: MODELS_DIR)
    early_stop_patience: int = 7


class Trainer:
    """
    Orchestrates training with a two-phase warm-up + fine-tuning strategy.

    Phase 1 (warm-up): Backbone frozen, only the small classifier head trains.
    Phase 2 (fine-tune): All layers unlocked at a lower learning rate.
    """

    def __init__(self, model: DriverClassifier, config: TrainConfig) -> None:
        self.model = model
        self.cfg   = config
        self.device = self._resolve_device(config.device)
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

        self.history: list[dict] = []
        self._best_val_loss = float("inf")
        self._epochs_no_improve = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        dataloaders: dict[str, DataLoader],
        resume_from: Optional[Path] = None,
    ) -> list[dict]:
        """
        Run the full training loop.

        Parameters
        ----------
        dataloaders : dict with "train" and "val" keys.
        resume_from : optional checkpoint path to resume training.

        Returns
        -------
        Training history as a list of per-epoch dicts.
        """
        if resume_from:
            self._load_checkpoint(resume_from)

        for epoch in range(1, self.cfg.num_epochs + 1):
            # Phase transition: unfreeze backbone after warm-up
            if epoch == self.cfg.warmup_epochs + 1:
                print(f"\n[Trainer] Phase 2 — unfreezing backbone at epoch {epoch}")
                self.model.unfreeze_backbone()
                # Reset optimiser with a lower lr for the backbone
                self.optimizer = AdamW(
                    self.model.parameters(),
                    lr=self.cfg.learning_rate * 0.1,
                    weight_decay=self.cfg.weight_decay,
                )
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer, mode="min", factor=0.5, patience=3
                )

            t0 = time.time()
            train_metrics = self._run_epoch(dataloaders["train"], training=True)
            val_metrics   = self._run_epoch(dataloaders["val"],   training=False)
            elapsed = time.time() - t0

            self.scheduler.step(val_metrics["loss"])

            record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc":  train_metrics["acc"],
                "val_loss":   val_metrics["loss"],
                "val_acc":    val_metrics["acc"],
                "lr":         self.optimizer.param_groups[0]["lr"],
                "elapsed_s":  round(elapsed, 1),
            }
            self.history.append(record)
            self._log_epoch(record)

            # Checkpoint best model
            if val_metrics["loss"] < self._best_val_loss:
                self._best_val_loss = val_metrics["loss"]
                self._epochs_no_improve = 0
                self._save_checkpoint(epoch)
            else:
                self._epochs_no_improve += 1

            # Early stopping
            if self._epochs_no_improve >= self.cfg.early_stop_patience:
                print(f"\n[Trainer] Early stopping triggered at epoch {epoch}.")
                break

        self._save_history()
        return self.history

    # ── Private helpers ───────────────────────────────────────────────────────

    def _run_epoch(
        self, loader: DataLoader, training: bool
    ) -> dict[str, float]:
        self.model.train(training)
        total_loss = 0.0
        correct = 0
        total = 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                loss   = self.criterion(logits, labels)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += images.size(0)

        return {
            "loss": total_loss / total,
            "acc":  correct / total,
        }

    def _save_checkpoint(self, epoch: int) -> None:
        path = self.cfg.checkpoint_dir / f"best_model.pt"
        self.cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "val_loss": self._best_val_loss,
            },
            path,
        )

    def _load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        print(f"[Trainer] Resumed from checkpoint at epoch {ckpt['epoch']}")

    def _save_history(self) -> None:
        history_path = self.cfg.checkpoint_dir / "training_history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w") as fh:
            json.dump(self.history, fh, indent=2)

    @staticmethod
    def _log_epoch(record: dict) -> None:
        print(
            f"Epoch {record['epoch']:3d} | "
            f"train_loss={record['train_loss']:.4f}  train_acc={record['train_acc']:.4f} | "
            f"val_loss={record['val_loss']:.4f}  val_acc={record['val_acc']:.4f} | "
            f"lr={record['lr']:.2e}  ({record['elapsed_s']}s)"
        )

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
