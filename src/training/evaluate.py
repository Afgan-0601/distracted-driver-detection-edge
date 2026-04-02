"""
Evaluation utilities — per-class metrics, confusion matrix, and report.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.config import CLASS_NAMES, INDEX_TO_CLASS
from src.models.classifier import DriverClassifier


def evaluate_model(
    model: DriverClassifier,
    loader: DataLoader,
    device: str = "cpu",
) -> dict:
    """
    Run inference on *loader* and return a comprehensive metrics dict.

    Returns
    -------
    dict with keys: accuracy, per_class_accuracy, confusion_matrix (flat),
    total_samples, correct_samples.
    """
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(dev, non_blocking=True)
            logits = model(images)
            preds  = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    return compute_metrics(all_labels, all_preds)


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
) -> dict:
    """
    Pure-Python metrics computation (no sklearn dependency).

    Returns accuracy, per-class accuracy, and a flat confusion matrix
    compatible with JSON serialisation.
    """
    n = len(CLASS_NAMES)
    # confusion matrix: cm[true][pred]
    cm: list[list[int]] = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    correct = sum(cm[i][i] for i in range(n))
    total   = sum(sum(row) for row in cm)

    per_class: dict[str, dict] = {}
    for idx in range(n):
        class_key = INDEX_TO_CLASS[idx]
        tp = cm[idx][idx]
        fn = sum(cm[idx]) - tp
        fp = sum(cm[r][idx] for r in range(n)) - tp
        tn = total - tp - fn - fp

        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        per_class[class_key] = {
            "label":     CLASS_NAMES[class_key],
            "support":   support,
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
        }

    return {
        "accuracy":          round(correct / total, 4) if total else 0.0,
        "correct_samples":   correct,
        "total_samples":     total,
        "per_class":         per_class,
        "confusion_matrix":  cm,
    }


def print_report(metrics: dict) -> None:
    """Pretty-print evaluation results to stdout."""
    print(f"\n{'='*60}")
    print(f"  Overall Accuracy: {metrics['accuracy']*100:.2f}%  "
          f"({metrics['correct_samples']}/{metrics['total_samples']})")
    print(f"{'='*60}")
    print(f"  {'Class':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*62}")
    for key, m in metrics["per_class"].items():
        print(
            f"  {key} {m['label']:<18} "
            f"{m['precision']:>10.4f} {m['recall']:>10.4f} "
            f"{m['f1']:>10.4f} {m['support']:>10}"
        )
    print(f"{'='*60}\n")
