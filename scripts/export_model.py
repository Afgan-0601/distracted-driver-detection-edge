#!/usr/bin/env python
"""
export_model.py — Export trained PyTorch weights to ONNX.

Usage
-----
    python scripts/export_model.py
    python scripts/export_model.py --weights models/weights/best_model.pt \
                                   --output  models/weights/driver_classifier.onnx \
                                   --opset   17
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import MODELS_DIR, ONNX_MODEL_NAME, PT_MODEL_NAME
from src.models.classifier import DriverClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    p.add_argument("--weights", type=Path, default=MODELS_DIR / "best_model.pt")
    p.add_argument("--output",  type=Path, default=MODELS_DIR / ONNX_MODEL_NAME)
    p.add_argument("--opset",   type=int,  default=17)
    p.add_argument("--verify",  action="store_true",
                   help="Run a quick sanity-check inference after export")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[export] Loading weights from {args.weights} …")
    model = DriverClassifier.load(args.weights)
    model.eval()

    print(f"[export] Exporting to ONNX (opset={args.opset}) → {args.output}")
    out_path = model.export_onnx(path=args.output, opset=args.opset)
    print(f"[export] Saved: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    if args.verify:
        _verify_onnx(out_path)


def _verify_onnx(onnx_path: Path) -> None:
    """Run a dummy inference through ONNX Runtime to verify correctness."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("[export] onnxruntime not installed — skipping verification.")
        return

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dummy = np.random.rand(1, 3, 224, 224).astype(np.float32)
    outputs = session.run(["logits"], {"image": dummy})
    logits = outputs[0][0]
    pred_class = int(logits.argmax())
    print(f"[export] Verification passed. Dummy prediction → class {pred_class}")


if __name__ == "__main__":
    main()
