"""
End-to-end detection pipeline.

Supports two inference backends:
* ``"pytorch"``  — torch model in eval mode (dev/training server)
* ``"onnx"``     — ONNX Runtime session (edge deployment target)

The pipeline combines the classifier with the RiskCalculator so every
call to :meth:`predict` returns both a classification result AND a
fully computed risk assessment.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.config import (
    CONFIDENCE_THRESHOLD,
    CLASS_NAMES,
    INDEX_TO_CLASS,
    MODELS_DIR,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    ONNX_MODEL_NAME,
    PT_MODEL_NAME,
    IMAGE_SIZE,
)
from src.data.transforms import get_inference_transforms
from src.models.classifier import DriverClassifier
from src.utils.risk_calculator import RiskCalculator, RiskResult


class DetectionPipeline:
    """
    Single-entry-point pipeline for distracted driver detection.

    Parameters
    ----------
    backend : "pytorch" | "onnx"
        Inference backend to use.
    model_path : Path, optional
        Explicit path to model weights / ONNX file.
    confidence_threshold : float
        Predictions with lower confidence are marked as ``low_confidence``.
    smoothing_alpha : float
        Passed to :class:`RiskCalculator` for temporal smoothing.
    alert_level : str
        Passed to :class:`RiskCalculator` to set the alert threshold.
    """

    def __init__(
        self,
        backend: Literal["pytorch", "onnx"] = "onnx",
        model_path: Optional[Path] = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        smoothing_alpha: float = 0.4,
        alert_level: str = "HIGH",
    ) -> None:
        self.backend = backend
        self.confidence_threshold = confidence_threshold
        self.transform = get_inference_transforms()
        self.risk_calc = RiskCalculator(
            smoothing_alpha=smoothing_alpha,
            alert_level=alert_level,
        )

        if backend == "pytorch":
            path = model_path or MODELS_DIR / PT_MODEL_NAME
            self._pt_model = DriverClassifier.load(path)
            self._pt_model.eval()
            self._ort_session = None
        elif backend == "onnx":
            self._pt_model = None
            self._ort_session = self._load_onnx(
                model_path or MODELS_DIR / ONNX_MODEL_NAME
            )
        else:
            raise ValueError(f"Unknown backend: {backend!r}")

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        image_input: "bytes | np.ndarray | Image.Image | Path",
        *,
        override_sustained_seconds: Optional[float] = None,
    ) -> dict:
        """
        Run full inference + risk assessment on a single driver image.

        Parameters
        ----------
        image_input : bytes, numpy array, PIL Image, or file path.
        override_sustained_seconds : float, optional
            Override the internal sustained-time timer (batch/offline mode).

        Returns
        -------
        dict with keys:
            class_id, class_key, label, confidence, low_confidence,
            all_scores (list[float]), risk (RiskResult.to_dict()).
        """
        pil_image = self._to_pil(image_input)
        probabilities = self._infer(pil_image)

        class_id   = int(np.argmax(probabilities))
        confidence = float(probabilities[class_id])
        class_key  = INDEX_TO_CLASS[class_id]

        risk_result: RiskResult = self.risk_calc.evaluate(
            class_id=class_id,
            confidence=confidence,
            override_sustained_seconds=override_sustained_seconds,
        )

        return {
            "class_id":       class_id,
            "class_key":      class_key,
            "label":          CLASS_NAMES[class_key],
            "confidence":     round(confidence, 4),
            "low_confidence": confidence < self.confidence_threshold,
            "all_scores":     [round(float(p), 4) for p in probabilities],
            "risk":           risk_result.to_dict(),
        }

    def predict_bytes(self, raw_bytes: bytes, **kwargs) -> dict:
        """Convenience wrapper that accepts raw image bytes directly."""
        return self.predict(raw_bytes, **kwargs)

    def reset_session(self) -> None:
        """Reset the risk calculator state between driver sessions."""
        self.risk_calc.reset()

    # ── Inference backends ────────────────────────────────────────────────────

    def _infer(self, image: Image.Image) -> np.ndarray:
        """Return softmax probability array of shape (NUM_CLASSES,)."""
        if self.backend == "pytorch":
            return self._infer_pytorch(image)
        return self._infer_onnx(image)

    def _infer_pytorch(self, image: Image.Image) -> np.ndarray:
        tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            logits = self._pt_model(tensor)
            probs  = F.softmax(logits, dim=1).squeeze(0)
        return probs.numpy()

    def _infer_onnx(self, image: Image.Image) -> np.ndarray:
        tensor = self.transform(image).unsqueeze(0).numpy()
        outputs = self._ort_session.run(["logits"], {"image": tensor})
        logits  = outputs[0][0]
        # Softmax in numpy
        exp     = np.exp(logits - np.max(logits))
        return exp / exp.sum()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_pil(image_input) -> Image.Image:
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        if isinstance(image_input, (str, Path)):
            return Image.open(image_input).convert("RGB")
        if isinstance(image_input, bytes):
            return Image.open(BytesIO(image_input)).convert("RGB")
        if isinstance(image_input, np.ndarray):
            return Image.fromarray(image_input).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    @staticmethod
    def _load_onnx(path: Path):
        """Lazy-import onnxruntime to keep it optional for PyTorch-only setups."""
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for ONNX inference. "
                "Install it with: pip install onnxruntime"
            ) from exc

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession(str(path), providers=providers)
        return session
