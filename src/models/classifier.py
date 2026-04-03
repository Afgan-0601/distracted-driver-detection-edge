"""
Model definition — MobileNetV2-based driver distraction classifier.

Why MobileNetV2?
* Designed for edge/mobile deployment (inverted residual bottlenecks).
* <4 M parameters — fits on Jetson Nano, Raspberry Pi 4, etc.
* Strong ImageNet pretrained weights available via torchvision.
* Easily exportable to ONNX / TFLite / TorchScript.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

from src.config import DROPOUT_RATE, MODELS_DIR, NUM_CLASSES, ONNX_MODEL_NAME, PT_MODEL_NAME


class DriverClassifier(nn.Module):
    """
    Transfer-learning classifier built on a MobileNetV2 backbone.

    The original classifier head is replaced with:
        GlobalAvgPool → Dropout → Linear(NUM_CLASSES)

    Parameters
    ----------
    num_classes : int
        Number of output classes (default: 10 from config).
    pretrained : bool
        Load ImageNet weights for the backbone.
    dropout_rate : float
        Dropout probability before the final linear layer.
    freeze_backbone : bool
        If True, only train the classifier head (feature extraction mode).
        Set to False for full fine-tuning after warm-up.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout_rate: float = DROPOUT_RATE,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)

        # Freeze backbone for initial warm-up training
        if freeze_backbone:
            for param in backbone.features.parameters():
                param.requires_grad = False

        # Replace the default classifier
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )

        self.backbone = backbone
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def unfreeze_backbone(self) -> None:
        """Un-freeze all backbone parameters for full fine-tuning."""
        for param in self.backbone.features.parameters():
            param.requires_grad = True

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        """Save model state dict. Returns the path written to."""
        save_path = Path(path or MODELS_DIR / PT_MODEL_NAME)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path)
        return save_path

    @classmethod
    def load(cls, path: Optional[Path] = None, **kwargs) -> "DriverClassifier":
        """Load a saved state dict or trainer checkpoint into a new model instance."""
        load_path = Path(path or MODELS_DIR / PT_MODEL_NAME)
        model = cls(**kwargs)
        data = torch.load(load_path, map_location="cpu", weights_only=False)
        state_dict = data["model_state"] if isinstance(data, dict) and "model_state" in data else data
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def export_onnx(
        self,
        path: Optional[Path] = None,
        input_shape: tuple = (1, 3, 224, 224),
        opset: int = 17,
        dynamo=False,
    ) -> Path:
        """
        Export to ONNX for cross-platform edge inference.

        The exported graph uses dynamic batch size so it works for both
        single-frame and batched inference on edge devices.
        """
        export_path = Path(path or MODELS_DIR / ONNX_MODEL_NAME)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        self.eval()
        dummy = torch.zeros(*input_shape)

        torch.onnx.export(
            self,
            dummy,
            str(export_path),
            opset_version=opset,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={
                "image":  {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            do_constant_folding=True,
        )
        return export_path

    def export_torchscript(self, path: Optional[Path] = None) -> Path:
        """Export to TorchScript (alternative to ONNX for PyTorch deployments)."""
        export_path = Path(path or MODELS_DIR / "driver_classifier.torchscript.pt")
        export_path.parent.mkdir(parents=True, exist_ok=True)
        scripted = torch.jit.script(self)
        scripted.save(str(export_path))
        return export_path


def build_model(pretrained: bool = True, freeze_backbone: bool = True) -> DriverClassifier:
    """Convenience factory used by training scripts."""
    return DriverClassifier(pretrained=pretrained, freeze_backbone=freeze_backbone)
