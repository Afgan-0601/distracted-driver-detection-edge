"""Smoke tests for the model architecture."""

import torch
import pytest

from src.models.classifier import DriverClassifier, build_model
from src.config import NUM_CLASSES, IMAGE_SIZE


class TestDriverClassifier:
    def setup_method(self):
        self.model = DriverClassifier(pretrained=False)

    def test_forward_pass_shape(self):
        x = torch.zeros(2, 3, *IMAGE_SIZE)
        out = self.model(x)
        assert out.shape == (2, NUM_CLASSES)

    def test_num_classes_matches_config(self):
        assert self.model.num_classes == NUM_CLASSES

    def test_frozen_backbone_no_grad(self):
        model = DriverClassifier(pretrained=False, freeze_backbone=True)
        for param in model.backbone.features.parameters():
            assert not param.requires_grad

    def test_unfreeze_backbone(self):
        model = DriverClassifier(pretrained=False, freeze_backbone=True)
        model.unfreeze_backbone()
        for param in model.backbone.features.parameters():
            assert param.requires_grad

    def test_save_and_load(self, tmp_path):
        save_path = tmp_path / "test_model.pt"
        self.model.save(save_path)
        loaded = DriverClassifier.load(save_path, pretrained=False)
        x = torch.zeros(1, 3, *IMAGE_SIZE)
        with torch.no_grad():
            out_orig   = self.model(x)
            out_loaded = loaded(x)
        assert torch.allclose(out_orig, out_loaded)

    def test_export_onnx(self, tmp_path):
        pytest.importorskip("onnxruntime")
        import onnxruntime as ort
        import numpy as np

        onnx_path = tmp_path / "model.onnx"
        self.model.eval()
        self.model.export_onnx(path=onnx_path)
        assert onnx_path.exists()

        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        dummy = np.random.rand(1, 3, 224, 224).astype(np.float32)
        outputs = session.run(["logits"], {"image": dummy})
        assert outputs[0].shape == (1, NUM_CLASSES)
