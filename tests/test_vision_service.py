"""
Tests unitaires pour VisionService.

Feature: image-vision
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image

from app.services.vision_service import VisionService, VisionError


@pytest.fixture(autouse=True)
def reset_singleton():
    """Réinitialise le singleton entre les tests."""
    VisionService._instance = None
    VisionService._model = None
    VisionService._processor = None
    yield
    VisionService._instance = None
    VisionService._model = None
    VisionService._processor = None


def _make_mock_service():
    """Crée un VisionService avec le modèle CLIP mocké."""
    with patch("app.services.vision_service.CLIPModel") as mock_model_cls, \
         patch("app.services.vision_service.CLIPProcessor") as mock_proc_cls:
        mock_model_cls.from_pretrained.return_value = MagicMock()
        mock_proc_cls.from_pretrained.return_value = MagicMock()
        return VisionService()


class TestVisionServiceValidation:
    def test_missing_file_raises(self):
        service = _make_mock_service()
        with pytest.raises(VisionError, match="introuvable"):
            service._validate_image("/tmp/nonexistent_image_99.png")

    def test_unsupported_format_raises(self, tmp_path):
        fake = tmp_path / "img.xyz"
        fake.write_bytes(b"fake image")
        service = _make_mock_service()
        with pytest.raises(VisionError, match="Format non supporté"):
            service._validate_image(str(fake))

    def test_supported_formats_includes_jpg(self):
        assert ".jpg" in VisionService.SUPPORTED_FORMATS

    def test_supported_formats_includes_png(self):
        assert ".png" in VisionService.SUPPORTED_FORMATS

    def test_supported_formats_includes_webp(self):
        assert ".webp" in VisionService.SUPPORTED_FORMATS


class TestVisionServiceAnalysis:
    def _build_clip_output(self, probs_list):
        """Construit un faux output CLIP avec les probabilités données."""
        import torch
        mock_output = MagicMock()
        mock_output.logits_per_image = MagicMock()
        fake_tensor = MagicMock()
        fake_tensor.__getitem__ = lambda self, idx: probs_list
        fake_tensor.softmax = MagicMock(return_value=fake_tensor)
        mock_output.logits_per_image.softmax = MagicMock(return_value=fake_tensor)
        return mock_output

    def test_analyze_image_returns_categorie(self, tmp_path):
        # Crée une image PNG valide (1x1 pixel blanc)
        img_path = tmp_path / "test.png"
        Image.new("RGB", (1, 1), color=(255, 255, 255)).save(str(img_path))

        service = _make_mock_service()

        # Mock _run_clip directement pour isoler le test
        service._run_clip = MagicMock(return_value={
            "categorie": "Conforme / Bon état",
            "confiance": 0.82,
            "score_conforme": 0.82,
            "score_defaut": 0.18,
            "top3_prompts": [],
        })

        result = service.analyze_image(str(img_path))
        assert result["categorie"] == "Conforme / Bon état"
        assert 0.0 <= result["confiance"] <= 1.0

    def test_analyze_batch_returns_list(self, tmp_path):
        img1 = tmp_path / "a.png"
        img2 = tmp_path / "b.jpg"
        Image.new("RGB", (1, 1)).save(str(img1))
        Image.new("RGB", (1, 1)).save(str(img2))

        service = _make_mock_service()
        service._run_clip = MagicMock(return_value={
            "categorie": "Incertain",
            "confiance": 0.5,
            "score_conforme": 0.5,
            "score_defaut": 0.5,
            "top3_prompts": [],
        })

        results = service.analyze_batch([str(img1), str(img2)])
        assert len(results) == 2
        for r in results:
            assert "image_path" in r
            assert "categorie" in r

    def test_analyze_batch_handles_invalid_file(self, tmp_path):
        valid = tmp_path / "ok.png"
        Image.new("RGB", (1, 1)).save(str(valid))
        invalid = "/tmp/nonexistent_batch_img.png"

        service = _make_mock_service()
        service._run_clip = MagicMock(return_value={
            "categorie": "Conforme / Bon état",
            "confiance": 0.9,
            "score_conforme": 0.9,
            "score_defaut": 0.1,
            "top3_prompts": [],
        })

        results = service.analyze_batch([str(valid), invalid])
        assert len(results) == 2
        error_results = [r for r in results if "error" in r]
        assert len(error_results) == 1
