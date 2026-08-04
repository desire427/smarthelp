"""
Tests unitaires pour ASRService.

Feature: audio-transcription
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from app.services.asr_service import ASRService, ASRError


@pytest.fixture(autouse=True)
def reset_singleton():
    """Réinitialise le singleton entre les tests."""
    ASRService._instance = None
    ASRService._pipeline = None
    yield
    ASRService._instance = None
    ASRService._pipeline = None


class TestASRServiceFormats:
    def test_supported_formats_contains_wav(self):
        assert ".wav" in ASRService.SUPPORTED_FORMATS

    def test_supported_formats_contains_mp3(self):
        assert ".mp3" in ASRService.SUPPORTED_FORMATS

    def test_supported_formats_contains_flac(self):
        assert ".flac" in ASRService.SUPPORTED_FORMATS

    def test_unsupported_format_raises(self, tmp_path):
        # Crée un faux fichier avec une extension non supportée
        fake = tmp_path / "audio.xyz"
        fake.write_bytes(b"fake audio data")

        with patch("app.services.asr_service.pipeline") as mock_pipeline:
            mock_pipeline.return_value = MagicMock()
            service = ASRService()
            with pytest.raises(ASRError, match="Format non supporté"):
                service.transcribe_full(str(fake))


class TestASRServiceTranscription:
    def test_transcribe_returns_text(self, tmp_path):
        fake_wav = tmp_path / "test.wav"
        fake_wav.write_bytes(b"RIFF fake wav content")

        mock_result = {"text": "Bonjour le monde", "chunks": [], "language": "fr"}

        with patch("app.services.asr_service.pipeline") as mock_pipeline:
            mock_instance = MagicMock(return_value=mock_result)
            mock_pipeline.return_value = mock_instance

            service = ASRService()
            text = service.transcribe(str(fake_wav))

        assert text == "Bonjour le monde"

    def test_transcribe_full_returns_metadata(self, tmp_path):
        fake_wav = tmp_path / "sample.wav"
        fake_wav.write_bytes(b"RIFF fake wav content")

        mock_result = {
            "text": "  Test transcription  ",
            "chunks": [{"timestamp": [0.0, 1.5], "text": "Test transcription"}],
            "language": "fr",
        }

        with patch("app.services.asr_service.pipeline") as mock_pipeline:
            mock_pipeline.return_value = MagicMock(return_value=mock_result)

            service = ASRService()
            result = service.transcribe_full(str(fake_wav))

        assert result["text"] == "Test transcription"
        assert result["language"] == "fr"
        assert len(result["chunks"]) == 1

    def test_missing_file_raises(self):
        with patch("app.services.asr_service.pipeline") as mock_pipeline:
            mock_pipeline.return_value = MagicMock()
            service = ASRService()
            with pytest.raises(ASRError, match="introuvable"):
                service.transcribe_full("/tmp/nonexistent_audio_12345.wav")
