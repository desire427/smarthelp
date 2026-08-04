"""
Service de reconnaissance vocale automatique (ASR).
Utilise le modèle Whisper pour la transcription audio.

Feature: audio-transcription
- Gestion des erreurs détaillée
- Support étendu des formats audio
- Détection de langue automatique
- Retour structuré avec métadonnées
"""

import os
import logging
from typing import Dict, Any

from transformers import pipeline

logger = logging.getLogger(__name__)


class ASRError(Exception):
    """Erreur spécifique au service ASR."""
    pass


class ASRService:
    """
    Service pour la transcription audio avec Whisper.
    Singleton pour éviter de charger le modèle plusieurs fois.
    """

    _instance = None
    _pipeline = None

    # Formats audio supportés (étendus dans cette feature)
    SUPPORTED_FORMATS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".webm"}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ASRService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._pipeline is None:
            self._load_model()

    def _load_model(self):
        """
        Charge le modèle Whisper depuis les variables d'environnement.
        Lève une ASRError si le chargement échoue.
        """
        model_name = os.getenv("ASR_MODEL", "openai/whisper-small")
        try:
            logger.info("Chargement du modèle ASR : %s", model_name)
            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=model_name,
                return_timestamps=True,
            )
            logger.info("Modèle ASR chargé avec succès.")
        except Exception as exc:
            logger.error("Échec du chargement du modèle ASR : %s", exc)
            raise ASRError(f"Impossible de charger le modèle ASR '{model_name}': {exc}") from exc

    def transcribe(self, audio_path: str) -> str:
        """
        Transcrit un fichier audio en texte (interface simplifiée).

        Args:
            audio_path: Chemin vers le fichier audio

        Returns:
            Texte transcrit
        """
        return self.transcribe_full(audio_path)["text"]

    def transcribe_full(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcrit un fichier audio et retourne les métadonnées complètes.

        Args:
            audio_path: Chemin vers le fichier audio

        Returns:
            Dictionnaire contenant :
              - text   : texte transcrit
              - chunks : segments horodatés (si disponibles)
              - language: langue détectée (si disponible)

        Raises:
            ASRError: si le fichier est absent ou le format non supporté
        """
        if not os.path.isfile(audio_path):
            raise ASRError(f"Fichier audio introuvable : {audio_path}")

        ext = os.path.splitext(audio_path)[1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ASRError(
                f"Format non supporté '{ext}'. "
                f"Formats acceptés : {sorted(self.SUPPORTED_FORMATS)}"
            )

        try:
            result = self._pipeline(audio_path)
        except Exception as exc:
            logger.error("Erreur lors de la transcription de '%s': %s", audio_path, exc)
            raise ASRError(f"Erreur de transcription : {exc}") from exc

        text = result.get("text", "").strip()
        chunks = result.get("chunks", [])

        return {
            "text": text,
            "chunks": chunks,
            "language": result.get("language", None),
            "audio_path": audio_path,
        }
