"""
Service de reconnaissance vocale automatique (ASR).
Utilise le modèle Whisper pour la transcription audio.
"""

import os
from transformers import pipeline

class ASRService:
    """
    Service pour la transcription audio avec Whisper.
    Singleton pour éviter de charger le modèle plusieurs fois.
    """
    
    _instance = None
    _pipeline = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ASRService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._pipeline is None:
            self._load_model()
    
    def _load_model(self):
        """
        Charge le modèle Whisper.
        Le nom du modèle est lu depuis les variables d'environnement.
        """
        model_name = os.getenv("ASR_MODEL", "openai/whisper-small")
        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_name
        )
    
    def transcribe(self, audio_path: str) -> str:
        """
        Transcrit un fichier audio en texte.
        Supporte les fichiers de toute durée grâce au découpage en segments de 30s.

        Args:
            audio_path: Chemin vers le fichier audio

        Returns:
            Texte transcrit
        """
        result = self._pipeline(
            audio_path,
            return_timestamps=True,
            chunk_length_s=30,
        )
        return result["text"].strip()
