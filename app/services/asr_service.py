"""
Service ASR — transcription audio avec Whisper.
Le pipeline est chargé une seule fois en mémoire via @lru_cache.
"""

import os
from functools import lru_cache
from transformers import pipeline


@lru_cache(maxsize=1)
def _get_asr_pipeline():
    model = os.getenv("ASR_MODEL", "openai/whisper-small")
    return pipeline("automatic-speech-recognition", model=model)


def transcribe(audio_path: str) -> str:
    """Transcrit un fichier audio en texte."""
    asr = _get_asr_pipeline()
    result = asr(audio_path, return_timestamps=True, chunk_length_s=30)
    return result["text"].strip()
