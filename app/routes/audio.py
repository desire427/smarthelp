"""
Routes dédiées à la transcription audio.

Feature: audio-transcription
- Endpoint POST /transcribe : transcription simple
- Endpoint POST /transcribe/full : transcription avec métadonnées (chunks, langue)
"""

import os
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from app.services.asr_service import ASRService, ASRError
from app.utils.file_utils import save_temp_file, cleanup_temp_files

router = APIRouter(prefix="/audio", tags=["Audio Transcription"])


@router.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Transcrit un fichier audio en texte.

    Formats supportés : .mp3, .wav, .ogg, .flac, .m4a, .webm
    """
    tmp_files = []
    try:
        path = save_temp_file(audio)
        tmp_files.append(path)

        service = ASRService()
        text = service.transcribe(path)

        return {"filename": audio.filename, "transcription": text}

    except ASRError as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"Erreur serveur: {exc}"})
    finally:
        cleanup_temp_files(tmp_files)


@router.post("/transcribe/full")
async def transcribe_audio_full(audio: UploadFile = File(...)):
    """
    Transcrit un fichier audio et retourne le texte complet
    ainsi que les segments horodatés et la langue détectée.
    """
    tmp_files = []
    try:
        path = save_temp_file(audio)
        tmp_files.append(path)

        service = ASRService()
        result = service.transcribe_full(path)

        return {
            "filename": audio.filename,
            "transcription": result["text"],
            "language": result.get("language"),
            "chunks": result.get("chunks", []),
        }

    except ASRError as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"Erreur serveur: {exc}"})
    finally:
        cleanup_temp_files(tmp_files)
