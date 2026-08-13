"""
POST /support-ticket — endpoint unique d'ingestion multimodal.
Accepte un audio (.mp3, .wav), une image (.png, .jpg, .jpeg) et/ou un texte.
"""

import os
import shutil
import tempfile
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.services import asr_service, vision_service
from app.rag import search
from app.rag.search import is_relevant

router = APIRouter(tags=["Support"])

ALLOWED_AUDIO = {".mp3", ".wav", ".webm", ".ogg"}
ALLOWED_IMAGE = {".png", ".jpg", ".jpeg", ".webp"}


def _save_tmp(upload: UploadFile) -> str:
    """Sauvegarde un fichier uploadé dans un fichier temporaire et retourne son chemin."""
    suffix = os.path.splitext(upload.filename)[1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with tmp:
        shutil.copyfileobj(upload.file, tmp)
    return tmp.name


@router.post("/support-ticket")
async def support_ticket(
    audio: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    description: Optional[str] = Form(None),
):
    tmp_files = []
    transcribed_text = None
    vision_diagnosis = None

    try:
        # 1. Transcription audio (ASR)
        if audio is not None:
            ext = os.path.splitext(audio.filename)[1].lower()
            if ext not in ALLOWED_AUDIO:
                raise HTTPException(status_code=400, detail=f"Format audio non supporté : {ext}. Formats acceptés : {ALLOWED_AUDIO}")
            path = _save_tmp(audio)
            tmp_files.append(path)
            transcribed_text = asr_service.transcribe(path)

        # 2. Analyse image (Vision)
        if image is not None:
            ext = os.path.splitext(image.filename)[1].lower()
            if ext not in ALLOWED_IMAGE:
                raise HTTPException(status_code=400, detail=f"Format image non supporté : {ext}. Formats acceptés : {ALLOWED_IMAGE}")
            path = _save_tmp(image)
            tmp_files.append(path)
            vision_diagnosis = vision_service.analyze_image(path)

        # 3. Recherche RAG
        query = transcribed_text or description or ""
        if not query.strip():
            return {
                "transcribed_text": None,
                "vision_diagnosis": vision_diagnosis,
                "matched_policy": None,
                "similarity_score": 0.0,
                "proposed_status": "À vérifier - Aucun texte fourni",
            }

        if not is_relevant(query):
            return {
                "transcribed_text": transcribed_text,
                "vision_diagnosis": vision_diagnosis,
                "matched_policy": None,
                "similarity_score": 0.0,
                "proposed_status": "Hors sujet - Contenu non lié au support client",
            }

        # Détection prioritaire : dommage causé par l'utilisateur
        if search.is_user_fault(query):
            return {
                "transcribed_text": transcribed_text,
                "vision_diagnosis": vision_diagnosis,
                "matched_policy": "Un produit cassé ou endommagé par l'utilisateur lui-même n'est pas éligible au remboursement ni à l'échange.",
                "similarity_score": 1.0,
                "proposed_status": "Refusé - Dommage causé par l'utilisateur",
            }

        matched_rule, similarity = search.search(query)

        # 4. Statut proposé
        proposed_status = search.propose_status(matched_rule, similarity, vision_diagnosis, query)

        return {
            "transcribed_text": transcribed_text,
            "vision_diagnosis": vision_diagnosis,
            "matched_policy": matched_rule or None,
            "similarity_score": round(similarity, 3),
            "proposed_status": proposed_status,
        }

    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})

    finally:
        for path in tmp_files:
            if os.path.exists(path):
                os.remove(path)
