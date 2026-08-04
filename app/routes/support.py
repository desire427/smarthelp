"""
Routes pour le support ticket.
Gère les endpoints de l'API.
"""

import os
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse

from app.services.asr_service import ASRService
from app.services.vision_service import VisionService
from app.rag.search import RAGSearch
from app.utils.file_utils import save_temp_file, cleanup_temp_files
from app.utils.file_utils import ALLOWED_AUDIO, ALLOWED_IMAGE

router = APIRouter()

@router.post("/support-ticket")
async def support_ticket(
    audio: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    description: Optional[str] = Form(None),
):
    """
    Endpoint principal pour le traitement d'un ticket support.
    Accepte un fichier audio, une image et une description textuelle.
    """
    tmp_files = []
    transcribed_text = None
    vision_diagnosis = None

    try:
        # --- Traitement Audio (ASR) ---
        if audio is not None:
            ext = os.path.splitext(audio.filename)[1].lower()
            if ext not in ALLOWED_AUDIO:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Format audio non supporté: {ext}"}
                )
            
            path = save_temp_file(audio)
            tmp_files.append(path)
            asr_service = ASRService()
            transcribed_text = asr_service.transcribe(path)

        # --- Traitement Visuel (CLIP) ---
        if image is not None:
            ext = os.path.splitext(image.filename)[1].lower()
            if ext not in ALLOWED_IMAGE:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Format image non supporté: {ext}"}
                )
            
            path = save_temp_file(image)
            tmp_files.append(path)
            vision_service = VisionService()
            vision_diagnosis = vision_service.analyze_image(path)

        # --- Recherche RAG ---
        query_text = transcribed_text or description or ""
        rag_search = RAGSearch()
        matched_rule, similarity = rag_search.search(query_text)
        
        # --- Proposition de statut ---
        status = rag_search.propose_status(
            matched_rule, 
            similarity, 
            vision_diagnosis
        )

        return {
            "transcribed_text": transcribed_text,
            "vision_diagnosis": vision_diagnosis,
            "matched_policy": matched_rule or None,
            "similarity_score": round(similarity, 3),
            "proposed_status": status,
        }

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": f"Erreur serveur: {str(exc)}"}
        )

    finally:
        # Nettoyage des fichiers temporaires
        cleanup_temp_files(tmp_files)
