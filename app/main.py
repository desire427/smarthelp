"""
SmartHelp - Micro-service FastAPI de support client multimodal (Audio & Vision)
Point d'entrée principal de l'application.

Feature: audio-transcription
- Ajout du router audio avec les endpoints /audio/transcribe
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

from app.routes import support
from app.routes import audio

app = FastAPI(
    title="SmartHelp API",
    version="1.1",
    description="API de support client multimodal avec ASR, Vision et RAG"
)

# Routes
app.include_router(support.router)
app.include_router(audio.router)


@app.get("/")
async def root():
    return {
        "service": "SmartHelp",
        "version": "1.1",
        "features": ["support-ticket", "audio-transcription"],
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
