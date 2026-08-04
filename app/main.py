"""
SmartHelp - Micro-service FastAPI de support client multimodal (Audio & Vision)
Point d'entrée principal de l'application.

Feature: image-vision
- Ajout du router vision avec les endpoints /vision/check et /vision/check/batch
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

from app.routes import support
from app.routes import vision

app = FastAPI(
    title="SmartHelp API",
    version="1.2",
    description="API de support client multimodal avec ASR, Vision et RAG"
)

# Routes
app.include_router(support.router)
app.include_router(vision.router)


@app.get("/")
async def root():
    return {
        "service": "SmartHelp",
        "version": "1.2",
        "features": ["support-ticket", "image-vision"],
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
