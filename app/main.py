"""
SmartHelp - Micro-service FastAPI de support client multimodal (Audio & Vision)
Point d'entrée principal de l'application.
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI

# Chargement des variables d'environnement
load_dotenv()

from app.routes import support

app = FastAPI(
    title="SmartHelp API",
    version="1.0",
    description="API de support client multimodal avec ASR, Vision et RAG"
)

# Inclusion des routes
app.include_router(support.router)

@app.get("/")
async def root():
    return {
        "service": "SmartHelp",
        "version": "1.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
