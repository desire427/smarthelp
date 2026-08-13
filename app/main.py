"""
SmartHelp – Micro-service FastAPI de support client multimodal
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import support, health

app = FastAPI(
    title="SmartHelp API",
    version="1.0",
    description="Micro-service de support client multimodal (Audio & Vision + RAG)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(support.router)
app.include_router(health.router)
