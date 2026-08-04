"""
SmartHelp - Micro-service FastAPI de support client multimodal (Audio & Vision)
Point d'entrée principal de l'application.

Feature: fastapi-routes
- Middleware CORS configurable via variables d'environnement
- Middleware de logging (X-Request-ID, X-Process-Time)
- Gestionnaires d'erreurs globaux (HTTPException, ValidationError, Exception)
- Health-checks étendus : /health, /health/details, /health/ready
"""

import os
import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

load_dotenv()

from app.routes import support
from app.routes.health import router as health_router
from app.middleware.logging_middleware import RequestLoggingMiddleware
from app.middleware.error_handler import (
    http_exception_handler,
    validation_exception_handler,
    unhandled_exception_handler,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SmartHelp API",
    version="1.4",
    description="API de support client multimodal avec ASR, Vision et RAG",
    # Désactiver les handlers d'exception par défaut pour utiliser les nôtres
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# Middleware CORS
# Les origines autorisées sont lues depuis CORS_ORIGINS (séparées par virgule).
# Par défaut : toutes les origines sont autorisées (dev uniquement).
# ---------------------------------------------------------------------------
_cors_origins_raw = os.getenv("CORS_ORIGINS", "*")
_cors_origins = (
    [o.strip() for o in _cors_origins_raw.split(",")]
    if _cors_origins_raw != "*"
    else ["*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_origins_raw != "*",  # credentials incompatible avec "*"
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Process-Time"],
)

# ---------------------------------------------------------------------------
# Middleware de logging
# ---------------------------------------------------------------------------
app.add_middleware(RequestLoggingMiddleware)

# ---------------------------------------------------------------------------
# Gestionnaires d'erreurs globaux
# ---------------------------------------------------------------------------
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
app.include_router(support.router)
app.include_router(health_router)


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "SmartHelp",
        "version": "1.4",
        "features": ["support-ticket", "fastapi-routes"],
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health/details",
    }
