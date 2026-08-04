"""
Routes de monitoring et health-check étendus.

Feature: fastapi-routes
- GET /health         : statut de base (compatible load balancer)
- GET /health/details : statut détaillé par composant (env vars, services)
- GET /health/ready   : readiness check (vérifie que les dépendances sont disponibles)
"""

import os
import sys
import platform
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(tags=["Health"])

# Timestamp de démarrage de l'application
_START_TIME = datetime.now(timezone.utc)


@router.get("/health")
async def health():
    """
    Health-check simple. Retourne 200 si l'application est démarrée.
    Utilisé par les load balancers et orchestrateurs (K8s liveness probe).
    """
    return {"status": "healthy"}


@router.get("/health/details")
async def health_details():
    """
    Health-check détaillé.
    Vérifie la présence des variables d'environnement et retourne les métadonnées runtime.
    """
    uptime_seconds = (datetime.now(timezone.utc) - _START_TIME).total_seconds()

    # Vérification des variables d'environnement critiques
    required_env_vars = ["ASR_MODEL", "CLIP_MODEL", "EMBEDDING_MODEL"]
    env_status = {
        var: "ok" if os.getenv(var) else "missing"
        for var in required_env_vars
    }
    env_healthy = all(v == "ok" for v in env_status.values())

    return {
        "status": "healthy" if env_healthy else "degraded",
        "uptime_seconds": round(uptime_seconds, 1),
        "started_at": _START_TIME.isoformat(),
        "python_version": sys.version,
        "platform": platform.system(),
        "environment": {
            "vars": env_status,
            "asr_model": os.getenv("ASR_MODEL", "non défini"),
            "clip_model": os.getenv("CLIP_MODEL", "non défini"),
            "embedding_model": os.getenv("EMBEDDING_MODEL", "non défini"),
            "similarity_threshold": os.getenv("SIMILARITY_THRESHOLD", "0.35"),
            "vision_confidence_threshold": os.getenv("VISION_CONFIDENCE_THRESHOLD", "0.5"),
        },
    }


@router.get("/health/ready")
async def health_ready():
    """
    Readiness check.
    Vérifie que les variables d'environnement critiques sont définies.
    Retourne 503 si l'application n'est pas prête à traiter des requêtes.
    """
    required = {
        "ASR_MODEL": os.getenv("ASR_MODEL"),
        "CLIP_MODEL": os.getenv("CLIP_MODEL"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL"),
    }
    missing = [k for k, v in required.items() if not v]

    if missing:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "missing_config": missing,
                "message": "Des variables d'environnement obligatoires sont absentes.",
            },
        )

    return {"status": "ready"}
