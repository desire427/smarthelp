"""
Gestionnaire d'erreurs global pour FastAPI.

Feature: fastapi-routes
- Capture les exceptions non gérées et retourne une réponse JSON cohérente
- Distingue les erreurs HTTP (HTTPException) des erreurs inattendues (500)
- Logue les erreurs 500 avec la stack trace complète
"""

import logging
import traceback

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("smarthelp.errors")


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    Gère les HTTPException levées par FastAPI ou Starlette.
    Retourne une réponse JSON structurée avec le code d'erreur et le message.
    """
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path),
            "request_id": request_id,
        },
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Gère les erreurs de validation Pydantic (corps de requête invalide, paramètres manquants).
    Retourne la liste des erreurs de validation avec leur localisation.
    """
    request_id = getattr(request.state, "request_id", None)
    errors = [
        {
            "field": " → ".join(str(loc) for loc in err["loc"]),
            "message": err["msg"],
            "type": err["type"],
        }
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "error": "Données de requête invalides.",
            "details": errors,
            "path": str(request.url.path),
            "request_id": request_id,
        },
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Capture toutes les exceptions non gérées et retourne un 500 propre.
    La stack trace complète est loguée côté serveur uniquement.
    """
    request_id = getattr(request.state, "request_id", None)
    logger.error(
        "Erreur non gérée | path=%s | request_id=%s\n%s",
        request.url.path,
        request_id,
        traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Une erreur interne est survenue. Veuillez réessayer.",
            "status_code": 500,
            "path": str(request.url.path),
            "request_id": request_id,
        },
    )
