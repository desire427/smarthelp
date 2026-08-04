"""
Middleware de logging des requêtes HTTP.

Feature: fastapi-routes
- Logue chaque requête entrante (méthode, URL, durée, statut)
- Ajoute un header X-Request-ID unique à chaque réponse
- Ajoute un header X-Process-Time avec la durée de traitement en ms
"""

import time
import uuid
import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("smarthelp.access")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware qui logue chaque requête HTTP avec sa durée et son statut.
    Ajoute les headers X-Request-ID et X-Process-Time à chaque réponse.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        # Attacher l'ID à l'état de la requête pour usage dans les routes
        request.state.request_id = request_id

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "%s %s | status=%s | duration=%.2fms | request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{duration_ms:.2f}ms"

        return response
