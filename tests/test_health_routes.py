"""
Tests unitaires pour les routes de health-check.

Feature: fastapi-routes
"""

import os
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Client de test FastAPI."""
    # Importer ici pour éviter les effets de bord au chargement du module
    from app.main import app
    return TestClient(app, raise_server_exceptions=False)


class TestHealthBasic:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"


class TestHealthDetails:
    def test_health_details_returns_200(self, client):
        response = client.get("/health/details")
        assert response.status_code == 200

    def test_health_details_has_required_keys(self, client):
        data = client.get("/health/details").json()
        assert "status" in data
        assert "uptime_seconds" in data
        assert "started_at" in data
        assert "environment" in data

    def test_health_details_uptime_is_positive(self, client):
        data = client.get("/health/details").json()
        assert data["uptime_seconds"] >= 0

    def test_health_details_status_healthy_with_env_vars(self, client):
        with patch.dict(os.environ, {
            "ASR_MODEL": "openai/whisper-small",
            "CLIP_MODEL": "openai/clip-vit-base-patch32",
            "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
        }):
            data = client.get("/health/details").json()
        assert data["status"] == "healthy"

    def test_health_details_degraded_without_env_vars(self, client):
        with patch.dict(os.environ, {}, clear=True):
            data = client.get("/health/details").json()
        assert data["status"] == "degraded"


class TestHealthReady:
    def test_health_ready_200_with_env_vars(self, client):
        with patch.dict(os.environ, {
            "ASR_MODEL": "openai/whisper-small",
            "CLIP_MODEL": "openai/clip-vit-base-patch32",
            "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
        }):
            response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    def test_health_ready_503_without_env_vars(self, client):
        with patch.dict(os.environ, {}, clear=True):
            response = client.get("/health/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"
        assert "missing_config" in data


class TestMiddlewareHeaders:
    def test_request_id_header_present(self, client):
        response = client.get("/health")
        assert "x-request-id" in response.headers

    def test_process_time_header_present(self, client):
        response = client.get("/health")
        assert "x-process-time" in response.headers

    def test_process_time_ends_with_ms(self, client):
        value = client.get("/health").headers.get("x-process-time", "")
        assert value.endswith("ms")

    def test_request_id_is_uuid_format(self, client):
        import uuid
        value = client.get("/health").headers.get("x-request-id", "")
        # Lève ValueError si format invalide
        uuid.UUID(value)


class TestErrorHandlers:
    def test_404_returns_json(self, client):
        response = client.get("/route_inexistante_xyz")
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "path" in data

    def test_404_includes_request_id(self, client):
        response = client.get("/route_inexistante_xyz")
        data = response.json()
        assert data.get("request_id") is not None
