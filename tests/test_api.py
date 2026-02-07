"""Tests for REST API."""

import pytest


@pytest.fixture
def client():
    try:
        from fastapi.testclient import TestClient
        from igqk.api.app import app

        return TestClient(app)
    except ImportError:
        pytest.skip("FastAPI not installed")


class TestHealthAndInfo:
    def test_health(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_info(self, client):
        response = client.get("/api/v1/info")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "compression_methods" in data
        assert "ternary" in data["compression_methods"]


class TestJobs:
    def test_job_not_found(self, client):
        response = client.get("/api/v1/jobs/nonexistent")
        assert response.status_code == 404

    def test_train_endpoint(self, client):
        response = client.post(
            "/api/v1/train",
            json={
                "model": "simple_fc",
                "dataset": "mnist",
                "compression": "ternary",
                "epochs": 1,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"
