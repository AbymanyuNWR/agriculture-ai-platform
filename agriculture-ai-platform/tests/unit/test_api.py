import pytest
from fastapi.testclient import TestClient
from src.api.app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Agriculture AI Platform" in response.json()["message"]

def test_diagnosis_endpoint():
    # This would need actual image data
    # Just testing endpoint exists
    response = client.post("/api/v1/diagnosis/predict")
    assert response.status_code in [400, 422]  # Expected without image

def test_crops_endpoint():
    response = client.get("/api/v1/crops")
    # Should return 401 without auth
    assert response.status_code == 403

def test_knowledge_endpoint():
    response = client.get("/api/v1/knowledge/diseases")
    # Should work without auth for public knowledge
    assert response.status_code in [200, 403]
