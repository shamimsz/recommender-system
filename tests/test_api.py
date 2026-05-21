from fastapi.testclient import TestClient

import src.api as api
from tests.test_recommender_service import FakeModel, make_ratings
from src.recommender_service import RecommenderService


def make_service():
    return RecommenderService(
        model=FakeModel({(1, 30): 4.5, (1, 40): 3.5}),
        ratings=make_ratings(),
        model_version="test",
    )


def test_health_uses_loaded_service(monkeypatch):
    monkeypatch.setattr(api, "service", make_service())
    monkeypatch.setattr(api, "startup_error", None)
    client = TestClient(api.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_name"] == "svd"


def test_recommendations_endpoint_returns_hardened_schema(monkeypatch):
    monkeypatch.setattr(api, "service", make_service())
    monkeypatch.setattr(api, "startup_error", None)
    client = TestClient(api.app)

    response = client.get("/recommendations", params={"user_id": 1, "k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == 1
    assert body["k_requested"] == 2
    assert body["fallback_used"] is False
    assert "latency_ms" in body
    assert body["recommendations"][0]["movie_id"] == 30


def test_recommend_alias_maps_n_to_k(monkeypatch):
    monkeypatch.setattr(api, "service", make_service())
    monkeypatch.setattr(api, "startup_error", None)
    client = TestClient(api.app)

    response = client.get("/recommend", params={"user_id": 1, "n": 1})

    assert response.status_code == 200
    assert response.json()["k_requested"] == 1
    assert len(response.json()["recommendations"]) == 1


def test_missing_artifacts_return_503(monkeypatch):
    monkeypatch.setattr(api, "service", None)
    monkeypatch.setattr(api, "startup_error", "Missing required artifact(s): model")
    client = TestClient(api.app)

    response = client.get("/recommendations", params={"user_id": 1, "k": 2})

    assert response.status_code == 503
    assert "Missing required artifact" in response.json()["detail"]
