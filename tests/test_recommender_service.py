import pandas as pd

from src.recommender_service import RecommenderService


class FakePrediction:
    def __init__(self, est):
        self.est = est


class FakeModel:
    def __init__(self, scores):
        self.scores = scores

    def predict(self, uid, iid):
        return FakePrediction(self.scores.get((uid, iid), 0.0))


def make_ratings():
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3],
            "item_id": [10, 20, 10, 30, 40],
            "rating": [5, 4, 3, 5, 4],
            "title": ["A", "B", "A", "C", "D"],
        }
    )


def test_known_user_recommendations_exclude_seen_items():
    service = RecommenderService(
        model=FakeModel({(1, 10): 5, (1, 20): 4, (1, 30): 3, (1, 40): 2}),
        ratings=make_ratings(),
    )

    response = service.recommend(user_id=1, k=5, exclude_seen=True)
    movie_ids = [rec["movie_id"] for rec in response["recommendations"]]

    assert response["fallback_used"] is False
    assert response["model_name"] == "svd"
    assert movie_ids == [30, 40]


def test_unknown_user_uses_popularity_fallback():
    service = RecommenderService(model=FakeModel({}), ratings=make_ratings())

    response = service.recommend(user_id=999, k=2, exclude_seen=True)

    assert response["fallback_used"] is True
    assert response["model_name"] == "popularity_fallback"
    assert len(response["recommendations"]) == 2
    assert response["recommendations"][0]["reason"] == "cold_start_popularity"


def test_health_reports_loaded_artifact_counts():
    service = RecommenderService(model=FakeModel({}), ratings=make_ratings())

    health = service.health()

    assert health["status"] == "healthy"
    assert health["users"] == 3
    assert health["movies"] == 4
    assert health["ratings"] == 5
