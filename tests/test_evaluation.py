import math

import pandas as pd

from src.evaluation import (
    build_recommendations,
    ranking_metrics,
    rating_metrics,
    seen_items_by_user,
    temporal_expanding_splits,
)


def test_temporal_expanding_splits_preserve_order():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "item_id": [10, 11, 12, 13, 14, 15],
            "rating": [5, 4, 3, 2, 4, 5],
            "timestamp_datetime": pd.date_range("2020-01-01", periods=6),
        }
    )

    folds = temporal_expanding_splits(df, n_splits=2)

    assert len(folds) == 2
    for fold in folds:
        assert fold.train["timestamp_datetime"].max() <= fold.test["timestamp_datetime"].min()
        assert len(fold.train) > 0
        assert len(fold.test) > 0
    assert len(folds[1].train) > len(folds[0].train)


def test_rating_metrics_match_hand_calculation():
    metrics = rating_metrics(actual=[5, 3, 1], predicted=[4, 3, 3])

    assert math.isclose(metrics["rmse"], math.sqrt((1 + 0 + 4) / 3))
    assert math.isclose(metrics["mae"], 1.0)


def test_ranking_metrics_match_hand_calculation():
    recommendations = {
        1: [10, 20, 30],
        2: [30, 40, 50],
    }
    relevant = {
        1: {10, 30},
        2: {60},
    }

    metrics = ranking_metrics(recommendations, relevant, all_items=[10, 20, 30, 40, 50, 60], k=3)

    assert math.isclose(metrics["precision_at_3"], (2 / 3 + 0) / 2)
    assert math.isclose(metrics["recall_at_3"], (1.0 + 0) / 2)
    assert 0 < metrics["ndcg_at_3"] < 1
    assert math.isclose(metrics["coverage_at_3"], 5 / 6)


def test_build_recommendations_excludes_seen_items():
    train = pd.DataFrame(
        {
            "user_id": [1, 1],
            "item_id": [10, 20],
            "rating": [5, 4],
        }
    )
    seen = seen_items_by_user(train)
    scores = {10: 1.0, 20: 0.9, 30: 0.8, 40: 0.7}

    recommendations = build_recommendations(
        users=[1],
        all_items=[10, 20, 30, 40],
        seen_by_user=seen,
        score_fn=lambda _user_id, item_id: scores[item_id],
        k=2,
    )

    assert recommendations[1] == [30, 40]
