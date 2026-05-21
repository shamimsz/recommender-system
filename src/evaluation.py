from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    """One expanding-window temporal validation fold."""

    fold: int
    train: pd.DataFrame
    test: pd.DataFrame


def temporal_expanding_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    timestamp_col: str = "timestamp_datetime",
) -> list[Fold]:
    """Create expanding-window temporal splits.

    The first block is used for training, and each following block becomes a
    test fold while the training window expands to include all previous blocks.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1")
    if timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")
    if len(df) < n_splits + 1:
        raise ValueError("Need at least n_splits + 1 rows for temporal CV")

    sorted_df = df.sort_values(timestamp_col).reset_index(drop=True)
    block_size = len(sorted_df) // (n_splits + 1)
    if block_size == 0:
        raise ValueError("Not enough rows to create non-empty folds")

    folds: list[Fold] = []
    for fold_idx in range(n_splits):
        train_end = block_size * (fold_idx + 1)
        test_start = train_end
        test_end = block_size * (fold_idx + 2)
        if fold_idx == n_splits - 1:
            test_end = len(sorted_df)

        train = sorted_df.iloc[:train_end].copy()
        test = sorted_df.iloc[test_start:test_end].copy()
        if train.empty or test.empty:
            raise ValueError("Temporal split produced an empty train or test fold")
        if train[timestamp_col].max() > test[timestamp_col].min():
            raise ValueError("Temporal split leakage detected")

        folds.append(Fold(fold=fold_idx + 1, train=train, test=test))

    return folds


def rating_metrics(actual: Iterable[float], predicted: Iterable[float]) -> dict[str, float]:
    actual_arr = np.asarray(list(actual), dtype=float)
    predicted_arr = np.asarray(list(predicted), dtype=float)
    if actual_arr.shape != predicted_arr.shape:
        raise ValueError("actual and predicted must have the same shape")
    if actual_arr.size == 0:
        raise ValueError("rating metrics require at least one prediction")

    errors = actual_arr - predicted_arr
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
    }


def seen_items_by_user(
    df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> dict[int, set[int]]:
    return {
        int(user_id): {int(item_id) for item_id in group[item_col].tolist()}
        for user_id, group in df.groupby(user_col)
    }


def relevant_items_by_user(
    df: pd.DataFrame,
    relevance_threshold: float = 4.0,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: str = "rating",
) -> dict[int, set[int]]:
    relevant = df[df[rating_col] >= relevance_threshold]
    return {
        int(user_id): {int(item_id) for item_id in group[item_col].tolist()}
        for user_id, group in relevant.groupby(user_col)
    }


def recommend_top_k_for_user(
    user_id: int,
    all_items: Iterable[int],
    seen_by_user: dict[int, set[int]],
    score_fn: Callable[[int, int], float],
    k: int = 10,
    exclude_seen: bool = True,
) -> list[int]:
    seen = seen_by_user.get(int(user_id), set()) if exclude_seen else set()
    candidates = [int(item_id) for item_id in all_items if int(item_id) not in seen]

    scored = [
        (item_id, float(score_fn(int(user_id), item_id)))
        for item_id in candidates
    ]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return [item_id for item_id, _score in scored[:k]]


def build_recommendations(
    users: Iterable[int],
    all_items: Iterable[int],
    seen_by_user: dict[int, set[int]],
    score_fn: Callable[[int, int], float],
    k: int = 10,
    exclude_seen: bool = True,
) -> dict[int, list[int]]:
    return {
        int(user_id): recommend_top_k_for_user(
            user_id=user_id,
            all_items=all_items,
            seen_by_user=seen_by_user,
            score_fn=score_fn,
            k=k,
            exclude_seen=exclude_seen,
        )
        for user_id in users
    }


def _dcg_at_k(items: list[int], relevant: set[int], k: int) -> float:
    score = 0.0
    for rank, item_id in enumerate(items[:k], start=1):
        if item_id in relevant:
            score += 1.0 / np.log2(rank + 1)
    return score


def ranking_metrics(
    recommendations_by_user: dict[int, list[int]],
    relevant_by_user: dict[int, set[int]],
    all_items: Iterable[int],
    k: int = 10,
) -> dict[str, float]:
    """Compute mean top-k metrics over users with at least one relevant item."""
    if k < 1:
        raise ValueError("k must be at least 1")

    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    recommended_items: set[int] = set()

    for user_id, recommendations in recommendations_by_user.items():
        top_k = [int(item_id) for item_id in recommendations[:k]]
        recommended_items.update(top_k)

        relevant = relevant_by_user.get(int(user_id), set())
        if not relevant:
            continue

        hits = len(set(top_k) & relevant)
        precisions.append(hits / k)
        recalls.append(hits / len(relevant))

        dcg = _dcg_at_k(top_k, relevant, k)
        ideal_len = min(len(relevant), k)
        ideal_dcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_len + 1))
        ndcgs.append(dcg / ideal_dcg if ideal_dcg else 0.0)

    item_count = len({int(item_id) for item_id in all_items})
    coverage = len(recommended_items) / item_count if item_count else 0.0

    return {
        f"precision_at_{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall_at_{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg_at_{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"coverage_at_{k}": float(coverage),
    }
