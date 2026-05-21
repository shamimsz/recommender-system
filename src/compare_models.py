from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Protocol

import mlflow
import numpy as np
import pandas as pd
from surprise import Dataset, KNNWithMeans, Reader, SVD

from src.evaluation import (
    build_recommendations,
    ranking_metrics,
    rating_metrics,
    relevant_items_by_user,
    seen_items_by_user,
    temporal_expanding_splits,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "cleaned" / "cleaned_ratings_with_time.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "model_comparison.csv"
DEFAULT_EXPERIMENT = "MovieRecommender_ResearchComparison"


class RecommenderModel(Protocol):
    name: str

    def fit(self, train: pd.DataFrame) -> None:
        ...

    def predict_rating(self, user_id: int, item_id: int) -> float:
        ...

    def score_item(self, user_id: int, item_id: int) -> float:
        ...


class PopularityBaseline:
    """Global mean for rating prediction, smoothed item mean for ranking."""

    name = "popularity_baseline"

    def __init__(self, shrinkage: float = 10.0) -> None:
        self.shrinkage = shrinkage
        self.global_mean = 0.0
        self.item_scores: dict[int, float] = {}

    def fit(self, train: pd.DataFrame) -> None:
        self.global_mean = float(train["rating"].mean())
        grouped = train.groupby("item_id")["rating"].agg(["sum", "count"])
        smoothed = (grouped["sum"] + self.shrinkage * self.global_mean) / (
            grouped["count"] + self.shrinkage
        )
        self.item_scores = {
            int(item_id): float(score)
            for item_id, score in smoothed.sort_values(ascending=False).items()
        }

    def predict_rating(self, user_id: int, item_id: int) -> float:
        return self.global_mean

    def score_item(self, user_id: int, item_id: int) -> float:
        return self.item_scores.get(int(item_id), self.global_mean)


class SurpriseModel:
    def __init__(self, name: str, algo) -> None:
        self.name = name
        self.algo = algo

    def fit(self, train: pd.DataFrame) -> None:
        reader = Reader(rating_scale=(1, 5))
        trainset = Dataset.load_from_df(
            train[["user_id", "item_id", "rating"]],
            reader,
        ).build_full_trainset()
        self.algo.fit(trainset)

    def predict_rating(self, user_id: int, item_id: int) -> float:
        return float(self.algo.predict(uid=int(user_id), iid=int(item_id)).est)

    def score_item(self, user_id: int, item_id: int) -> float:
        return self.predict_rating(user_id, item_id)


def build_models(svd_factors: int = 100) -> list[RecommenderModel]:
    return [
        PopularityBaseline(),
        SurpriseModel(
            "knn_with_means",
            KNNWithMeans(
                k=40,
                min_k=3,
                sim_options={"name": "pearson_baseline", "user_based": False},
                verbose=False,
            ),
        ),
        SurpriseModel(
            "svd",
            SVD(
                n_factors=svd_factors,
                n_epochs=20,
                lr_all=0.005,
                reg_all=0.02,
                random_state=42,
            ),
        ),
    ]


def load_ratings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp_datetime"] = pd.to_datetime(df["timestamp_datetime"])
    return df


def _fold_metrics(
    model: RecommenderModel,
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int,
    relevance_threshold: float,
) -> dict[str, float]:
    actual = test["rating"].tolist()
    predicted = [
        model.predict_rating(int(row.user_id), int(row.item_id))
        for row in test.itertuples(index=False)
    ]
    metrics = rating_metrics(actual, predicted)

    train_items = sorted(int(item_id) for item_id in train["item_id"].unique())
    seen = seen_items_by_user(train)
    relevant = relevant_items_by_user(
        test[test["item_id"].isin(train_items)],
        relevance_threshold=relevance_threshold,
    )
    users = sorted(int(user_id) for user_id in test["user_id"].unique())
    recommendations = build_recommendations(
        users=users,
        all_items=train_items,
        seen_by_user=seen,
        score_fn=model.score_item,
        k=k,
        exclude_seen=True,
    )
    metrics.update(
        ranking_metrics(
            recommendations_by_user=recommendations,
            relevant_by_user=relevant,
            all_items=train_items,
            k=k,
        )
    )
    return metrics


def evaluate_models(
    df: pd.DataFrame,
    n_splits: int = 5,
    k: int = 10,
    relevance_threshold: float = 4.0,
    svd_factors: int = 100,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    folds = temporal_expanding_splits(df, n_splits=n_splits)

    for fold in folds:
        for model in build_models(svd_factors=svd_factors):
            print(f"Evaluating {model.name} on fold {fold.fold}/{n_splits}...")
            model.fit(fold.train)
            metrics = _fold_metrics(
                model=model,
                train=fold.train,
                test=fold.test,
                k=k,
                relevance_threshold=relevance_threshold,
            )
            rows.append(
                {
                    "model_name": model.name,
                    "fold": fold.fold,
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    f"precision_at_{k}": metrics[f"precision_at_{k}"],
                    f"recall_at_{k}": metrics[f"recall_at_{k}"],
                    f"ndcg_at_{k}": metrics[f"ndcg_at_{k}"],
                    f"coverage_at_{k}": metrics[f"coverage_at_{k}"],
                }
            )

    return pd.DataFrame(rows)


def summarize_results(results: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    metric_cols = [
        "rmse",
        "mae",
        f"precision_at_{k}",
        f"recall_at_{k}",
        f"ndcg_at_{k}",
        f"coverage_at_{k}",
    ]
    summary = results.groupby("model_name")[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def log_to_mlflow(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    experiment_name: str,
    k: int,
    n_splits: int,
    relevance_threshold: float,
    svd_factors: int,
) -> None:
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}",
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    for model_name, model_rows in results.groupby("model_name"):
        with mlflow.start_run(run_name=f"{model_name}_temporal_cv"):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("evaluation_type", "temporal_expanding_cv")
            mlflow.log_param("n_splits", n_splits)
            mlflow.log_param("top_k", k)
            mlflow.log_param("relevance_threshold", relevance_threshold)
            if model_name == "svd":
                mlflow.log_param("n_factors", svd_factors)

            for row in model_rows.itertuples(index=False):
                for metric in [
                    "rmse",
                    "mae",
                    f"precision_at_{k}",
                    f"recall_at_{k}",
                    f"ndcg_at_{k}",
                    f"coverage_at_{k}",
                ]:
                    mlflow.log_metric(f"fold_{row.fold}_{metric}", float(getattr(row, metric)))

            summary_row = summary[summary["model_name"] == model_name].iloc[0]
            for col, value in summary_row.items():
                if col != "model_name":
                    mlflow.log_metric(col, float(value) if not pd.isna(value) else 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare recommender models with temporal CV.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--relevance-threshold", type=float, default=4.0)
    parser.add_argument("--svd-factors", type=int, default=100)
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_ratings(args.data_path)
    if args.sample_rows is not None:
        df = df.sort_values("timestamp_datetime").head(args.sample_rows).copy()
        print(f"Using first {len(df)} rows for smoke testing.")

    results = evaluate_models(
        df=df,
        n_splits=args.n_splits,
        k=args.k,
        relevance_threshold=args.relevance_threshold,
        svd_factors=args.svd_factors,
    )
    summary = summarize_results(results, k=args.k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\nFold-level results")
    print(results.to_string(index=False))
    print("\nSummary")
    print(summary.to_string(index=False))
    print(f"\nSaved fold results to {args.output}")
    print(f"Saved summary to {summary_path}")

    if not args.no_mlflow:
        log_to_mlflow(
            results=results,
            summary=summary,
            experiment_name=DEFAULT_EXPERIMENT,
            k=args.k,
            n_splits=args.n_splits,
            relevance_threshold=args.relevance_threshold,
            svd_factors=args.svd_factors,
        )
        print(f"Logged results to MLflow experiment: {DEFAULT_EXPERIMENT}")


if __name__ == "__main__":
    main()
