from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Recommendation:
    rank: int
    movie_id: int
    title: str
    score: float
    reason: str


class ArtifactError(RuntimeError):
    pass


class RecommenderService:
    def __init__(
        self,
        model: Any,
        ratings: pd.DataFrame,
        model_name: str = "svd",
        model_version: str = "local",
    ) -> None:
        self.model = model
        self.ratings = ratings.copy()
        self.model_name = model_name
        self.model_version = model_version
        self.movies = (
            self.ratings[["item_id", "title"]]
            .drop_duplicates()
            .sort_values("item_id")
            .set_index("item_id")
        )
        self.known_users = {int(user_id) for user_id in self.ratings["user_id"].unique()}
        self.seen_by_user = {
            int(user_id): {int(item_id) for item_id in group["item_id"].tolist()}
            for user_id, group in self.ratings.groupby("user_id")
        }
        self.popularity_scores = self._build_popularity_scores()

    @classmethod
    def from_artifacts(
        cls,
        model_path: Path,
        data_path: Path,
        model_name: str = "svd",
        model_version: str = "local",
    ) -> "RecommenderService":
        missing = [str(path) for path in [model_path, data_path] if not path.exists()]
        if missing:
            raise ArtifactError(
                "Missing required artifact(s): "
                + ", ".join(missing)
                + ". Run `python -m src.train_svd` after preparing the data."
            )

        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        ratings = pd.read_csv(data_path)
        return cls(
            model=model,
            ratings=ratings,
            model_name=model_name,
            model_version=model_version,
        )

    def health(self) -> dict[str, Any]:
        return {
            "status": "healthy",
            "model_name": self.model_name,
            "model_version": self.model_version,
            "users": len(self.known_users),
            "movies": len(self.movies),
            "ratings": len(self.ratings),
        }

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> dict[str, Any]:
        start_time = time.perf_counter()
        user_id = int(user_id)
        fallback_used = user_id not in self.known_users

        if fallback_used:
            ranked = self._popular_recommendations(user_id, k, exclude_seen)
            response_model_name = "popularity_fallback"
        else:
            ranked = self._model_recommendations(user_id, k, exclude_seen)
            response_model_name = self.model_name

        latency_ms = (time.perf_counter() - start_time) * 1000
        return {
            "user_id": user_id,
            "k_requested": k,
            "model_name": response_model_name,
            "model_version": self.model_version,
            "fallback_used": fallback_used,
            "exclude_seen": exclude_seen,
            "latency_ms": round(latency_ms, 2),
            "recommendations": [rec.__dict__ for rec in ranked],
        }

    def _build_popularity_scores(self, shrinkage: float = 10.0) -> dict[int, float]:
        global_mean = float(self.ratings["rating"].mean())
        grouped = self.ratings.groupby("item_id")["rating"].agg(["sum", "count"])
        smoothed = (grouped["sum"] + shrinkage * global_mean) / (
            grouped["count"] + shrinkage
        )
        return {int(item_id): float(score) for item_id, score in smoothed.items()}

    def _candidate_items(self, user_id: int, exclude_seen: bool) -> list[int]:
        seen = self.seen_by_user.get(user_id, set()) if exclude_seen else set()
        return [int(item_id) for item_id in self.movies.index.tolist() if int(item_id) not in seen]

    def _model_recommendations(
        self,
        user_id: int,
        k: int,
        exclude_seen: bool,
    ) -> list[Recommendation]:
        scored = []
        for item_id in self._candidate_items(user_id, exclude_seen):
            pred = self.model.predict(uid=user_id, iid=item_id)
            scored.append((item_id, float(pred.est)))

        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return self._format_recommendations(scored[:k], reason="personalized_svd_score")

    def _popular_recommendations(
        self,
        user_id: int,
        k: int,
        exclude_seen: bool,
    ) -> list[Recommendation]:
        scored = [
            (item_id, self.popularity_scores.get(item_id, 0.0))
            for item_id in self._candidate_items(user_id, exclude_seen)
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return self._format_recommendations(scored[:k], reason="cold_start_popularity")

    def _format_recommendations(
        self,
        scored_items: list[tuple[int, float]],
        reason: str,
    ) -> list[Recommendation]:
        recommendations = []
        for rank, (item_id, score) in enumerate(scored_items, start=1):
            title = str(self.movies.loc[item_id, "title"])
            recommendations.append(
                Recommendation(
                    rank=rank,
                    movie_id=int(item_id),
                    title=title,
                    score=round(float(score), 3),
                    reason=reason,
                )
            )
        return recommendations
