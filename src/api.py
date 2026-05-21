from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

from src.recommender_service import ArtifactError, RecommenderService


PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "svd_model.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "cleaned" / "cleaned_ratings_with_time.csv"

app = FastAPI(title="Movie Recommender API")
service: RecommenderService | None = None
startup_error: str | None = None

try:
    service = RecommenderService.from_artifacts(
        model_path=MODEL_PATH,
        data_path=DATA_PATH,
        model_name="svd",
        model_version="svd-n100-e20",
    )
except ArtifactError as exc:
    startup_error = str(exc)


def get_service() -> RecommenderService:
    if service is None:
        raise HTTPException(
            status_code=503,
            detail=startup_error or "Recommendation service is not available.",
        )
    return service


@app.get("/")
def root():
    return {
        "message": "Movie Recommender API is running.",
        "docs": "/docs",
        "recommendations": "/recommendations?user_id=196&k=10",
    }


@app.get("/health")
def health():
    if service is None:
        return {"status": "unhealthy", "detail": startup_error}
    return service.health()


@app.get("/recommendations")
def recommendations(
    user_id: int,
    k: int = Query(10, ge=1, le=50),
    exclude_seen: bool = True,
):
    return get_service().recommend(user_id=user_id, k=k, exclude_seen=exclude_seen)


@app.get("/recommend")
def recommend_compat(
    user_id: int,
    n: int = Query(10, ge=1, le=50),
    exclude_seen: bool = True,
):
    """Backward-compatible alias for the original Streamlit UI."""
    return get_service().recommend(user_id=user_id, k=n, exclude_seen=exclude_seen)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
