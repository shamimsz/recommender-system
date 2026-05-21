from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "cleaned" / "cleaned_ratings_with_time.csv"
RESULTS_PATH = PROJECT_ROOT / "reports" / "model_comparison_summary.csv"
API_BASE_URL = os.getenv("RECOMMENDER_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Movie Recommender", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def load_ratings() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    df["timestamp_datetime"] = pd.to_datetime(df["timestamp_datetime"])
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_research_results() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        return pd.DataFrame()
    results = pd.read_csv(RESULTS_PATH)
    label_map = {
        "popularity_baseline": "Popularity",
        "knn_with_means": "KNNWithMeans",
        "svd": "SVD",
    }
    results["model"] = results["model_name"].map(label_map).fillna(results["model_name"])
    return results


@st.cache_data(ttl=30, show_spinner=False)
def fetch_health(api_base_url: str) -> dict:
    response = requests.get(f"{api_base_url}/health", timeout=5)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_recommendations(api_base_url: str, user_id: int, k: int, exclude_seen: bool) -> dict:
    params = {"user_id": user_id, "k": k, "exclude_seen": exclude_seen}
    response = requests.get(f"{api_base_url}/recommendations", params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def format_metric(value: float | int | None, decimals: int = 4) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{decimals}f}"


def model_status_label(model_name: str, fallback_used: bool) -> str:
    if fallback_used:
        return "Cold-start fallback: popularity ranking"
    if model_name == "svd":
        return "Personalized SVD recommendations"
    return model_name.replace("_", " ").title()


def render_research_panel(results: pd.DataFrame) -> None:
    st.subheader("Research Results")
    if results.empty:
        st.info("Run `make evaluate` to generate the research results table.")
        return

    rmse_winner = results.loc[results["rmse_mean"].idxmin()]
    precision_winner = results.loc[results["precision_at_10_mean"].idxmax()]
    coverage_winner = results.loc[results["coverage_at_10_mean"].idxmax()]

    col1, col2, col3 = st.columns(3)
    col1.metric("Best RMSE", rmse_winner["model"], format_metric(rmse_winner["rmse_mean"]))
    col2.metric("Best Precision@10", precision_winner["model"], format_metric(precision_winner["precision_at_10_mean"]))
    col3.metric("Best Coverage@10", coverage_winner["model"], format_metric(coverage_winner["coverage_at_10_mean"]))

    st.caption(
        "Main finding: SVD predicts ratings best, while popularity is strongest for offline top-k ranking."
    )

    table = results[
        [
            "model",
            "rmse_mean",
            "mae_mean",
            "precision_at_10_mean",
            "recall_at_10_mean",
            "ndcg_at_10_mean",
            "coverage_at_10_mean",
        ]
    ].rename(
        columns={
            "model": "Model",
            "rmse_mean": "RMSE",
            "mae_mean": "MAE",
            "precision_at_10_mean": "Precision@10",
            "recall_at_10_mean": "Recall@10",
            "ndcg_at_10_mean": "nDCG@10",
            "coverage_at_10_mean": "Coverage@10",
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)


def render_user_profile(ratings: pd.DataFrame, user_id: int) -> None:
    st.subheader("User Profile")
    if ratings.empty:
        st.info("Run `make data` to enable local user history.")
        return

    user_history = ratings[ratings["user_id"] == user_id].sort_values(
        ["rating", "timestamp_datetime"], ascending=[False, False]
    )
    if user_history.empty:
        st.info("No local rating history for this user. The API will use the cold-start fallback.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Ratings", len(user_history))
    col2.metric("Average", format_metric(user_history["rating"].mean(), decimals=2))
    col3.metric("Latest", user_history["timestamp_datetime"].max().date().isoformat())

    history = user_history[["title", "rating", "timestamp_datetime"]].head(8).copy()
    history["timestamp_datetime"] = history["timestamp_datetime"].dt.date.astype(str)
    history = history.rename(
        columns={
            "title": "Movie",
            "rating": "Rating",
            "timestamp_datetime": "Rated On",
        }
    )
    st.dataframe(history, use_container_width=True, hide_index=True)


def render_recommendations(data: dict) -> None:
    recommendations = data.get("recommendations", [])
    model_name = data.get("model_name", "unknown")
    fallback_used = bool(data.get("fallback_used", False))

    status_text = model_status_label(model_name, fallback_used)
    if fallback_used:
        st.warning(status_text)
    else:
        st.success(status_text)

    meta1, meta2, meta3 = st.columns(3)
    meta1.metric("Returned", len(recommendations))
    meta2.metric("Latency", f"{data.get('latency_ms', 0):.2f} ms")
    meta3.metric("Exclude Seen", "Yes" if data.get("exclude_seen", True) else "No")

    if not recommendations:
        st.info("No recommendations returned. Try allowing seen movies or choosing another user.")
        return

    recs = pd.DataFrame(recommendations)
    recs["score"] = recs["score"].map(lambda value: f"{float(value):.3f}")
    recs["reason"] = recs["reason"].str.replace("_", " ").str.title()
    recs = recs[["rank", "title", "score", "reason"]].rename(
        columns={
            "rank": "Rank",
            "title": "Movie",
            "score": "Score",
            "reason": "Reason",
        }
    )
    st.dataframe(recs, use_container_width=True, hide_index=True)


ratings = load_ratings()
results = load_research_results()

with st.sidebar:
    st.title("Movie Recommender")
    api_base_url = st.text_input("API URL", value=API_BASE_URL)
    user_id = st.number_input(
        "User ID",
        min_value=1,
        max_value=9999,
        value=196,
        step=1,
        help="MovieLens 100k known users are 1 to 943; higher IDs trigger fallback.",
    )
    k = st.slider("Recommendations", min_value=1, max_value=20, value=10)
    exclude_seen = st.checkbox("Hide already-rated movies", value=True)
    recommend_button = st.button("Recommend", type="primary", use_container_width=True)

st.title("Movie Recommender Research Interface")
st.caption("Temporal evaluation, SVD serving, popularity fallback, and user-history inspection.")

try:
    health = fetch_health(api_base_url)
    if health.get("status") == "healthy":
        st.caption(
            f"API healthy | {health.get('model_name')} {health.get('model_version')} | "
            f"{health.get('users')} users | {health.get('movies')} movies"
        )
    else:
        st.warning(f"API unhealthy: {health.get('detail', 'unknown error')}")
except requests.exceptions.ConnectionError:
    st.error("API is not reachable. Start it with `make api` and keep this app open.")
except requests.exceptions.Timeout:
    st.error("API health check timed out. The server may still be starting.")
except requests.exceptions.HTTPError as exc:
    st.error(f"API health check failed with status {exc.response.status_code}.")
except Exception as exc:
    st.error(f"API health check failed: {exc}")

left, right = st.columns([1.4, 1.0], gap="large")

with left:
    st.subheader("Recommendations")
    if recommend_button:
        try:
            data = fetch_recommendations(api_base_url, user_id, k, exclude_seen)
            render_recommendations(data)
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the API. Run `make api`, then try again.")
        except requests.exceptions.Timeout:
            st.error("The API request timed out. Try a smaller recommendation count or restart the API.")
        except requests.exceptions.HTTPError as exc:
            detail = ""
            try:
                detail = exc.response.json().get("detail", "")
            except Exception:
                detail = exc.response.text
            st.error(f"API error {exc.response.status_code}: {detail}")
        except Exception as exc:
            st.error(f"Recommendation request failed: {exc}")
    else:
        st.info("Choose a user and press Recommend.")

    render_user_profile(ratings, user_id)

with right:
    render_research_panel(results)
