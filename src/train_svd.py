from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import mlflow
import pandas as pd
from surprise import Dataset, Reader, SVD

from src.evaluation import rating_metrics, temporal_expanding_splits


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "cleaned" / "cleaned_ratings_with_time.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "svd_model.pkl"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "svd_model_metadata.json"
DEFAULT_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
DEFAULT_EXPERIMENT = "MovieRecommender_ProductionTraining"


def load_ratings(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}. Run `python src/etl.py` first.")
    df = pd.read_csv(path)
    df["timestamp_datetime"] = pd.to_datetime(df["timestamp_datetime"])
    return df


def train_svd_model(
    df: pd.DataFrame,
    n_factors: int = 100,
    n_epochs: int = 20,
    lr_all: float = 0.005,
    reg_all: float = 0.02,
    random_state: int = 42,
) -> SVD:
    reader = Reader(rating_scale=(1, 5))
    trainset = Dataset.load_from_df(
        df[["user_id", "item_id", "rating"]],
        reader,
    ).build_full_trainset()
    model = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=lr_all,
        reg_all=reg_all,
        random_state=random_state,
    )
    model.fit(trainset)
    return model


def evaluate_svd_temporal(
    df: pd.DataFrame,
    n_splits: int,
    n_factors: int,
    n_epochs: int,
    lr_all: float,
    reg_all: float,
    random_state: int,
) -> pd.DataFrame:
    rows = []
    for fold in temporal_expanding_splits(df, n_splits=n_splits):
        print(f"Evaluating SVD fold {fold.fold}/{n_splits}...")
        model = train_svd_model(
            fold.train,
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=random_state,
        )
        predictions = [
            model.predict(uid=int(row.user_id), iid=int(row.item_id)).est
            for row in fold.test.itertuples(index=False)
        ]
        metrics = rating_metrics(fold.test["rating"].tolist(), predictions)
        rows.append({"model_name": "svd", "fold": fold.fold, **metrics})
    return pd.DataFrame(rows)


def summarize_fold_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    summary = fold_results.groupby("model_name")[["rmse", "mae"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def save_model(model: SVD, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as model_file:
        pickle.dump(model, model_file)


def save_metadata(metadata: dict, metadata_path: Path) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def log_training_to_mlflow(
    metadata: dict,
    fold_results: pd.DataFrame,
    summary: pd.DataFrame,
    tracking_uri: str,
    experiment_name: str,
) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=metadata["model_version"]):
        for key in [
            "model_name",
            "model_version",
            "n_factors",
            "n_epochs",
            "lr_all",
            "reg_all",
            "random_state",
            "n_splits",
            "training_rows",
            "user_count",
            "movie_count",
        ]:
            mlflow.log_param(key, metadata[key])

        if not fold_results.empty:
            for row in fold_results.itertuples(index=False):
                mlflow.log_metric(f"fold_{row.fold}_rmse", float(row.rmse))
                mlflow.log_metric(f"fold_{row.fold}_mae", float(row.mae))

        if not summary.empty:
            summary_row = summary.iloc[0]
            for metric in ["rmse_mean", "rmse_std", "mae_mean", "mae_std"]:
                mlflow.log_metric(metric, float(summary_row[metric]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the production SVD model artifact.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--n-factors", type=int, default=100)
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--lr-all", type=float, default=0.005)
    parser.add_argument("--reg-all", type=float, default=0.02)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_ratings(args.data_path)

    fold_results = pd.DataFrame()
    summary = pd.DataFrame()
    if not args.no_eval:
        fold_results = evaluate_svd_temporal(
            df=df,
            n_splits=args.n_splits,
            n_factors=args.n_factors,
            n_epochs=args.n_epochs,
            lr_all=args.lr_all,
            reg_all=args.reg_all,
            random_state=args.random_state,
        )
        summary = summarize_fold_results(fold_results)
        print("\nTemporal validation summary")
        print(summary.to_string(index=False))

    print("\nTraining final SVD model on all ratings...")
    model = train_svd_model(
        df,
        n_factors=args.n_factors,
        n_epochs=args.n_epochs,
        lr_all=args.lr_all,
        reg_all=args.reg_all,
        random_state=args.random_state,
    )
    save_model(model, args.model_path)

    model_version = f"svd-n{args.n_factors}-e{args.n_epochs}"
    metadata = {
        "model_name": "svd",
        "model_version": model_version,
        "n_factors": args.n_factors,
        "n_epochs": args.n_epochs,
        "lr_all": args.lr_all,
        "reg_all": args.reg_all,
        "random_state": args.random_state,
        "n_splits": args.n_splits,
        "training_rows": int(len(df)),
        "user_count": int(df["user_id"].nunique()),
        "movie_count": int(df["item_id"].nunique()),
        "data_path": str(args.data_path),
        "model_path": str(args.model_path),
    }
    if not summary.empty:
        metadata.update(
            {
                "rmse_mean": float(summary.iloc[0]["rmse_mean"]),
                "rmse_std": float(summary.iloc[0]["rmse_std"]),
                "mae_mean": float(summary.iloc[0]["mae_mean"]),
                "mae_std": float(summary.iloc[0]["mae_std"]),
            }
        )
    save_metadata(metadata, args.metadata_path)

    if not args.no_mlflow:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
        log_training_to_mlflow(
            metadata=metadata,
            fold_results=fold_results,
            summary=summary,
            tracking_uri=tracking_uri,
            experiment_name=DEFAULT_EXPERIMENT,
        )
        print(f"Logged training run to MLflow experiment: {DEFAULT_EXPERIMENT}")

    print(f"Saved model to {args.model_path}")
    print(f"Saved metadata to {args.metadata_path}")


if __name__ == "__main__":
    main()
