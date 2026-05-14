import pandas as pd
import numpy as np
import mlflow
import pickle
from pathlib import Path
from surprise import Dataset, Reader, SVD
from surprise import accuracy
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).parent.parent
df = pd.read_csv(PROJECT_ROOT / "data" / "cleaned" / "cleaned_ratings_with_time.csv")
df['timestamp_datetime'] = pd.to_datetime(df['timestamp_datetime'])

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MovieRecommender_Models")

def get_best_n_factors():
    client = MlflowClient()
    
    try:
        experiment = client.get_experiment_by_name("MovieRecommender_Models")
        if experiment is None:
            print("Warning: No experiment found. Using default n_factors=100.")
            return 100
        
        runs = client.search_runs(
            experiment.experiment_id,
            filter_string="params.experiment_type = 'hyperparameter_tuning'",
            order_by=["metrics.rmse_mean ASC"]
        )
        
        if not runs:
            print("Warning: No tuning runs found. Using default n_factors=100.")
            return 100
        
        best_run = runs[0]
        best_n_factors = int(best_run.data.params.get("n_factors", 100))
        best_rmse = best_run.data.metrics.get("rmse_mean")
        
        print(f"Auto-detected best n_factors = {best_n_factors} (RMSE = {best_rmse:.4f})")
        return best_n_factors
    
    except Exception as e:
        print(f"Warning: Could not fetch best params: {e}")
        return 100

def get_baseline_cv_results():
    client = MlflowClient()
    
    try:
        experiment = client.get_experiment_by_name("MovieRecommender_Baselines")
        if experiment is None:
            print("Warning: Baseline experiment not found. Using default values.")
            return {'rmse_mean': 1.1268, 'rmse_std': 0.0312}
        
        runs = client.search_runs(
            experiment.experiment_id,
            order_by=["start_time desc"]
        )
        
        for run in runs:
            params = run.data.params
            if params.get("split_method") == "time_series_cv":
                rmse_mean = run.data.metrics.get("rmse_mean")
                rmse_std = run.data.metrics.get("rmse_std")
                if rmse_mean and rmse_std:
                    print(f"Loaded baseline CV results: RMSE = {rmse_mean:.4f} ± {rmse_std:.4f}")
                    return {'rmse_mean': rmse_mean, 'rmse_std': rmse_std}
        
        print("Warning: No baseline CV found. Using defaults.")
        return {'rmse_mean': 1.1268, 'rmse_std': 0.0312}
    
    except Exception as e:
        print(f"Warning: Could not fetch baseline: {e}")
        return {'rmse_mean': 1.1268, 'rmse_std': 0.0312}

def time_series_cv_svd(df, n_factors, n_epochs=20, lr=0.005, reg=0.02, n_splits=3):
    df_sorted = df.sort_values('timestamp_datetime').reset_index(drop=True)
    total_rows = len(df_sorted)
    test_size = total_rows // n_splits
    
    rmse_scores = []
    mae_scores = []
    
    for fold in range(n_splits):
        test_start = (fold + 1) * test_size
        test_end = test_start + test_size
        
        if fold == n_splits - 1:
            test_end = total_rows
        
        train = df_sorted.iloc[:test_start]
        test = df_sorted.iloc[test_start:test_end]
        
        reader = Reader(rating_scale=(1, 5))
        trainset = Dataset.load_from_df(train[['user_id', 'item_id', 'rating']], reader).build_full_trainset()
        testset = [(row['user_id'], row['item_id'], row['rating']) for _, row in test.iterrows()]
        
        algo = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr, reg_all=reg, random_state=42)
        algo.fit(trainset)
        
        predictions = algo.test(testset)
        
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        
        rmse_scores.append(rmse)
        mae_scores.append(mae)
    
    return {
        'rmse_mean': np.mean(rmse_scores),
        'rmse_std': np.std(rmse_scores),
        'mae_mean': np.mean(mae_scores),
        'mae_std': np.std(mae_scores)
    }

def train_svd_cv_with_mlflow(df, n_factors, n_epochs=20, lr=0.005, reg=0.02, n_splits=3):
    run_name = f"SVD_Final_{n_factors}factors"
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("n_factors", n_factors)
        mlflow.log_param("n_epochs", n_epochs)
        mlflow.log_param("lr_all", lr)
        mlflow.log_param("reg_all", reg)
        mlflow.log_param("n_splits", n_splits)
        mlflow.log_param("evaluation_type", "time_series_cv")
        mlflow.log_param("experiment_type", "final_model")
        
        results = time_series_cv_svd(df, n_factors, n_epochs, lr, reg, n_splits)
        
        mlflow.log_metric("rmse_mean", results['rmse_mean'])
        mlflow.log_metric("rmse_std", results['rmse_std'])
        mlflow.log_metric("mae_mean", results['mae_mean'])
        mlflow.log_metric("mae_std", results['mae_std'])
        
        print(f"SVD (n_factors={n_factors}) - RMSE: {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}")
        return results

def compare_with_baseline_cv(svd_results):
    baseline = get_baseline_cv_results()
    improvement = (baseline['rmse_mean'] - svd_results['rmse_mean']) / baseline['rmse_mean'] * 100
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (Time Series Cross-Validation)")
    print("=" * 60)
    print(f"{'Model':<20} {'RMSE (mean ± std)':<25} {'Improvement':<15}")
    print("-" * 60)
    print(f"{'Popularity Baseline':<20} {baseline['rmse_mean']:.4f} ± {baseline['rmse_std']:.4f} {'---':<15}")
    print(f"{'SVD':<20} {svd_results['rmse_mean']:.4f} ± {svd_results['rmse_std']:.4f} {improvement:>+.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    print("=" * 60)
    print("SVD WITH TIME SERIES CROSS-VALIDATION")
    print("=" * 60)
    
    best_n_factors = get_best_n_factors()
    
    svd_results = train_svd_cv_with_mlflow(
        df, 
        n_factors=best_n_factors,
        n_epochs=20, 
        lr=0.005, 
        reg=0.02, 
        n_splits=3
    )
    
    compare_with_baseline_cv(svd_results)
    
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    
    print("\nTraining final production model on all data...")
    reader = Reader(rating_scale=(1, 5))
    full_trainset = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader).build_full_trainset()
    
    final_model = SVD(n_factors=best_n_factors, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    final_model.fit(full_trainset)
    
    with open(models_dir / "svd_model.pkl", "wb") as f:
        pickle.dump(final_model, f)
    
    print(f"Final model saved to {models_dir / 'svd_model.pkl'} with n_factors={best_n_factors}")
    print("Done. Check MLflow UI at http://localhost:5000")