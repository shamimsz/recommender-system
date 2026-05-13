import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
df = pd.read_csv(PROJECT_ROOT / "data" / "cleaned" / "cleaned_ratings_with_time.csv")
df['timestamp_datetime'] = pd.to_datetime(df['timestamp_datetime'])

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MovieRecommender_Baselines")

def single_split_baseline(df, split_method):
    run_name = f"Popularity_{split_method.capitalize()}"
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", "global_mean")
        mlflow.log_param("split_method", split_method)
        mlflow.log_param("train_ratio", 0.8)
        
        if split_method == "random":
            train, test = train_test_split(df, test_size=0.2, random_state=42)
        elif split_method == "temporal":
            df_sorted = df.sort_values('timestamp_datetime')
            split_idx = int(0.8 * len(df_sorted))
            train = df_sorted.iloc[:split_idx]
            test = df_sorted.iloc[split_idx:]
        else:
            raise ValueError(f"Unknown split_method: {split_method}")
        
        global_mean = train['rating'].mean()
        predictions = [global_mean] * len(test)
        actual = test['rating'].values
        
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mae = mean_absolute_error(actual, predictions)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        
        print(f"{split_method.capitalize()} Split - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        return rmse, mae

def cross_validate_baseline(df, n_splits=3):
    df_sorted = df.sort_values('timestamp_datetime').reset_index(drop=True)
    rmse_scores = []
    mae_scores = []
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    print(f"\n{n_splits}-fold Time Series Cross-Validation:")
    
    with mlflow.start_run(run_name=f"Popularity_CrossValidation_{n_splits}folds"):
        mlflow.log_param("model_type", "global_mean")
        mlflow.log_param("split_method", "time_series_cv")
        mlflow.log_param("n_splits", n_splits)
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(df_sorted)):
            train = df_sorted.iloc[train_idx]
            test = df_sorted.iloc[test_idx]
            
            global_mean = train['rating'].mean()
            predictions = [global_mean] * len(test)
            actual = test['rating'].values
            
            rmse = np.sqrt(mean_squared_error(actual, predictions))
            mae = mean_absolute_error(actual, predictions)
            
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            
            mlflow.log_metric(f"fold_{fold+1}_rmse", rmse)
            mlflow.log_metric(f"fold_{fold+1}_mae", mae)
            
            print(f"  Fold {fold+1}: RMSE = {rmse:.4f}, MAE = {mae:.4f}")
        
        rmse_mean = np.mean(rmse_scores)
        rmse_std = np.std(rmse_scores)
        mae_mean = np.mean(mae_scores)
        mae_std = np.std(mae_scores)
        
        mlflow.log_metric("rmse_mean", rmse_mean)
        mlflow.log_metric("rmse_std", rmse_std)
        mlflow.log_metric("mae_mean", mae_mean)
        mlflow.log_metric("mae_std", mae_std)
        
        print(f"\nSummary: RMSE = {rmse_mean:.4f} ± {rmse_std:.4f}, MAE = {mae_mean:.4f} ± {mae_std:.4f}")
        
        return {
            'rmse_mean': rmse_mean,
            'rmse_std': rmse_std,
            'mae_mean': mae_mean,
            'mae_std': mae_std,
            'rmse_scores': rmse_scores,
            'mae_scores': mae_scores
        }

if __name__ == "__main__":
    print("=" * 50)
    print("Popularity Baseline Evaluation")
    print("=" * 50)
    
    single_split_baseline(df, "random")
    single_split_baseline(df, "temporal")
    cross_validate_baseline(df, n_splits=3)
    
    print("\nDone. Check MLflow UI at http://localhost:5000")