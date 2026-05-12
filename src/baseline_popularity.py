import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

# Load data
PROJECT_ROOT = Path(__file__).parent.parent
df = pd.read_csv(PROJECT_ROOT / "data" / "cleaned" / "cleaned_ratings_with_time.csv")
df['timestamp_datetime'] = pd.to_datetime(df['timestamp_datetime'])

# Set MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MovieRecommender_Baselines")

def evaluate_with_mlflow(df, split_method):
    """Evaluate baseline and log to MLflow."""
    # Use a single run name that includes the split method
    run_name = f"Popularity_{split_method.capitalize()}"
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("model_type", "global_mean")
        mlflow.log_param("split_method", split_method)
        mlflow.log_param("train_ratio", 0.8)
        
        # Split data
        if split_method == "random":
            train, test = train_test_split(df, test_size=0.2, random_state=42)
        elif split_method == "temporal":
            df_sorted = df.sort_values('timestamp_datetime')
            split_idx = int(0.8 * len(df_sorted))
            train = df_sorted.iloc[:split_idx]
            test = df_sorted.iloc[split_idx:]
        else:
            raise ValueError(f"Unknown split_method: {split_method}")
        
        # Calculate baseline
        global_mean = train['rating'].mean()
        predictions = [global_mean] * len(test)
        actual = test['rating'].values
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mae = mean_absolute_error(actual, predictions)
        
        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        
        # Log training set mean as a parameter
        mlflow.log_param("global_mean_train", global_mean)
        
        # Log first 100 predictions as artifact
        sample_df = test.head(100).copy()
        sample_df['predicted_rating'] = predictions[:100]
        sample_df = sample_df[['user_id', 'item_id', 'rating', 'predicted_rating']]
        sample_path = f"predictions_sample_{split_method}.csv"
        sample_df.to_csv(sample_path, index=False)
        mlflow.log_artifact(sample_path)
        
        print(f"{split_method.capitalize()} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        return rmse, mae

if __name__ == "__main__":
    print("=" * 50)
    print("Popularity Baseline Evaluation with MLflow")
    print("=" * 50)
    print()
    
    # Run evaluations with MLflow
    print("Running Random Split...")
    rmse_random, mae_random = evaluate_with_mlflow(df, "random")
    
    print("Running Time-Based Split...")
    rmse_temporal, mae_temporal = evaluate_with_mlflow(df, "temporal")
    
    # Calculate differences
    rmse_diff = rmse_temporal - rmse_random
    mae_diff = mae_temporal - mae_random
    
    # Output results
    print()
    print("-" * 50)
    print("RESULTS")
    print("-" * 50)
    print()
    
    print("Random Split:")
    print(f"  RMSE: {rmse_random:.4f}")
    print(f"  MAE:  {mae_random:.4f}")
    print()
    
    print("Time-Based Split:")
    print(f"  RMSE: {rmse_temporal:.4f}")
    print(f"  MAE:  {mae_temporal:.4f}")
    print()
    
    print("Difference (Time - Random):")
    print(f"  RMSE: {rmse_diff:+.4f} {'(worse)' if rmse_diff > 0 else '(better)'}")
    print(f"  MAE:  {mae_diff:+.4f} {'(worse)' if mae_diff > 0 else '(better)'}")
    print()
    
    # Additional information
    print("-" * 50)
    print("DATA INFO")
    print("-" * 50)
    print(f"Total ratings: {len(df):,}")
    print(f"Global mean rating: {df['rating'].mean():.4f}")
    print()
    
    print(f"Random split - Train size: {int(0.8 * len(df)):,}, Test size: {int(0.2 * len(df)):,}")
    
    # For temporal split, show date range
    df_sorted = df.sort_values('timestamp_datetime')
    split_idx = int(0.8 * len(df_sorted))
    train_end = df_sorted.iloc[split_idx - 1]['timestamp_datetime']
    test_start = df_sorted.iloc[split_idx]['timestamp_datetime']
    
    print(f"Time-based split - Train: up to {train_end.date()}, Test: from {test_start.date()}")
    print()
    
    print("=" * 50)
    print("MLflow Logging Complete!")
    print("=" * 50)
    print(f"View results at: http://localhost:5000")
    print("Experiment: MovieRecommender_Baselines")
    print("Runs: Popularity_Random and Popularity_Temporal")
    print()
    print("Interpretation:")
    print("If time-based RMSE > random RMSE, ratings have temporal drift.")
    print("Time-based split is more realistic for production systems.")