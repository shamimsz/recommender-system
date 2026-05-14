import pandas as pd
import numpy as np
import mlflow
from pathlib import Path
from surprise import Dataset, Reader, SVD
from surprise import accuracy

PROJECT_ROOT = Path(__file__).parent.parent
df = pd.read_csv(PROJECT_ROOT / "data" / "cleaned" / "cleaned_ratings_with_time.csv")
df['timestamp_datetime'] = pd.to_datetime(df['timestamp_datetime'])

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MovieRecommender_Models")

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

def tune_n_factors(df, n_factors_list=[20, 50, 100, 150]):
    print("=" * 60)
    print("HYPERPARAMETER TUNING: n_factors")
    print("=" * 60)
    
    results = []
    
    for n_factors in n_factors_list:
        print(f"\nTraining with n_factors={n_factors}...")
        
        with mlflow.start_run(run_name=f"SVD_Tune_nfactors_{n_factors}"):
            mlflow.log_param("n_factors", n_factors)
            mlflow.log_param("n_epochs", 20)
            mlflow.log_param("lr_all", 0.005)
            mlflow.log_param("reg_all", 0.02)
            mlflow.log_param("n_splits", 3)
            mlflow.log_param("experiment_type", "hyperparameter_tuning")
            
            cv_results = time_series_cv_svd(df, n_factors=n_factors)
            
            mlflow.log_metric("rmse_mean", cv_results['rmse_mean'])
            mlflow.log_metric("rmse_std", cv_results['rmse_std'])
            mlflow.log_metric("mae_mean", cv_results['mae_mean'])
            mlflow.log_metric("mae_std", cv_results['mae_std'])
            
            results.append({
                'n_factors': n_factors,
                'rmse_mean': cv_results['rmse_mean'],
                'rmse_std': cv_results['rmse_std']
            })
            
            print(f"  RMSE: {cv_results['rmse_mean']:.4f} ± {cv_results['rmse_std']:.4f}")
    
    print("\n" + "=" * 60)
    print("TUNING RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'n_factors':<12} {'RMSE (mean ± std)':<25}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['n_factors']:<12} {r['rmse_mean']:.4f} ± {r['rmse_std']:.4f}")
    
    best = min(results, key=lambda x: x['rmse_mean'])
    print("-" * 60)
    print(f"\n✅ Best: n_factors = {best['n_factors']} with RMSE = {best['rmse_mean']:.4f} ± {best['rmse_std']:.4f}")
    
    return results

if __name__ == "__main__":
    tune_n_factors(df, n_factors_list=[20, 50, 100, 150])