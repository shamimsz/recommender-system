import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ml-100k"
OUTPUT_DIR = PROJECT_ROOT / "data" / "cleaned"

def load_ratings():
    df = pd.read_csv(
        DATA_DIR / "u.data",
        sep='\t',
        names=['user_id', 'item_id', 'rating', 'timestamp_unix']
    )
    
    df['timestamp_datetime'] = pd.to_datetime(df['timestamp_unix'], unit='s')
    
    df['rating_year'] = df['timestamp_datetime'].dt.year
    df['rating_month'] = df['timestamp_datetime'].dt.month
    df['rating_dayofweek'] = df['timestamp_datetime'].dt.dayofweek
    
    return df

def load_movies():
    df = pd.read_csv(
        DATA_DIR / "u.item",
        sep='|',
        encoding='ISO-8859-1',
        names=['item_id', 'title'],
        usecols=[0, 1]
    )
    return df

def prepare_data():
    ratings = load_ratings()
    movies = load_movies()
    
    merged = pd.merge(ratings, movies, on='item_id', how='inner')
    
    final = merged[[
        'user_id',
        'item_id',
        'rating',
        'timestamp_unix',
        'timestamp_datetime',
        'rating_year',
        'rating_month',
        'rating_dayofweek',
        'title'
    ]]
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    output_path = OUTPUT_DIR / "cleaned_ratings_with_time.csv"
    final.to_csv(output_path, index=False)
    
    return final

if __name__ == "__main__":
    print("Loading ratings with timestamp conversion...")
    ratings = load_ratings()
    print(f"Ratings shape: {ratings.shape}")
    print(f"Columns: {ratings.columns.tolist()}")
    print(ratings.head())
    print()
    
    print("Loading movies...")
    movies = load_movies()
    print(f"Movies shape: {movies.shape}")
    print(movies.head())
    print()
    
    print("Preparing final dataset...")
    final = prepare_data()
    print(f"Final shape: {final.shape}")
    print(f"Columns: {final.columns.tolist()}")
    print(final.head())
    print()
    
    print(f"Done! Saved to {OUTPUT_DIR / 'cleaned_ratings_with_time.csv'}")
    