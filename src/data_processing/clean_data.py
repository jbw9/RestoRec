"""
Phase 2.1: Data Cleaning
Clean and filter the merged dataset for model training
"""

import pandas as pd
import numpy as np
import os


def clean_reviews_data(input_path='data/raw/stage4_merged_reviews.csv',
                       output_path='data/processed/user_restaurant_ratings.csv',
                       min_reviews_per_user=3):
    """
    Clean the merged reviews dataset

    Steps:
    1. Remove duplicates
    2. Handle missing values
    3. Filter users with insufficient reviews (< min_reviews_per_user)
    4. Encode user and restaurant IDs
    5. Save cleaned dataset

    Args:
        input_path: Path to merged reviews CSV
        output_path: Path to save cleaned data
        min_reviews_per_user: Minimum reviews required per user (default: 3 for CF)
    """

    print("="*70)
    print("DATA CLEANING PIPELINE")
    print("="*70)

    # Load data
    print(f"\n1. Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"   Initial size: {len(df):,} reviews")
    print(f"   Users: {df['user_id'].nunique():,}")
    print(f"   Restaurants: {df['restaurant_id'].nunique():,}")

    # Remove duplicates
    print(f"\n2. Removing duplicates...")
    initial_count = len(df)
    df = df.drop_duplicates(subset=['user_id', 'restaurant_id', 'rating'], keep='first')
    duplicates_removed = initial_count - len(df)
    print(f"   Removed {duplicates_removed:,} duplicate reviews")
    print(f"   Remaining: {len(df):,} reviews")

    # Remove rows with missing ratings
    print(f"\n3. Handling missing values...")
    missing_ratings = df['rating'].isna().sum()
    df = df[df['rating'].notna()]
    print(f"   Removed {missing_ratings:,} reviews with missing ratings")

    # Filter users with insufficient reviews
    print(f"\n4. Filtering users (minimum {min_reviews_per_user} reviews)...")
    user_review_counts = df.groupby('user_id').size()
    valid_users = user_review_counts[user_review_counts >= min_reviews_per_user].index

    users_before = df['user_id'].nunique()
    df = df[df['user_id'].isin(valid_users)]
    users_after = df['user_id'].nunique()
    users_filtered = users_before - users_after

    print(f"   Removed {users_filtered:,} users with < {min_reviews_per_user} reviews")
    print(f"   Remaining users: {users_after:,}")
    print(f"   Remaining reviews: {len(df):,}")

    # Encode user and restaurant IDs to continuous integers
    print(f"\n5. Encoding IDs...")

    # Create user ID mapping
    unique_users = sorted(df['user_id'].unique())
    user_id_to_encoded = {uid: idx for idx, uid in enumerate(unique_users)}
    df['user_id_encoded'] = df['user_id'].map(user_id_to_encoded)

    # Create restaurant ID mapping
    unique_restaurants = sorted(df['restaurant_id'].unique())
    restaurant_id_to_encoded = {rid: idx for idx, rid in enumerate(unique_restaurants)}
    df['restaurant_id_encoded'] = df['restaurant_id'].map(restaurant_id_to_encoded)

    print(f"   Encoded {len(unique_users):,} users (0 to {len(unique_users)-1})")
    print(f"   Encoded {len(unique_restaurants):,} restaurants (0 to {len(unique_restaurants)-1})")

    # Save ID mappings for later use
    print(f"\n6. Saving ID mappings...")
    os.makedirs('data/processed', exist_ok=True)

    user_mapping_df = pd.DataFrame([
        {'user_id_original': uid, 'user_id_encoded': encoded}
        for uid, encoded in user_id_to_encoded.items()
    ])
    user_mapping_df.to_csv('data/processed/user_id_mapping.csv', index=False)

    restaurant_mapping_df = pd.DataFrame([
        {'restaurant_id_original': rid, 'restaurant_id_encoded': encoded}
        for rid, encoded in restaurant_id_to_encoded.items()
    ])
    restaurant_mapping_df.to_csv('data/processed/restaurant_id_mapping.csv', index=False)

    print(f"   Saved user mapping to data/processed/user_id_mapping.csv")
    print(f"   Saved restaurant mapping to data/processed/restaurant_id_mapping.csv")

    # Select relevant columns for ratings matrix
    ratings_df = df[[
        'user_id', 'user_id_encoded',
        'restaurant_id', 'restaurant_id_encoded', 'restaurant_name',
        'rating', 'review_date', 'review_text'
    ]].copy()

    # Save cleaned data
    print(f"\n7. Saving cleaned data...")
    ratings_df.to_csv(output_path, index=False)
    print(f"   Saved to {output_path}")

    # Statistics
    print("\n" + "="*70)
    print("CLEANED DATASET STATISTICS")
    print("="*70)

    print(f"\n📊 Final Dataset:")
    print(f"   Total reviews: {len(ratings_df):,}")
    print(f"   Unique users: {ratings_df['user_id'].nunique():,}")
    print(f"   Unique restaurants: {ratings_df['restaurant_id'].nunique():,}")

    user_counts = ratings_df.groupby('user_id').size()
    print(f"\n👥 Reviews per user:")
    print(f"   Mean: {user_counts.mean():.2f}")
    print(f"   Median: {user_counts.median():.1f}")
    print(f"   Min: {user_counts.min()}")
    print(f"   Max: {user_counts.max()}")

    resto_counts = ratings_df.groupby('restaurant_id').size()
    print(f"\n🍽️  Reviews per restaurant:")
    print(f"   Mean: {resto_counts.mean():.2f}")
    print(f"   Median: {resto_counts.median():.1f}")
    print(f"   Min: {resto_counts.min()}")
    print(f"   Max: {resto_counts.max()}")

    print(f"\n⭐ Rating distribution:")
    print(ratings_df['rating'].value_counts().sort_index())

    # Data sparsity
    total_possible = len(unique_users) * len(unique_restaurants)
    sparsity = (1 - len(ratings_df) / total_possible) * 100
    print(f"\n🎯 Data density:")
    print(f"   Sparsity: {sparsity:.2f}%")
    print(f"   Density: {100-sparsity:.4f}%")

    print("\n" + "="*70)
    print("✓ Data cleaning complete!")
    print("="*70)

    return ratings_df


if __name__ == '__main__':
    clean_reviews_data()
