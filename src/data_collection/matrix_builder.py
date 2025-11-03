"""
Stage 3: Build User-Restaurant Preference Matrix
Creates a matrix where rows = reviewers, columns = restaurants, values = ratings
Used for collaborative filtering recommendations
"""

import pandas as pd
import numpy as np
from .config import PATHS


def build_user_restaurant_matrix():
    """Build user-restaurant preference matrix from Stage 2 reviews"""

    print("\n" + "="*70)
    print("STAGE 3: BUILDING USER-RESTAURANT PREFERENCE MATRIX")
    print("="*70)

    # Load reviews
    try:
        reviews_df = pd.read_csv(PATHS['stage2_reviews'])
        print(f"\nLoaded {len(reviews_df)} reviews from Stage 2")
    except FileNotFoundError:
        print("✗ Stage 2 reviews file not found")
        return None

    # Basic stats
    unique_reviewers = reviews_df['reviewer_url'].nunique()
    unique_restaurants = reviews_df['restaurant_id'].nunique()

    print(f"Unique reviewers: {unique_reviewers}")
    print(f"Unique restaurants: {unique_restaurants}")
    print(f"Data sparsity: {(1 - len(reviews_df) / (unique_reviewers * unique_restaurants)) * 100:.1f}%")

    # Create pivot table: rows = reviewers, columns = restaurants, values = ratings
    print("\nBuilding pivot table...")
    matrix = reviews_df.pivot_table(
        index='reviewer_url',
        columns='restaurant_id',
        values='rating',
        aggfunc='mean'  # In case of duplicate reviews
    )

    print(f"\nMatrix shape: {matrix.shape}")
    print(f"  Rows (reviewers): {matrix.shape[0]}")
    print(f"  Columns (restaurants): {matrix.shape[1]}")

    # Save matrices
    output_dir = PATHS['data_raw']

    # Full sparse matrix
    matrix_file = f"{output_dir}/stage3_user_restaurant_matrix.csv"
    matrix.to_csv(matrix_file)
    print(f"\n✓ Full matrix saved to: {matrix_file}")

    # Dense matrix (only for reviewers with >= 5 reviews)
    reviewers_with_data = (matrix.notna().sum(axis=1) >= 5)
    dense_matrix = matrix[reviewers_with_data]

    dense_file = f"{output_dir}/stage3_user_restaurant_matrix_dense.csv"
    dense_matrix.to_csv(dense_file)
    print(f"✓ Dense matrix (reviewers with 5+ reviews) saved to: {dense_file}")
    print(f"  Size: {dense_matrix.shape[0]} reviewers × {dense_matrix.shape[1]} restaurants")

    # Create reviewers profile
    print("\nBuilding reviewer profiles...")
    reviewer_profiles = []

    for reviewer_url in matrix.index:
        reviews = reviews_df[reviews_df['reviewer_url'] == reviewer_url]
        profile = {
            'reviewer_url': reviewer_url,
            'reviewer_name': reviews.iloc[0]['reviewer_name'],
            'num_reviews': len(reviews),
            'avg_rating': reviews['rating'].mean(),
            'restaurants_rated': reviews['restaurant_id'].nunique(),
            'date_range': f"{reviews['review_date'].min()} to {reviews['review_date'].max()}",
        }
        reviewer_profiles.append(profile)

    profiles_df = pd.DataFrame(reviewer_profiles).sort_values('num_reviews', ascending=False)
    profiles_file = f"{output_dir}/stage3_reviewer_profiles.csv"
    profiles_df.to_csv(profiles_file, index=False)
    print(f"✓ Reviewer profiles saved to: {profiles_file}")

    # Top reviewers
    print("\nTop 10 reviewers by review count:")
    for idx, row in profiles_df.head(10).iterrows():
        print(f"  {idx+1:2d}. {row['reviewer_name']:25s} | Reviews: {row['num_reviews']:3.0f} | "
              f"Avg Rating: {row['avg_rating']:.2f}⭐ | Restaurants: {row['restaurants_rated']:.0f}")

    # Create restaurant profile
    print("\nBuilding restaurant profiles...")
    restaurant_profiles = []

    for restaurant_id in matrix.columns:
        rest_reviews = reviews_df[reviews_df['restaurant_id'] == restaurant_id]
        profile = {
            'restaurant_id': restaurant_id,
            'restaurant_name': rest_reviews.iloc[0]['restaurant_name'],
            'num_reviews_in_matrix': (matrix[restaurant_id].notna()).sum(),
            'avg_rating_in_matrix': rest_reviews['rating'].mean(),
            'total_reviewers': len(rest_reviews),
        }
        restaurant_profiles.append(profile)

    rest_profiles_df = pd.DataFrame(restaurant_profiles).sort_values('num_reviews_in_matrix', ascending=False)
    rest_profiles_file = f"{output_dir}/stage3_restaurant_profiles.csv"
    rest_profiles_df.to_csv(rest_profiles_file, index=False)
    print(f"✓ Restaurant profiles saved to: {rest_profiles_file}")

    # Top restaurants
    print("\nTop 10 restaurants by review count (in matrix):")
    for idx, row in rest_profiles_df.head(10).iterrows():
        print(f"  {idx+1:2d}. {row['restaurant_name']:40s} | Reviews: {row['num_reviews_in_matrix']:3.0f} | "
              f"Avg Rating: {row['avg_rating_in_matrix']:.2f}⭐")

    return matrix, dense_matrix, profiles_df, rest_profiles_df


def analyze_matrix_quality():
    """Analyze the quality of the preference matrix"""

    print("\n" + "="*70)
    print("MATRIX QUALITY ANALYSIS")
    print("="*70)

    matrix_file = f"{PATHS['data_raw']}/stage3_user_restaurant_matrix.csv"
    matrix = pd.read_csv(matrix_file, index_col=0)

    # Sparsity analysis
    total_cells = matrix.shape[0] * matrix.shape[1]
    filled_cells = matrix.notna().sum().sum()
    sparsity = 1 - (filled_cells / total_cells)

    print(f"\nMatrix statistics:")
    print(f"  Total cells: {total_cells:,}")
    print(f"  Filled cells: {filled_cells:,}")
    print(f"  Sparsity: {sparsity*100:.2f}%")

    # Distribution of reviews
    print(f"\nReviews per reviewer:")
    reviews_per_user = matrix.notna().sum(axis=1)
    print(f"  Mean: {reviews_per_user.mean():.1f}")
    print(f"  Median: {reviews_per_user.median():.1f}")
    print(f"  Min: {reviews_per_user.min():.0f}")
    print(f"  Max: {reviews_per_user.max():.0f}")

    print(f"\nReviews per restaurant:")
    reviews_per_rest = matrix.notna().sum(axis=0)
    print(f"  Mean: {reviews_per_rest.mean():.1f}")
    print(f"  Median: {reviews_per_rest.median():.1f}")
    print(f"  Min: {reviews_per_rest.min():.0f}")
    print(f"  Max: {reviews_per_rest.max():.0f}")

    # Rating distribution
    all_ratings = matrix.values.flatten()
    all_ratings = all_ratings[~np.isnan(all_ratings)]
    print(f"\nRating distribution:")
    for rating in sorted(all_ratings[~np.isnan(all_ratings)].astype(int)):
        count = (all_ratings == rating).sum()
        print(f"  {rating}⭐: {count:,} ({count/len(all_ratings)*100:.1f}%)")


if __name__ == "__main__":
    matrix, dense_matrix, profiles, rest_profiles = build_user_restaurant_matrix()
    if matrix is not None:
        analyze_matrix_quality()
