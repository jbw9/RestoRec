"""
Filter Stage 1 restaurants by quality metrics (review count, rating)
Keep only restaurants with sufficient datapoints for meaningful analysis
"""

import pandas as pd
from src.data_collection.config import PATHS

# Configuration for filtering
MIN_REVIEWS = 10  # Minimum number of reviews
MIN_RATING = None  # No minimum rating threshold


def filter_restaurants_by_quality():
    """Filter restaurants to only high-quality ones with sufficient data"""

    print("\n" + "="*70)
    print("STAGE 1: FILTERING RESTAURANTS BY QUALITY")
    print("="*70)

    # Read all restaurants
    df = pd.read_csv(PATHS['stage1_restaurants'])
    print(f"\nLoaded {len(df)} total restaurants from Stage 1")

    # Convert reviews_count to numeric (handle commas in numbers)
    df['reviews_count'] = pd.to_numeric(df['reviews_count'].astype(str).str.replace(',', ''), errors='coerce')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    print(f"\nReview count range: {df['reviews_count'].min():.0f} - {df['reviews_count'].max():.0f}")
    print(f"Rating range: {df['rating'].min():.1f} - {df['rating'].max():.1f}")

    # Apply filters
    print(f"\n" + "="*70)
    print(f"FILTERING CRITERIA:")
    print(f"  - Minimum reviews: {MIN_REVIEWS}")
    if MIN_RATING is not None:
        print(f"  - Minimum rating: {MIN_RATING}")
    else:
        print(f"  - Minimum rating: None (no limit)")
    print("="*70)

    # Filter
    if MIN_RATING is not None:
        filtered_df = df[(df['reviews_count'] >= MIN_REVIEWS) & (df['rating'] >= MIN_RATING)].copy()
    else:
        filtered_df = df[df['reviews_count'] >= MIN_REVIEWS].copy()

    print(f"\nRestaurants passing filters: {len(filtered_df)}/{len(df)} ({len(filtered_df)/len(df)*100:.1f}%)")

    # Sort by review count descending (quality metric)
    filtered_df = filtered_df.sort_values('reviews_count', ascending=False).reset_index(drop=True)

    # Save filtered dataset
    output_file = PATHS['stage1_restaurants'].replace('.csv', '_filtered.csv')
    filtered_df.to_csv(output_file, index=False)

    print(f"\n✓ Filtered data saved to: {output_file}")

    # Statistics
    print("\n" + "="*70)
    print("QUALITY STATISTICS")
    print("="*70)
    print(f"Total restaurants: {len(filtered_df)}")
    print(f"Average reviews per restaurant: {filtered_df['reviews_count'].mean():.0f}")
    print(f"Average rating: {filtered_df['rating'].mean():.2f}")
    print(f"Median reviews: {filtered_df['reviews_count'].median():.0f}")
    print(f"Median rating: {filtered_df['rating'].median():.2f}")

    # Top restaurants
    print("\n" + "="*70)
    print("TOP 20 RESTAURANTS BY REVIEW COUNT")
    print("="*70)
    for idx, row in filtered_df.head(20).iterrows():
        print(f"{idx+1:3d}. {row['name']:40s} | Reviews: {row['reviews_count']:4.0f} | Rating: {row['rating']:.1f}⭐")

    return output_file, filtered_df


if __name__ == "__main__":
    output_file, filtered_df = filter_restaurants_by_quality()
