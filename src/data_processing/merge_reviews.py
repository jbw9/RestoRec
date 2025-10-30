"""
Stage 4: Merge and Deduplicate Reviews

Combines reviews from Stage 2 (restaurant pages) and Stage 3 (user profiles),
removes duplicates, and produces final dataset for model training
"""

import sys
import os
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collection.config import PATHS
from data_collection.scraper_utils import create_review_hash


def merge_and_deduplicate():
    """
    Merge restaurant reviews and user profile reviews, removing duplicates

    Returns:
        Path to merged CSV file
    """
    print("\n" + "="*60)
    print("STAGE 4: MERGE & DEDUPLICATE REVIEWS")
    print("="*60 + "\n")

    # Load Stage 2 reviews (restaurant pages)
    stage2_file = PATHS['stage2_reviews']
    if not os.path.exists(stage2_file):
        print(f"✗ Error: Stage 2 file not found: {stage2_file}")
        print("  Please run Stage 2 first")
        return None

    print(f"Loading Stage 2 reviews: {stage2_file}")
    restaurant_reviews = pd.read_csv(stage2_file)
    print(f"✓ Loaded {len(restaurant_reviews)} reviews from restaurant pages\n")

    # Load Stage 3 reviews (user profiles)
    stage3_file = PATHS['stage3_reviews']
    user_profile_reviews = pd.DataFrame()

    if os.path.exists(stage3_file):
        print(f"Loading Stage 3 reviews: {stage3_file}")
        user_profile_reviews = pd.read_csv(stage3_file)
        print(f"✓ Loaded {len(user_profile_reviews)} reviews from user profiles\n")
    else:
        print("⚠ Stage 3 file not found - skipping user profile reviews")
        print("  (This is OK if you only want to use Stage 2 data)\n")

    # Ensure same columns
    if len(user_profile_reviews) > 0:
        common_columns = list(set(restaurant_reviews.columns) & set(user_profile_reviews.columns))
        restaurant_reviews = restaurant_reviews[common_columns]
        user_profile_reviews = user_profile_reviews[common_columns]

        # Concatenate
        all_reviews = pd.concat([restaurant_reviews, user_profile_reviews], ignore_index=True)
    else:
        all_reviews = restaurant_reviews

    print(f"Combined reviews (with potential duplicates): {len(all_reviews)}\n")

    # Create hash for deduplication
    print("Creating review hashes for deduplication...")
    all_reviews['review_hash'] = all_reviews.apply(
        lambda row: create_review_hash(
            row['user_id'],
            row['restaurant_id'],
            row['review_text'] if pd.notna(row['review_text']) else ''
        ),
        axis=1
    )

    # Remove duplicates (keep first occurrence - from restaurant page)
    print("Removing duplicate reviews...")
    before_dedup = len(all_reviews)
    all_reviews_dedup = all_reviews.drop_duplicates(subset=['review_hash'], keep='first')
    after_dedup = len(all_reviews_dedup)

    # Drop the hash column
    all_reviews_dedup = all_reviews_dedup.drop(columns=['review_hash'])

    print(f"✓ Removed {before_dedup - after_dedup} duplicate reviews\n")

    # Calculate statistics
    print("="*60)
    print("FINAL DATASET STATISTICS")
    print("="*60)

    unique_users = all_reviews_dedup['user_id'].nunique()
    unique_restaurants = all_reviews_dedup['restaurant_id'].nunique()
    total_ratings = len(all_reviews_dedup)

    print(f"Total reviews: {total_ratings:,}")
    print(f"Unique users: {unique_users:,}")
    print(f"Unique restaurants: {unique_restaurants:,}")
    print(f"Average reviews per user: {total_ratings / unique_users:.1f}")
    print(f"Average reviews per restaurant: {total_ratings / unique_restaurants:.1f}")

    # Calculate matrix sparsity
    possible_ratings = unique_users * unique_restaurants
    sparsity = 1 - (total_ratings / possible_ratings)
    print(f"\nUser-Restaurant Matrix:")
    print(f"  Possible ratings: {possible_ratings:,}")
    print(f"  Actual ratings: {total_ratings:,}")
    print(f"  Sparsity: {sparsity*100:.2f}%")

    # Rating distribution
    print(f"\nRating distribution:")
    rating_dist = all_reviews_dedup['rating'].value_counts().sort_index()
    for rating, count in rating_dist.items():
        pct = (count / total_ratings) * 100
        print(f"  {rating}★: {count:,} ({pct:.1f}%)")

    # Data sources breakdown (if available)
    if 'source' in all_reviews_dedup.columns:
        print(f"\nData sources:")
        source_counts = all_reviews_dedup['source'].value_counts()
        for source, count in source_counts.items():
            pct = (count / total_ratings) * 100
            print(f"  {source}: {count:,} ({pct:.1f}%)")

    # Save merged data
    output_file = PATHS['stage4_merged']
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    all_reviews_dedup.to_csv(output_file, index=False)

    print(f"\n{'='*60}")
    print("✓ STAGE 4 COMPLETE!")
    print("="*60)
    print(f"\nMerged data saved to: {output_file}")
    print("\nYou now have a clean, deduplicated dataset ready for model training!")
    print("="*60 + "\n")

    return output_file


if __name__ == '__main__':
    merge_and_deduplicate()
