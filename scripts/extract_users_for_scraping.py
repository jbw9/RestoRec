"""
Extract unique users from Stage 2 reviews for Stage 3 profile scraping

This script creates a prioritized list of users to scrape based on their
review count (more prolific reviewers = more valuable data)
"""

import sys
import os
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collection.config import PATHS


def extract_unique_users():
    """
    Extract unique users from Stage 2 reviews

    Returns:
        DataFrame of users to scrape
    """
    print("\n" + "="*60)
    print("EXTRACTING USERS FOR STAGE 3")
    print("="*60 + "\n")

    # Load Stage 2 reviews
    reviews_file = PATHS['stage2_reviews']

    if not os.path.exists(reviews_file):
        print(f"✗ Error: Reviews file not found: {reviews_file}")
        print("  Please run Stage 2 first")
        return None

    print(f"Loading reviews from: {reviews_file}")
    reviews_df = pd.read_csv(reviews_file)

    print(f"✓ Loaded {len(reviews_df)} reviews\n")

    # Get unique users with their URLs
    user_df = reviews_df[['user_id', 'reviewer_url']].drop_duplicates()

    # Filter out users with no URL (can't scrape them)
    initial_count = len(user_df)
    user_df = user_df[user_df['reviewer_url'].notna()]
    user_df = user_df[user_df['reviewer_url'] != '']
    user_df = user_df[user_df['reviewer_url'] != 'N/A']

    filtered_count = len(user_df)
    print(f"Users with valid URLs: {filtered_count} (filtered out {initial_count - filtered_count})")

    # Add review count metadata
    review_counts = reviews_df['user_id'].value_counts()
    user_df['review_count_stage2'] = user_df['user_id'].map(review_counts)

    # Sort by review count (scrape prolific reviewers first - more valuable)
    user_df = user_df.sort_values('review_count_stage2', ascending=False)

    # Save to file
    output_file = PATHS['users_to_scrape']
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    user_df.to_csv(output_file, index=False)

    print(f"\n✓ Extracted {len(user_df)} unique users")
    print(f"✓ Saved to: {output_file}\n")

    # Print statistics
    print("="*60)
    print("USER STATISTICS")
    print("="*60)
    print(f"Total unique users: {len(user_df)}")
    print(f"Average reviews per user (from Stage 2): {user_df['review_count_stage2'].mean():.1f}")
    print(f"Median reviews per user: {user_df['review_count_stage2'].median():.0f}")
    print(f"Max reviews by a single user: {user_df['review_count_stage2'].max():.0f}")
    print(f"Min reviews by a single user: {user_df['review_count_stage2'].min():.0f}")

    # Show top 10 reviewers
    print(f"\nTop 10 most prolific reviewers:")
    print(user_df.head(10)[['user_id', 'review_count_stage2']].to_string(index=False))

    print("\n" + "="*60)
    print("You can now proceed to Stage 3 to scrape user profiles")
    print("="*60 + "\n")

    return user_df


if __name__ == '__main__':
    extract_unique_users()
