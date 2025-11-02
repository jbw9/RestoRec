#!/usr/bin/env python3
"""
Filter to get top N most active users for Stage 3
"""

import pandas as pd
import sys

# Configuration
TOP_N_USERS = 2500  # Target 2,500 users (compromise between 2K-3K)

print("="*70)
print("FILTERING TOP USERS FOR STAGE 3")
print("="*70)

# Load Stage 2 reviews
print("\nLoading Stage 2 reviews...")
reviews_df = pd.read_csv('data/raw/stage2_restaurant_reviews.csv')

# Count reviews per user
print("Analyzing user activity...")
user_review_counts = reviews_df.groupby('user_id').size().reset_index(name='review_count')
user_review_counts = user_review_counts.sort_values('review_count', ascending=False)

print(f"\nTotal users: {len(user_review_counts):,}")
print(f"Target users to scrape: {TOP_N_USERS:,}")

# Get top N users
top_users = user_review_counts.head(TOP_N_USERS)

print(f"\nTop {TOP_N_USERS} users statistics:")
print(f"  Min reviews: {top_users['review_count'].min()}")
print(f"  Max reviews: {top_users['review_count'].max()}")
print(f"  Median reviews: {top_users['review_count'].median():.1f}")
print(f"  Mean reviews: {top_users['review_count'].mean():.2f}")

# Get user details from user_mapping
print("\nLoading user mapping...")
user_mapping_df = pd.read_csv('data/raw/stage2_user_mapping.csv')

# Merge to get user details
top_users_with_details = top_users.merge(
    user_mapping_df[['user_id', 'reviewer_identifier', 'reviewer_url']],
    on='user_id',
    how='left'
)

# Remove duplicates (same user might appear multiple times)
top_users_with_details = top_users_with_details.drop_duplicates(subset=['user_id'])

# Filter out users without URLs
users_with_urls = top_users_with_details[top_users_with_details['reviewer_url'].notna()]

print(f"\nUsers with valid URLs: {len(users_with_urls):,}")

# Save to users_to_scrape.csv
output_file = 'data/raw/users_to_scrape.csv'
users_with_urls.to_csv(output_file, index=False)

print(f"\n✓ Saved {len(users_with_urls):,} users to: {output_file}")

# Show sample
print("\nSample of top users:")
print(users_with_urls.head(10)[['user_id', 'reviewer_identifier', 'review_count']])

# Estimate time
estimated_minutes = len(users_with_urls) * 0.3  # ~18 seconds per user
estimated_hours = estimated_minutes / 60

print(f"\n⏱️  ESTIMATED TIME FOR STAGE 3:")
print(f"   {estimated_minutes:.0f} minutes ({estimated_hours:.1f} hours)")

print("\n" + "="*70)
print("Ready to run Stage 3!")
print("="*70)
print("\nCommand to run:")
print(f"  python3 run_all_stages.py --stage 3 --start 0 --end {len(users_with_urls)}")
print("\nOr run in auto mode:")
print("  python3 run_all_stages.py --stage 3 --auto")
print("="*70)
