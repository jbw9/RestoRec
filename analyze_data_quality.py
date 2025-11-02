#!/usr/bin/env python3
"""
Analyze Stage 2 data quality to decide if Stage 3 is necessary
"""

import pandas as pd
import numpy as np

print("="*70)
print("STAGE 2 DATA QUALITY ANALYSIS")
print("="*70)

# Load Stage 2 reviews
print("\nLoading Stage 2 reviews...")
reviews_df = pd.read_csv('data/raw/stage2_restaurant_reviews.csv')

# Basic stats
total_reviews = len(reviews_df)
unique_users = reviews_df['user_id'].nunique()
unique_restaurants = reviews_df['restaurant_id'].nunique()

print(f"\n📊 BASIC STATISTICS:")
print(f"  Total reviews: {total_reviews:,}")
print(f"  Unique users: {unique_users:,}")
print(f"  Unique restaurants: {unique_restaurants:,}")

# Reviews per user
reviews_per_user = reviews_df.groupby('user_id').size()
print(f"\n👥 REVIEWS PER USER:")
print(f"  Mean: {reviews_per_user.mean():.2f}")
print(f"  Median: {reviews_per_user.median():.1f}")
print(f"  Min: {reviews_per_user.min()}")
print(f"  Max: {reviews_per_user.max()}")

print(f"\n  Distribution:")
print(f"    Users with 1 review: {(reviews_per_user == 1).sum():,} ({(reviews_per_user == 1).sum()/unique_users*100:.1f}%)")
print(f"    Users with 2-3 reviews: {((reviews_per_user >= 2) & (reviews_per_user <= 3)).sum():,} ({((reviews_per_user >= 2) & (reviews_per_user <= 3)).sum()/unique_users*100:.1f}%)")
print(f"    Users with 4-9 reviews: {((reviews_per_user >= 4) & (reviews_per_user <= 9)).sum():,} ({((reviews_per_user >= 4) & (reviews_per_user <= 9)).sum()/unique_users*100:.1f}%)")
print(f"    Users with 10+ reviews: {(reviews_per_user >= 10).sum():,} ({(reviews_per_user >= 10).sum()/unique_users*100:.1f}%)")

# Reviews per restaurant
reviews_per_restaurant = reviews_df.groupby('restaurant_id').size()
print(f"\n🍽️  REVIEWS PER RESTAURANT:")
print(f"  Mean: {reviews_per_restaurant.mean():.2f}")
print(f"  Median: {reviews_per_restaurant.median():.1f}")
print(f"  Min: {reviews_per_restaurant.min()}")
print(f"  Max: {reviews_per_restaurant.max()}")

# Data sparsity
total_possible_pairs = unique_users * unique_restaurants
sparsity = (1 - total_reviews / total_possible_pairs) * 100

print(f"\n🎯 DATA DENSITY:")
print(f"  Total possible user-restaurant pairs: {total_possible_pairs:,}")
print(f"  Actual ratings: {total_reviews:,}")
print(f"  Sparsity: {sparsity:.2f}%")
print(f"  Density: {100-sparsity:.4f}%")

# Users suitable for CF (3+ reviews)
users_with_3plus = (reviews_per_user >= 3).sum()
users_with_5plus = (reviews_per_user >= 5).sum()

print(f"\n🤖 COLLABORATIVE FILTERING READINESS:")
print(f"  Users with 3+ reviews: {users_with_3plus:,} ({users_with_3plus/unique_users*100:.1f}%)")
print(f"  Users with 5+ reviews: {users_with_5plus:,} ({users_with_5plus/unique_users*100:.1f}%)")

# Recommendation
print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

if reviews_per_user.median() >= 3 and users_with_5plus/unique_users >= 0.3:
    print("\n✅ YOUR DATA IS EXCELLENT!")
    print("   Stage 3 is OPTIONAL - you have sufficient data for training.")
    print("\n   Reasons:")
    print(f"   - {total_reviews:,} reviews is substantial")
    print(f"   - Median {reviews_per_user.median():.0f} reviews/user is good for CF")
    print(f"   - {users_with_5plus/unique_users*100:.1f}% of users have 5+ reviews")
    print("\n   You can proceed to Stage 4 (merge) and then modeling!")

elif reviews_per_user.median() >= 2:
    print("\n⚠️  YOUR DATA IS DECENT BUT COULD BE BETTER")
    print("   Stage 3 is RECOMMENDED but not critical.")
    print("\n   Current state:")
    print(f"   - {total_reviews:,} reviews is okay")
    print(f"   - Median {reviews_per_user.median():.0f} reviews/user is borderline")
    print(f"   - Only {users_with_5plus/unique_users*100:.1f}% of users have 5+ reviews")
    print("\n   Stage 3 benefits:")
    print("   - Could add 2-3x more reviews per user")
    print("   - Would improve CF model quality significantly")
    print("   - Better handling of cold start problem")
    print("\n   Options:")
    print("   1. Skip Stage 3, go to modeling (faster, less risk)")
    print("   2. Run Stage 3 on subset (e.g., top 500 users)")
    print("   3. Run full Stage 3 (best quality, more time)")

else:
    print("\n❌ YOUR DATA NEEDS MORE REVIEWS")
    print("   Stage 3 is HIGHLY RECOMMENDED.")
    print("\n   Issues:")
    print(f"   - Median {reviews_per_user.median():.0f} reviews/user is too low for good CF")
    print(f"   - Only {users_with_5plus/unique_users*100:.1f}% of users have 5+ reviews")
    print("\n   Stage 3 will significantly improve your model quality!")

print("\n" + "="*70)
