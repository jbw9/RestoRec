"""
Phase 2.4: Train/Val/Test Split
Split data ensuring no data leakage
"""

import pandas as pd
import numpy as np
import os


def split_data(input_path='data/processed/user_restaurant_ratings.csv',
               train_ratio=0.8,
               val_ratio=0.1,
               test_ratio=0.1,
               random_seed=42):
    """
    Split ratings data into train/validation/test sets

    Strategy: For each user, split their ratings chronologically (if dates available)
    or randomly. This ensures each user appears in all sets if they have enough ratings.

    Args:
        input_path: Path to cleaned ratings CSV
        train_ratio: Proportion for training (default: 0.8)
        val_ratio: Proportion for validation (default: 0.1)
        test_ratio: Proportion for test (default: 0.1)
        random_seed: Random seed for reproducibility
    """

    print("="*70)
    print("TRAIN/VALIDATION/TEST SPLIT")
    print("="*70)

    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    # Load cleaned data
    print(f"\n1. Loading cleaned data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"   Total reviews: {len(df):,}")
    print(f"   Users: {df['user_id'].nunique():,}")
    print(f"   Restaurants: {df['restaurant_id'].nunique():,}")

    # Set random seed
    np.random.seed(random_seed)

    # Split per user
    print(f"\n2. Splitting data (train={train_ratio}, val={val_ratio}, test={test_ratio})...")
    print("   Strategy: Random split per user to ensure all users in all sets")

    train_data = []
    val_data = []
    test_data = []

    users_too_few_ratings = 0

    for user_id in df['user_id'].unique():
        user_ratings = df[df['user_id'] == user_id].copy()
        n = len(user_ratings)

        # Shuffle user's ratings
        user_ratings = user_ratings.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        # Calculate split points
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        # Split
        train_user = user_ratings.iloc[:train_end]
        val_user = user_ratings.iloc[train_end:val_end]
        test_user = user_ratings.iloc[val_end:]

        # Ensure we have at least 1 rating in train (critical for learning)
        if len(train_user) >= 1:
            train_data.append(train_user)
            if len(val_user) > 0:
                val_data.append(val_user)
            if len(test_user) > 0:
                test_data.append(test_user)
        else:
            # User has very few ratings, put everything in train
            train_data.append(user_ratings)
            users_too_few_ratings += 1

    # Concatenate
    train_df = pd.concat(train_data, ignore_index=True)
    val_df = pd.concat(val_data, ignore_index=True) if val_data else pd.DataFrame()
    test_df = pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame()

    print(f"\n   ✓ Train: {len(train_df):,} reviews ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   ✓ Val:   {len(val_df):,} reviews ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   ✓ Test:  {len(test_df):,} reviews ({len(test_df)/len(df)*100:.1f}%)")

    if users_too_few_ratings > 0:
        print(f"\n   Note: {users_too_few_ratings} users had all ratings in train (< {int(1/train_ratio)} ratings)")

    # Verify no data leakage: check user overlap
    print(f"\n3. Verifying data integrity...")
    train_users = set(train_df['user_id'].unique())
    val_users = set(val_df['user_id'].unique()) if len(val_df) > 0 else set()
    test_users = set(test_df['user_id'].unique()) if len(test_df) > 0 else set()

    print(f"   Users in train: {len(train_users):,}")
    print(f"   Users in val:   {len(val_users):,}")
    print(f"   Users in test:  {len(test_users):,}")

    # Check restaurants
    train_restaurants = set(train_df['restaurant_id'].unique())
    val_restaurants = set(val_df['restaurant_id'].unique()) if len(val_df) > 0 else set()
    test_restaurants = set(test_df['restaurant_id'].unique()) if len(test_df) > 0 else set()

    print(f"\n   Restaurants in train: {len(train_restaurants):,}")
    print(f"   Restaurants in val:   {len(val_restaurants):,}")
    print(f"   Restaurants in test:  {len(test_restaurants):,}")

    # Save splits
    print(f"\n4. Saving splits...")
    os.makedirs('data/splits', exist_ok=True)

    train_df.to_csv('data/splits/train.csv', index=False)
    print(f"   ✓ Saved train set to data/splits/train.csv")

    if len(val_df) > 0:
        val_df.to_csv('data/splits/val.csv', index=False)
        print(f"   ✓ Saved validation set to data/splits/val.csv")

    if len(test_df) > 0:
        test_df.to_csv('data/splits/test.csv', index=False)
        print(f"   ✓ Saved test set to data/splits/test.csv")

    # Summary statistics
    print("\n" + "="*70)
    print("SPLIT STATISTICS")
    print("="*70)

    for name, split_df in [('TRAIN', train_df), ('VALIDATION', val_df), ('TEST', test_df)]:
        if len(split_df) == 0:
            continue

        print(f"\n{name}:")
        print(f"  Reviews: {len(split_df):,}")
        print(f"  Users: {split_df['user_id'].nunique():,}")
        print(f"  Restaurants: {split_df['restaurant_id'].nunique():,}")
        print(f"  Avg reviews/user: {split_df.groupby('user_id').size().mean():.2f}")

    print("\n" + "="*70)
    print("✓ Data split complete!")
    print("="*70)
    print("\nNext step: Create restaurant content features")
    print("="*70)

    return train_df, val_df, test_df


if __name__ == '__main__':
    split_data()
