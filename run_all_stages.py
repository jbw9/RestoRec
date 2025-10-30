#!/usr/bin/env python3
"""
Master Orchestrator Script for RestoRec Data Collection

This script runs all 4 stages of data collection:
  Stage 1: Restaurant Discovery (FULLY AUTOMATED - reads from search_keywords.txt)
  Stage 2: Restaurant Review Scraping (automated)
  Stage 3: User Profile Scraping (automated with CU filtering)
  Stage 4: Merge & Deduplicate (automated)

Usage:
    python run_all_stages.py                    # Run all stages with prompts between stages
    python run_all_stages.py --auto             # Run all stages FULLY AUTOMATICALLY (no prompts!)
    python run_all_stages.py --stage 2          # Run only Stage 2
    python run_all_stages.py --stage 3 --start 0 --end 100  # Run Stage 3 with custom range

Auto Mode:
    Use --auto flag to run everything without any prompts. Perfect for overnight runs!
    Example: python run_all_stages.py --auto
"""

import sys
import os
import argparse
import pandas as pd

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_collection.restaurant_scraper import run_restaurant_scraper
from src.data_collection.restaurant_review_scraper import run_restaurant_review_scraper
from src.data_collection.user_profile_scraper import run_user_profile_scraper
from src.data_processing.merge_reviews import merge_and_deduplicate
from scripts.extract_users_for_scraping import extract_unique_users
from src.data_collection.config import PATHS


def print_banner(text):
    """Print a formatted banner"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def confirm_continue(stage_name, auto_mode=False):
    """Ask user to confirm before continuing to next stage (unless auto mode)"""
    if auto_mode:
        print(f"\n{'='*70}")
        print(f"AUTO MODE: Proceeding to {stage_name}...")
        print("="*70)
        return True

    print(f"\n{'='*70}")
    print(f"Ready to proceed to {stage_name}?")
    print("="*70)
    response = input("Continue? [Y/n]: ").strip().lower()
    if response in ['n', 'no']:
        print("\n⚠ Stopping. You can resume later by running this script again.")
        return False
    return True


def run_stage_1(auto_mode=False):
    """Run Stage 1: Restaurant Discovery"""
    print_banner("STAGE 1: AUTOMATED RESTAURANT DISCOVERY")

    print("This stage automatically collects restaurants from Google Maps using keywords.")
    print("It will read from 'search_keywords.txt' and:")
    print("  - Automatically search for each keyword")
    print("  - Scroll through all results")
    print("  - Collect restaurant URLs")
    print("  - Scrape detailed information")

    if not auto_mode:
        if not confirm_continue("Stage 1", auto_mode):
            return False

    result = run_restaurant_scraper()
    return result is not None


def run_stage_2(start_row=None, end_row=None, auto_mode=False):
    """Run Stage 2: Restaurant Review Scraping"""
    print_banner("STAGE 2: RESTAURANT REVIEW SCRAPING")

    # Check if Stage 1 data exists
    if not os.path.exists(PATHS['stage1_restaurants']):
        print("✗ Error: Stage 1 data not found. Please run Stage 1 first.")
        return False

    # Load restaurant data
    df = pd.read_csv(PATHS['stage1_restaurants'])
    total_restaurants = len(df)

    print(f"Found {total_restaurants} restaurants from Stage 1")
    print("\nThis stage will scrape reviews from each restaurant page.")
    print("This builds the user-restaurant rating matrix for collaborative filtering.")
    print(f"\nEstimated time: {total_restaurants * 0.3:.0f}-{total_restaurants * 0.5:.0f} minutes")

    # Get range if not specified
    if start_row is None or end_row is None:
        if auto_mode:
            # In auto mode, process all restaurants
            start_row = 0
            end_row = total_restaurants
            print(f"\nAUTO MODE: Processing all {total_restaurants} restaurants")
        else:
            print(f"\nYou can process in batches (recommended for large datasets)")
            print(f"Total restaurants: {total_restaurants}")

            if start_row is None:
                start_row = int(input(f"Enter starting row (0-{total_restaurants-1}) [0]: ").strip() or "0")
            if end_row is None:
                end_row_input = input(f"Enter ending row (1-{total_restaurants}) [{total_restaurants}]: ").strip()
                end_row = int(end_row_input) if end_row_input else total_restaurants

    if not auto_mode:
        if not confirm_continue("Stage 2", auto_mode):
            return False

    result = run_restaurant_review_scraper(start_row, end_row)
    return result is not None


def run_extract_users(auto_mode=False):
    """Run user extraction (between Stage 2 and 3)"""
    print_banner("EXTRACTING USERS FOR STAGE 3")

    print("This step extracts all unique users discovered in Stage 2.")
    print("These users will be scraped in Stage 3 to get their full CU review history.")

    if not auto_mode:
        if not confirm_continue("user extraction", auto_mode):
            return False

    result = extract_unique_users()
    return result is not None


def run_stage_3(start_idx=None, end_idx=None, auto_mode=False):
    """Run Stage 3: User Profile Scraping"""
    print_banner("STAGE 3: USER PROFILE SCRAPING")

    # Check if users list exists
    if not os.path.exists(PATHS['users_to_scrape']):
        print("✗ Error: Users list not found. Running extraction first...\n")
        if not run_extract_users(auto_mode):
            return False

    # Load users data
    users_df = pd.read_csv(PATHS['users_to_scrape'])
    total_users = len(users_df)

    print(f"Found {total_users} users to scrape")
    print("\nThis stage will visit each user's profile and extract ALL their CU restaurant reviews.")
    print("This significantly increases data density for better recommendations.")
    print(f"\nEstimated time: {total_users * 0.2:.0f}-{total_users * 0.4:.0f} minutes")

    # Get range if not specified
    if start_idx is None or end_idx is None:
        if auto_mode:
            # In auto mode, process all users
            start_idx = 0
            end_idx = total_users
            print(f"\nAUTO MODE: Processing all {total_users} users")
        else:
            print(f"\nYou can process in batches (HIGHLY recommended)")
            print(f"Total users: {total_users}")
            print("Tip: Start with 50-100 users to test, then scale up")

            if start_idx is None:
                start_idx = int(input(f"Enter starting index (0-{total_users-1}) [0]: ").strip() or "0")
            if end_idx is None:
                end_idx_input = input(f"Enter ending index (1-{total_users}) [100]: ").strip()
                end_idx = int(end_idx_input) if end_idx_input else min(100, total_users)

    if not auto_mode:
        if not confirm_continue("Stage 3", auto_mode):
            return False

    result = run_user_profile_scraper(start_idx, end_idx)
    return result is not None


def run_stage_4(auto_mode=False):
    """Run Stage 4: Merge & Deduplicate"""
    print_banner("STAGE 4: MERGE & DEDUPLICATE")

    print("This stage will:")
    print("  1. Combine reviews from Stage 2 (restaurant pages) and Stage 3 (user profiles)")
    print("  2. Remove duplicate reviews")
    print("  3. Generate final dataset statistics")

    if not auto_mode:
        if not confirm_continue("Stage 4", auto_mode):
            return False

    result = merge_and_deduplicate()
    return result is not None


def main():
    """Main orchestrator"""
    parser = argparse.ArgumentParser(
        description="RestoRec Data Collection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--stage',
        type=int,
        choices=[1, 2, 3, 4],
        help='Run only a specific stage (1-4)'
    )
    parser.add_argument(
        '--start',
        type=int,
        help='Starting index/row for Stage 2 or 3'
    )
    parser.add_argument(
        '--end',
        type=int,
        help='Ending index/row for Stage 2 or 3'
    )
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Run in automatic mode (no confirmations)'
    )

    args = parser.parse_args()

    print_banner("RESTOREC DATA COLLECTION PIPELINE")
    print("This pipeline collects restaurant review data for the recommendation system.")
    print("\nThe pipeline has 4 stages:")
    print("  Stage 1: Restaurant Discovery (FULLY AUTOMATED)")
    print("  Stage 2: Restaurant Review Scraping (automated)")
    print("  Stage 3: User Profile Scraping (automated)")
    print("  Stage 4: Merge & Deduplicate (automated)")
    print("\nYou can run all stages sequentially, or run individual stages.")

    if args.auto:
        print("\n⚡ AUTO MODE ENABLED - All stages will run without prompts!")

    # Run specific stage
    if args.stage:
        if args.stage == 1:
            run_stage_1(args.auto)
        elif args.stage == 2:
            run_stage_2(args.start, args.end, args.auto)
        elif args.stage == 3:
            run_stage_3(args.start, args.end, args.auto)
        elif args.stage == 4:
            run_stage_4(args.auto)
        return

    # Run all stages
    print("\n" + "="*70)
    if args.auto:
        print("Running ALL stages in AUTOMATIC mode...")
    else:
        print("Running ALL stages sequentially...")
    print("="*70)

    # Stage 1
    if not run_stage_1(args.auto):
        print("\n⚠ Stage 1 failed or was cancelled")
        return

    # Stage 2
    if not run_stage_2(auto_mode=args.auto):
        print("\n⚠ Stage 2 failed or was cancelled")
        return

    # Extract users (between Stage 2 and 3)
    if not run_extract_users(args.auto):
        print("\n⚠ User extraction failed")
        return

    # Stage 3
    if not run_stage_3(auto_mode=args.auto):
        print("\n⚠ Stage 3 failed or was cancelled")
        return

    # Stage 4
    if not run_stage_4(args.auto):
        print("\n⚠ Stage 4 failed or was cancelled")
        return

    # Final summary
    print_banner("🎉 ALL STAGES COMPLETE! 🎉")

    print("Your data collection is complete! Here's what you have:\n")

    if os.path.exists(PATHS['stage4_merged']):
        df = pd.read_csv(PATHS['stage4_merged'])
        print(f"✓ Final dataset: {len(df):,} reviews")
        print(f"✓ Unique users: {df['user_id'].nunique():,}")
        print(f"✓ Unique restaurants: {df['restaurant_id'].nunique():,}")
        print(f"✓ Data saved to: {PATHS['stage4_merged']}")

    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("\n1. Proceed to Phase 2: Data Processing & Feature Engineering")
    print("2. Clean and prepare data for model training")
    print("3. Build baseline models (CF, Content-based)")
    print("4. Train hybrid neural network model")
    print("\nSee CU_Restaurant_Recommender_Plan.md for detailed next steps.")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user. You can resume by running this script again.")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
