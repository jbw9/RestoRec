"""
Automated Full Pipeline: Stage 1 → Stage 2 → Stage 3
Stage 1 scrapes pre-filtered restaurants, Stage 2 extracts reviews, Stage 3 builds matrix
"""

import os
import sys
import time
import subprocess
from datetime import datetime
from src.data_collection.restaurant_review_scraper_enhanced import EnhancedRestaurantReviewScraper
from src.data_collection.matrix_builder import build_user_restaurant_matrix, analyze_matrix_quality
from src.data_collection.config import PATHS
import pandas as pd


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def print_stage(stage_num, title):
    """Print stage header"""
    print("\n" + "#"*70)
    print(f"# STAGE {stage_num}: {title}")
    print("#"*70 + "\n")


def run_stage1():
    """Stage 1: Scrape filtered restaurants with complete details"""
    print_stage(1, "RESTAURANT SCRAPING & DETAIL EXTRACTION")

    try:
        from src.data_collection.restaurant_scraper_enhanced import scrape_restaurants_by_urls

        # Load existing restaurants
        if not os.path.exists(PATHS['stage1_restaurants']):
            print(f"✗ Stage 1 source data not found: {PATHS['stage1_restaurants']}")
            return False, None

        existing_df = pd.read_csv(PATHS['stage1_restaurants'])
        print(f"Loaded existing restaurant data: {len(existing_df)} restaurants")

        # Filter by minimum reviews (10+)
        MIN_REVIEWS = 10
        print(f"\nFiltering restaurants with {MIN_REVIEWS}+ reviews...")
        print("-"*70)

        # Convert reviews_count to numeric (handle commas)
        existing_df['reviews_count'] = pd.to_numeric(
            existing_df['reviews_count'].astype(str).str.replace(',', ''),
            errors='coerce'
        )

        filtered_df = existing_df[existing_df['reviews_count'] >= MIN_REVIEWS].copy()
        print(f"Restaurants to scrape: {len(filtered_df)}/{len(existing_df)} ({len(filtered_df)/len(existing_df)*100:.1f}%)")

        # Extract URLs from filtered restaurants
        urls = filtered_df['url'].tolist()
        print(f"\nExtracting URLs from {len(urls)} restaurants...")

        # Scrape only the filtered restaurants
        print("\nStarting detailed scraping of filtered restaurants...")
        print("-"*70 + "\n")

        output_file = scrape_restaurants_by_urls(urls, output_csv_path=PATHS['stage1_restaurants'])

        # Load results
        if output_file and os.path.exists(output_file):
            df = pd.read_csv(output_file)
            print("\n" + "-"*70)
            print("✓ Stage 1 Complete!")
            print(f"  Total restaurants scraped: {len(df)}")

            # Data quality check
            print(f"\nData Quality Check:")
            cuisine_filled = (df['cuisine'] != 'N/A').sum()
            hours_filled = (df['hours'] != 'N/A').sum()
            desc_filled = (df['description'] != 'N/A').sum()
            dining_filled = (df['dining_options'] != 'N/A').sum() if 'dining_options' in df.columns else 0

            print(f"  - Cuisine populated: {cuisine_filled}/{len(df)} ({cuisine_filled/len(df)*100:.1f}%)")
            print(f"  - Hours populated: {hours_filled}/{len(df)} ({hours_filled/len(df)*100:.1f}%)")
            print(f"  - Description populated: {desc_filled}/{len(df)} ({desc_filled/len(df)*100:.1f}%)")
            if 'dining_options' in df.columns:
                print(f"  - Dining options populated: {dining_filled}/{len(df)} ({dining_filled/len(df)*100:.1f}%)")
            else:
                print(f"  - Dining options: NOT IN DATA (column missing)")

            return True, df
        else:
            print("\n✗ Stage 1 Failed: No output file created")
            return False, None

    except Exception as e:
        print(f"\n✗ Stage 1 Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


def run_stage2(filtered_df):
    """Stage 2: Scrape reviews on filtered restaurants"""
    print_stage(2, "REVIEW & REVIEWER URL EXTRACTION")

    try:
        # Use the filtered data directly from Stage 1
        # (it's already been scraped and filtered to 10+ reviews)

        # Clear old data
        print("Clearing old Stage 2 data...")
        for file in [PATHS['stage2_reviews'], PATHS['stage2_user_mapping'], PATHS['stage2_processed_restaurants']]:
            if os.path.exists(file):
                os.remove(file)
                print(f"  ✓ Removed: {file}")

        print(f"\nStarting review scraping for {len(filtered_df)} restaurants...")
        print("-"*70)

        # Run scraper using the filtered restaurants directly
        # Stage 1 output (stage1_restaurants.csv) is already filtered to 10+ reviews
        scraper = EnhancedRestaurantReviewScraper(input_csv_path=PATHS['stage1_restaurants'])
        scraper.process_restaurants(start_row=0, end_row=len(filtered_df))

        # Verify output
        if os.path.exists(PATHS['stage2_reviews']):
            reviews_df = pd.read_csv(PATHS['stage2_reviews'])
            print("\n" + "-"*70)
            print("✓ Stage 2 Complete!")
            print(f"  Total reviews extracted: {len(reviews_df)}")
            print(f"  Unique reviewers: {reviews_df['reviewer_url'].nunique()}")
            print(f"  Unique restaurants: {reviews_df['restaurant_id'].nunique()}")
            return True, reviews_df
        else:
            print("\n✗ Stage 2 Failed: Reviews file not created")
            return False, None

    except Exception as e:
        print(f"\n✗ Stage 2 Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


def run_stage3():
    """Stage 3: Build user-restaurant preference matrix"""
    print_stage(3, "USER-RESTAURANT PREFERENCE MATRIX")

    try:
        print("Building preference matrix...")
        print("-"*70)

        matrix, dense_matrix, profiles, rest_profiles = build_user_restaurant_matrix()

        if matrix is not None:
            print("\n" + "-"*70)
            print("Running matrix quality analysis...")
            print("-"*70)

            analyze_matrix_quality()

            print("\n✓ Stage 3 Complete!")
            return True
        else:
            print("\n✗ Stage 3 Failed: Could not build matrix")
            return False

    except Exception as e:
        print(f"\n✗ Stage 3 Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def print_summary(stage1_success, stage2_success, stage3_success, start_time):
    """Print final summary"""
    elapsed_time = time.time() - start_time
    elapsed_hours = elapsed_time / 3600

    print_header("PIPELINE EXECUTION SUMMARY")

    print("Stage Results:")
    print(f"  Stage 1 (Scraping):         {'✅ PASS' if stage1_success else '❌ FAIL'}")
    print(f"  Stage 2 (Reviews):          {'✅ PASS' if stage2_success else '❌ FAIL'}")
    print(f"  Stage 3 (Matrix):           {'✅ PASS' if stage3_success else '❌ FAIL'}")

    overall_success = stage1_success and stage2_success and stage3_success
    print(f"\n  Overall Status:             {'✅ SUCCESS' if overall_success else '❌ FAILED'}")

    print(f"\nExecution Time: {elapsed_hours:.2f} hours ({elapsed_time:.0f} seconds)")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\nOutput Files:")
    print(f"  ✓ {PATHS['stage1_restaurants']}")
    print(f"  ✓ {PATHS['stage2_reviews']}")
    print(f"  ✓ {PATHS['stage2_user_mapping']}")
    print(f"  ✓ {PATHS['data_raw']}/stage3_user_restaurant_matrix.csv")
    print(f"  ✓ {PATHS['data_raw']}/stage3_user_restaurant_matrix_dense.csv")
    print(f"  ✓ {PATHS['data_raw']}/stage3_reviewer_profiles.csv")
    print(f"  ✓ {PATHS['data_raw']}/stage3_restaurant_profiles.csv")

    if overall_success:
        print("\n🎉 Pipeline completed successfully!")
        print("\nNext Steps:")
        print("  1. Analyze the preference matrix")
        print("  2. Build collaborative filtering recommender")
        print("  3. Test recommendations on sample users")
    else:
        print("\n⚠️  Pipeline encountered errors. Check output above.")


def main():
    """Run full automated pipeline"""

    print_header("RestoRec AUTOMATED PIPELINE")
    print("Stage 1: Restaurant Scraping (pre-filtered by review count)")
    print("Stage 2: Review & Reviewer URL Extraction")
    print("Stage 3: User-Restaurant Preference Matrix")
    print("\nStarting at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    start_time = time.time()

    # Stage 1: Scrape filtered restaurants
    print("\n[1/3] Running Stage 1: Restaurant Scraping...")
    stage1_success, stage1_df = run_stage1()

    if not stage1_success:
        print_summary(False, False, False, start_time)
        return

    # Stage 2: Scrape reviews
    print("\n[2/3] Running Stage 2: Review Scraping...")
    stage2_success, reviews_df = run_stage2(stage1_df)

    if not stage2_success:
        print_summary(stage1_success, False, False, start_time)
        return

    # Stage 3: Build matrix
    print("\n[3/3] Running Stage 3: Building Preference Matrix...")
    stage3_success = run_stage3()

    # Final summary
    print_summary(stage1_success, stage2_success, stage3_success, start_time)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
