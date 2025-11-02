"""
Stage 3: User Profile Scraping
Scrapes user profiles to get ALL their CU restaurant reviews (not just top reviews)

Enhanced with fuzzy matching to filter for Champaign-Urbana restaurants only
"""

import csv
import os
import time
import random
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from fuzzywuzzy import fuzz

from .config import PATHS, FUZZY_MATCHING, get_random_delay, get_batch_size
from .scraper_utils import (
    normalize_restaurant_name, scroll_element, expand_all_text,
    extract_rating_from_stars, extract_review_metadata,
    safe_get_text, print_progress, has_review_photos
)


class UserProfileScraper:
    """Scraper for collecting reviews from user profiles with CU filtering"""

    def __init__(self, cu_restaurants_csv=None, users_to_scrape_csv=None):
        """
        Initialize scraper

        Args:
            cu_restaurants_csv: Path to Stage 1 restaurant CSV
            users_to_scrape_csv: Path to users list CSV
        """
        self.cu_restaurants_csv = cu_restaurants_csv or PATHS['stage1_restaurants']
        self.users_to_scrape_csv = users_to_scrape_csv or PATHS['users_to_scrape']
        self.output_file = PATHS['stage3_reviews']
        self.processed_users_file = PATHS['stage3_processed_users']

        # Load CU restaurants for filtering
        self.cu_restaurants = self.load_cu_restaurants()

        # Load processed users
        self.processed_users = self.load_processed_users()

    def load_cu_restaurants(self):
        """Load CU restaurant names and addresses for filtering"""
        if not os.path.exists(self.cu_restaurants_csv):
            print(f"✗ Error: Restaurant file not found: {self.cu_restaurants_csv}")
            return []

        df = pd.read_csv(self.cu_restaurants_csv)

        restaurants = []
        for _, row in df.iterrows():
            restaurants.append({
                'restaurant_id': row['restaurant_id'],
                'name': row['name'].lower().strip() if pd.notna(row['name']) else '',
                'address': row['address'].lower().strip() if pd.notna(row['address']) else '',
                'name_normalized': normalize_restaurant_name(row['name']) if pd.notna(row['name']) else ''
            })

        print(f"✓ Loaded {len(restaurants)} CU restaurants for filtering")
        return restaurants

    def is_cu_restaurant(self, restaurant_name, restaurant_address=''):
        """
        Check if a restaurant is in our CU list using fuzzy matching

        Args:
            restaurant_name: Restaurant name from user review
            restaurant_address: Restaurant address (optional)

        Returns:
            Tuple of (is_match: bool, restaurant_id: str or None)
        """
        if not restaurant_name:
            return False, None

        normalized_name = normalize_restaurant_name(restaurant_name)

        # Try exact match first
        for resto in self.cu_restaurants:
            if resto['name_normalized'] == normalized_name:
                return True, resto['restaurant_id']

        # Fuzzy matching
        best_match_score = 0
        best_match_id = None

        for resto in self.cu_restaurants:
            # Name similarity
            name_score = fuzz.ratio(normalized_name, resto['name_normalized'])

            # Address similarity (if available)
            address_score = 0
            if restaurant_address and resto['address']:
                address_score = fuzz.ratio(
                    restaurant_address.lower(),
                    resto['address']
                )

            # Combined score
            name_weight = FUZZY_MATCHING['name_weight']
            address_weight = FUZZY_MATCHING['address_weight']
            combined_score = name_weight * name_score + address_weight * address_score

            if combined_score > best_match_score:
                best_match_score = combined_score
                best_match_id = resto['restaurant_id']

        # Check threshold
        threshold = FUZZY_MATCHING['name_match_threshold']
        if best_match_score > threshold:
            return True, best_match_id

        return False, None

    def load_processed_users(self):
        """Load already processed user IDs"""
        if os.path.exists(self.processed_users_file):
            with open(self.processed_users_file, 'r') as f:
                return set(line.strip() for line in f)
        return set()

    def save_processed_user(self, user_id):
        """Mark user as processed"""
        os.makedirs(os.path.dirname(self.processed_users_file), exist_ok=True)
        with open(self.processed_users_file, 'a') as f:
            f.write(f"{user_id}\n")
        self.processed_users.add(str(user_id))

    def scrape_user_reviews(self, driver, user_id, user_url):
        """
        Scrape all reviews from a user's profile, filter for CU restaurants

        Args:
            driver: Selenium WebDriver instance
            user_id: User ID
            user_url: User's profile URL

        Returns:
            List of CU restaurant reviews
        """
        print(f"  Scraping user {user_id}")

        try:
            driver.get(user_url)
            time.sleep(3)

            # Scroll to load all reviews
            print(f"    Scrolling to load all reviews...")
            scrollable = driver.find_element(By.CSS_SELECTOR, "div.m6QErb.DxyBCb.kA9KIf.dS8AEf")
            scroll_element(driver, scrollable)

            # Expand all review text
            print(f"    Expanding review text...")
            expand_all_text(driver)

            # Extract reviews
            review_elements = driver.find_elements(By.CSS_SELECTOR, "div.jJc9Ad")
            total_reviews = len(review_elements)

            print(f"    Found {total_reviews} total reviews")

            cu_reviews = []

            for review_elem in review_elements:
                try:
                    # Extract restaurant name
                    place_name = safe_get_text(review_elem, "div.DUwDvf", "")

                    if not place_name:
                        continue

                    # Try to get address (may not always be available)
                    address = safe_get_text(review_elem, "div.RfDO5c", "")

                    # CHECK: Is this a CU restaurant?
                    is_cu, restaurant_id = self.is_cu_restaurant(place_name, address)

                    if not is_cu:
                        continue  # Skip non-CU restaurants

                    # Extract review data
                    rating = extract_rating_from_stars(review_elem)
                    if rating is None:
                        continue

                    date = safe_get_text(review_elem, "span.rsqaWe", "")
                    text = safe_get_text(review_elem, "span.wiI7pd", "")

                    # Extract additional metadata
                    metadata = extract_review_metadata(review_elem)

                    # Check if review has photos
                    has_photos = has_review_photos(review_elem)

                    review_data = {
                        'restaurant_id': restaurant_id,
                        'restaurant_name': place_name,
                        'user_id': user_id,
                        'rating': rating,
                        'review_date': date,
                        'review_text': text,
                        'has_photos': has_photos,
                        'dining_type': metadata.get('dining_type', ''),
                        'food_rating': metadata.get('food_rating', ''),
                        'service_rating': metadata.get('service_rating', ''),
                        'atmosphere_rating': metadata.get('atmosphere_rating', ''),
                        'recommended_dishes': metadata.get('recommended_dishes', ''),
                        'source': 'user_profile'
                    }

                    cu_reviews.append(review_data)

                except Exception as e:
                    print(f"    ✗ Error processing review: {str(e)}")
                    continue

            print(f"    ✓ Found {len(cu_reviews)} CU restaurant reviews (out of {total_reviews} total)")
            return cu_reviews

        except Exception as e:
            print(f"    ✗ Error scraping user {user_id}: {str(e)}")
            return []

    def save_reviews(self, reviews_data):
        """Append reviews to CSV"""
        if not reviews_data:
            return

        file_exists = os.path.exists(self.output_file)

        # Create directory if needed
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=reviews_data[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows(reviews_data)

    def process_users(self, start_idx=0, end_idx=None):
        """
        Process user profiles in batches

        Args:
            start_idx: Starting index in users CSV
            end_idx: Ending index (None for all)
        """
        print("\n" + "="*60)
        print("STAGE 3: USER PROFILE SCRAPING")
        print("="*60)

        # Load users to scrape
        if not os.path.exists(self.users_to_scrape_csv):
            print(f"\n✗ Error: Users file not found: {self.users_to_scrape_csv}")
            print("  Please run the extract_users script first")
            return

        users_df = pd.read_csv(self.users_to_scrape_csv)
        total_users = len(users_df)

        if end_idx is None:
            end_idx = total_users

        print(f"\nTotal users to scrape: {total_users}")
        print(f"Already processed: {len(self.processed_users)}")
        print(f"Processing users from index {start_idx} to {end_idx-1}")
        print("="*60 + "\n")

        # Initialize webdriver
        driver = webdriver.Chrome()
        driver.maximize_window()

        batch_size = get_batch_size('users')
        total_reviews_collected = 0

        try:
            for idx in range(start_idx, end_idx):
                user_row = users_df.iloc[idx]
                user_id = str(user_row['user_id'])
                user_url = user_row['reviewer_url']

                # Skip if already processed
                if user_id in self.processed_users:
                    print(f"[{idx+1}/{end_idx}] Skipping user {user_id} - already processed\n")
                    continue

                print(f"[{idx+1}/{end_idx}] User ID: {user_id}")

                # Scrape user's reviews
                reviews = self.scrape_user_reviews(driver, user_id, user_url)

                if reviews:
                    self.save_reviews(reviews)
                    total_reviews_collected += len(reviews)
                    print(f"  ✓ Saved {len(reviews)} CU reviews")
                else:
                    print(f"  ⚠ No CU reviews found")

                # Mark as processed
                self.save_processed_user(user_id)

                # Random delay between users
                if idx < end_idx - 1:
                    delay = get_random_delay('between_users')
                    print(f"  Waiting {delay:.1f}s...\n")
                    time.sleep(delay)

                # Take a break every batch_size users
                if (idx - start_idx + 1) % batch_size == 0 and idx < end_idx - 1:
                    break_time = 120  # 2 minutes
                    print(f"\n{'='*60}")
                    print(f"Completed {idx - start_idx + 1} users.")
                    print(f"Taking a {break_time/60:.0f} minute break...")
                    print(f"{'='*60}\n")
                    time.sleep(break_time)

        except KeyboardInterrupt:
            print("\n\n⚠ Scraping interrupted by user")

        finally:
            driver.quit()

            print("\n" + "="*60)
            print("STAGE 3 SUMMARY")
            print("="*60)
            print(f"Users processed: {idx - start_idx + 1}")
            print(f"Total CU reviews collected: {total_reviews_collected}")
            print(f"\nData saved to: {self.output_file}")
            print("\nYou can now proceed to Stage 4 (Merge & Deduplicate)")


def run_user_profile_scraper(start_idx=0, end_idx=None):
    """
    Main function to run user profile scraper

    Args:
        start_idx: Starting user index
        end_idx: Ending user index (None for all)

    Returns:
        Path to reviews CSV file
    """
    scraper = UserProfileScraper()
    scraper.process_users(start_idx, end_idx)
    return scraper.output_file


if __name__ == "__main__":
    # Interactive mode
    scraper = UserProfileScraper()

    try:
        users_df = pd.read_csv(scraper.users_to_scrape_csv)
        total_users = len(users_df)

        print(f"\nFound {total_users} users to scrape")
        print(f"Already processed: {len(scraper.processed_users)}")
        print("\nYou can process in batches")

        start = int(input(f"\nEnter starting index (0-{total_users-1}): "))
        end = int(input(f"Enter ending index (1-{total_users}): "))

        scraper.process_users(start, end)

    except FileNotFoundError:
        print(f"\n✗ Error: Could not find users file")
        print("  Please run extract_users script first")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
