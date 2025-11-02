"""
Stage 2: Restaurant Review Scraping
Scrapes reviews from each restaurant page and builds user-restaurant rating matrix

Adapted from previous restaurant review scraper
"""

import csv
import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from .config import PATHS, get_random_delay, get_batch_size
from .scraper_utils import (
    scroll_element, expand_all_text, extract_rating_from_stars,
    extract_review_metadata, safe_get_text, safe_get_attribute,
    print_progress, has_review_photos
)


class RestaurantReviewScraper:
    """Scraper for collecting reviews from restaurant pages"""

    def __init__(self, input_csv_path=None):
        """
        Initialize scraper

        Args:
            input_csv_path: Path to Stage 1 restaurant CSV (uses default if None)
        """
        self.input_csv_path = input_csv_path or PATHS['stage1_restaurants']
        self.reviews_file = PATHS['stage2_reviews']
        self.user_mapping_file = PATHS['stage2_user_mapping']
        self.processed_restaurants_file = PATHS['stage2_processed_restaurants']

        # Load or initialize user mapping
        self.user_id_map = self.load_user_mapping()
        self.next_user_id = max(self.user_id_map.values(), default=0) + 1

        # Load processed restaurants
        self.processed_restaurant_ids = self.load_processed_restaurants()

    def load_processed_restaurants(self):
        """Load set of already processed restaurant IDs"""
        try:
            if os.path.exists(self.processed_restaurants_file):
                with open(self.processed_restaurants_file, 'r') as f:
                    return set(line.strip() for line in f)
            return set()
        except Exception as e:
            print(f"Error loading processed restaurants: {e}")
            return set()

    def save_processed_restaurant(self, restaurant_id):
        """Save restaurant ID to processed list"""
        os.makedirs(os.path.dirname(self.processed_restaurants_file), exist_ok=True)
        with open(self.processed_restaurants_file, 'a') as f:
            f.write(f"{restaurant_id}\n")
        self.processed_restaurant_ids.add(restaurant_id)

    def load_user_mapping(self):
        """Load existing user mapping or create new one"""
        try:
            if os.path.exists(self.user_mapping_file):
                df = pd.read_csv(self.user_mapping_file)
                return dict(zip(df['reviewer_identifier'], df['user_id']))
            return {}
        except Exception as e:
            print(f"Error loading user mapping: {e}")
            return {}

    def save_user_mapping(self):
        """Save user mapping to CSV"""
        os.makedirs(os.path.dirname(self.user_mapping_file), exist_ok=True)
        df = pd.DataFrame([
            {'user_id': user_id, 'reviewer_identifier': identifier, 'reviewer_url': identifier.split('||')[1]}
            for identifier, user_id in self.user_id_map.items()
        ])
        df.to_csv(self.user_mapping_file, index=False)

    def get_user_id(self, reviewer_name, reviewer_url):
        """
        Get existing user ID or create new one based on name and URL

        Args:
            reviewer_name: Reviewer's display name
            reviewer_url: Reviewer's profile URL

        Returns:
            Integer user ID
        """
        reviewer_identifier = f"{reviewer_name}||{reviewer_url}"
        if reviewer_identifier not in self.user_id_map:
            self.user_id_map[reviewer_identifier] = self.next_user_id
            self.next_user_id += 1
            self.save_user_mapping()
        return self.user_id_map[reviewer_identifier]

    def click_reviews_tab(self, driver):
        """Click the Reviews tab on restaurant page"""
        try:
            reviews_tab = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//div[contains(@class, 'Gpq6kf') and contains(text(), 'Reviews')]")
                )
            )
            reviews_tab.click()
            time.sleep(2)
            return True
        except TimeoutException:
            print("  ✗ Could not find or click the Reviews tab")
            return False

    def get_reviews(self, driver, restaurant_id, restaurant_name):
        """
        Get all reviews for a restaurant

        Args:
            driver: Selenium WebDriver instance
            restaurant_id: Unique restaurant ID
            restaurant_name: Restaurant name

        Returns:
            List of review dictionaries
        """
        reviews_data = []
        processed_review_ids = set()

        try:
            # Click Reviews tab
            if not self.click_reviews_tab(driver):
                return reviews_data

            # Get scrollable container
            try:
                scrollable = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "div.m6QErb.DxyBCb.kA9KIf.dS8AEf")
                    )
                )
            except TimeoutException:
                print("  ✗ Could not find scrollable reviews container")
                return reviews_data

            # Scroll to load all reviews
            print("  Scrolling to load reviews...")
            scroll_element(driver, scrollable)

            # Expand all "More" buttons
            print("  Expanding review text...")
            expand_all_text(driver)

            # Get all review elements
            reviews = driver.find_elements(By.CSS_SELECTOR, "div.jJc9Ad")
            print(f"  Found {len(reviews)} reviews")

            for review in reviews:
                try:
                    # Get reviewer info
                    reviewer_button = review.find_element(By.CSS_SELECTOR, "button.al6Kxe")
                    reviewer_name = safe_get_text(reviewer_button, "div.d4r55", "Anonymous")
                    reviewer_url = safe_get_attribute(reviewer_button, "button.al6Kxe", "data-href", "")

                    # Get user_id using both name and URL
                    user_id = self.get_user_id(reviewer_name, reviewer_url)

                    # Get review content
                    rating = extract_rating_from_stars(review)
                    if rating is None:
                        continue  # Skip reviews without ratings

                    date = safe_get_text(review, "span.rsqaWe", "")
                    text = safe_get_text(review, "span.wiI7pd", "")

                    # Create unique review identifier
                    review_id = f"{user_id}:{restaurant_id}:{text[:50]}"
                    if review_id in processed_review_ids:
                        continue

                    processed_review_ids.add(review_id)

                    # Get additional metadata
                    metadata = extract_review_metadata(review)

                    # Check if review has photos
                    has_photos = has_review_photos(review)

                    review_data = {
                        'restaurant_id': restaurant_id,
                        'restaurant_name': restaurant_name,
                        'user_id': user_id,
                        'reviewer_name': reviewer_name,
                        'reviewer_url': reviewer_url,
                        'rating': rating,
                        'review_date': date,
                        'review_text': text,
                        'has_photos': has_photos,
                        'dining_type': metadata.get('dining_type', ''),
                        'price_range': metadata.get('price_range', ''),
                        'food_rating': metadata.get('food_rating', ''),
                        'service_rating': metadata.get('service_rating', ''),
                        'atmosphere_rating': metadata.get('atmosphere_rating', ''),
                        'recommended_dishes': metadata.get('recommended_dishes', ''),
                        'source': 'restaurant_page'
                    }

                    reviews_data.append(review_data)

                except Exception as e:
                    print(f"  ✗ Error processing a review: {str(e)}")
                    continue

        except Exception as e:
            print(f"  ✗ Error in get_reviews: {str(e)}")

        return reviews_data

    def save_reviews(self, reviews_data):
        """Save reviews to CSV file"""
        if not reviews_data:
            return

        file_exists = os.path.exists(self.reviews_file)

        # Create directory if needed
        os.makedirs(os.path.dirname(self.reviews_file), exist_ok=True)

        with open(self.reviews_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=reviews_data[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows(reviews_data)

    def process_restaurants(self, start_row=0, end_row=None):
        """
        Process restaurants and collect reviews

        Args:
            start_row: Starting row index (0-based)
            end_row: Ending row index (exclusive), None for all remaining
        """
        print("\n" + "="*60)
        print("STAGE 2: RESTAURANT REVIEW SCRAPING")
        print("="*60)

        # Read the input CSV
        if not os.path.exists(self.input_csv_path):
            print(f"\n✗ Error: Input file not found: {self.input_csv_path}")
            print("  Please run Stage 1 first to collect restaurants.")
            return

        df = pd.read_csv(self.input_csv_path)
        total_restaurants = len(df)

        print(f"\nFound {total_restaurants} restaurants in dataset")
        print(f"Already processed: {len(self.processed_restaurant_ids)} restaurants")

        # Validate row numbers
        start_row = max(0, min(start_row, total_restaurants - 1))
        end_row = min(end_row if end_row is not None else total_restaurants, total_restaurants)

        if start_row >= end_row:
            print("\n✗ Invalid range: start_row must be less than end_row")
            return

        print(f"\nProcessing restaurants from row {start_row} to {end_row-1}")
        print("="*60 + "\n")

        # Initialize webdriver
        driver = webdriver.Chrome()
        driver.maximize_window()

        batch_size = get_batch_size('restaurants')
        total_reviews_collected = 0

        try:
            for idx in range(start_row, end_row):
                restaurant_id = str(df.loc[idx, 'restaurant_id'])

                # Skip if already processed
                if restaurant_id in self.processed_restaurant_ids:
                    print(f"[{idx+1}/{end_row}] Skipping {restaurant_id} - already processed\n")
                    continue

                restaurant_url = df.loc[idx, 'url']
                restaurant_name = df.loc[idx, 'name']

                print(f"[{idx+1}/{end_row}] {restaurant_name}")
                print(f"  ID: {restaurant_id}")

                try:
                    driver.get(restaurant_url)
                    time.sleep(2)

                    # Get reviews
                    reviews = self.get_reviews(driver, restaurant_id, restaurant_name)

                    if reviews:
                        self.save_reviews(reviews)
                        total_reviews_collected += len(reviews)
                        print(f"  ✓ Saved {len(reviews)} reviews")
                    else:
                        print(f"  ⚠ No reviews found")

                    # Mark restaurant as processed
                    self.save_processed_restaurant(restaurant_id)

                    # Random delay between restaurants
                    if idx < end_row - 1:
                        delay = get_random_delay('between_restaurants')
                        print(f"  Waiting {delay:.1f}s...\n")
                        time.sleep(delay)

                    # Take a break every batch_size restaurants
                    if (idx - start_row + 1) % batch_size == 0 and idx < end_row - 1:
                        break_time = get_random_delay('batch_break')
                        print(f"\n{'='*60}")
                        print(f"Completed {idx - start_row + 1} restaurants.")
                        print(f"Taking a {break_time/60:.1f} minute break...")
                        print(f"{'='*60}\n")
                        time.sleep(break_time)

                except Exception as e:
                    print(f"  ✗ Error processing restaurant: {str(e)}\n")
                    continue

        except KeyboardInterrupt:
            print("\n\n⚠ Scraping interrupted by user")

        finally:
            driver.quit()
            self.save_user_mapping()

            print("\n" + "="*60)
            print("STAGE 2 SUMMARY")
            print("="*60)
            print(f"Restaurants processed: {idx - start_row + 1}")
            print(f"Total reviews collected: {total_reviews_collected}")
            print(f"Unique users discovered: {len(self.user_id_map)}")
            print(f"\nData saved to:")
            print(f"  Reviews: {self.reviews_file}")
            print(f"  User mapping: {self.user_mapping_file}")
            print("\nYou can now proceed to Stage 3 (User Profile Scraping)")


def run_restaurant_review_scraper(start_row=0, end_row=None):
    """
    Main function to run restaurant review scraper

    Args:
        start_row: Starting row index
        end_row: Ending row index (None for all)

    Returns:
        Path to reviews CSV file
    """
    scraper = RestaurantReviewScraper()
    scraper.process_restaurants(start_row, end_row)
    return scraper.reviews_file


if __name__ == "__main__":
    # Interactive mode
    scraper = RestaurantReviewScraper()

    try:
        df = pd.read_csv(scraper.input_csv_path)
        total_rows = len(df)

        print(f"\nFound {total_rows} restaurants")
        print(f"Already processed: {len(scraper.processed_restaurant_ids)}")
        print("\nYou can process in batches")

        start = int(input(f"\nEnter starting row (0-{total_rows-1}): "))
        end = int(input(f"Enter ending row (1-{total_rows}): "))

        scraper.process_restaurants(start, end)

    except FileNotFoundError:
        print(f"\n✗ Error: Could not find {scraper.input_csv_path}")
        print("  Please run Stage 1 first")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
