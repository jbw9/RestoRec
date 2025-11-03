"""
Enhanced Stage 2: Restaurant Review Scraping
Improved scraper with better metadata extraction for dining options, ratings, and dishes

Features:
- Better extraction of dining type (dine-in, takeout, delivery)
- Improved food/service/atmosphere rating extraction
- Better recommended dishes extraction
- Multiple selector fallbacks for robustness
- Enhanced error handling
"""

import csv
import os
import time
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from .config import PATHS, get_random_delay, get_batch_size
from .scraper_utils import (
    scroll_element, expand_all_text, extract_rating_from_stars,
    safe_get_text, safe_get_attribute, has_review_photos
)


class EnhancedRestaurantReviewScraper:
    """Enhanced scraper for collecting comprehensive review data"""

    def __init__(self, input_csv_path=None):
        """Initialize enhanced scraper"""
        self.input_csv_path = input_csv_path or PATHS['stage1_restaurants']
        self.reviews_file = PATHS['stage2_reviews']
        self.user_mapping_file = PATHS['stage2_user_mapping']
        self.processed_restaurants_file = PATHS['stage2_processed_restaurants']

        self.user_id_map = self.load_user_mapping()
        self.next_user_id = max(self.user_id_map.values(), default=0) + 1
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
        """Load existing user mapping"""
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
        """Get existing user ID or create new one"""
        reviewer_identifier = f"{reviewer_name}||{reviewer_url}"
        if reviewer_identifier not in self.user_id_map:
            self.user_id_map[reviewer_identifier] = self.next_user_id
            self.next_user_id += 1
            self.save_user_mapping()
        return self.user_id_map[reviewer_identifier]

    def click_reviews_tab(self, driver):
        """Click the Reviews tab on restaurant page"""
        try:
            selectors = [
                (By.XPATH, "//div[contains(@class, 'Gpq6kf') and contains(text(), 'Reviews')]"),
                (By.XPATH, "//button[contains(text(), 'Reviews')]"),
                (By.XPATH, "//div[@role='button' and contains(., 'Reviews')]"),
            ]

            for by, selector in selectors:
                try:
                    reviews_tab = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((by, selector))
                    )
                    reviews_tab.click()
                    time.sleep(2)
                    return True
                except TimeoutException:
                    continue

            return False
        except Exception as e:
            print(f"  ✗ Could not find or click the Reviews tab")
            return False

    def extract_enhanced_metadata(self, review_element):
        """
        Extract comprehensive metadata from review including dining type,
        ratings, and recommended dishes
        """
        metadata = {
            'dining_type': '',
            'price_range': '',
            'food_rating': '',
            'service_rating': '',
            'atmosphere_rating': '',
            'recommended_dishes': '',
        }

        try:
            # Try multiple selectors for detail spans
            detail_selectors = [
                "span.RfDO5c",
                "div.EWp08c span",
                "div.N4mzCf span",
                "span.Bj57Yf",
                "div[data-review-detail] span"
            ]

            detail_spans = []
            for selector in detail_selectors:
                try:
                    detail_spans = review_element.find_elements(By.CSS_SELECTOR, selector)
                    if detail_spans:
                        break
                except Exception:
                    continue

            # If still no spans, try getting all text and parsing
            if not detail_spans:
                try:
                    review_text = review_element.text
                    # Look for dining type indicators in text
                    if 'Dine in' in review_text:
                        metadata['dining_type'] = 'Dine-in'
                    elif 'Takeout' in review_text:
                        metadata['dining_type'] = 'Takeout'
                    elif 'Delivery' in review_text:
                        metadata['dining_type'] = 'Delivery'

                    # Look for rating indicators
                    if 'Food:' in review_text:
                        match = re.search(r'Food:\s*(\d+\.?\d*)', review_text)
                        if match:
                            metadata['food_rating'] = match.group(1)

                    if 'Service:' in review_text:
                        match = re.search(r'Service:\s*(\d+\.?\d*)', review_text)
                        if match:
                            metadata['service_rating'] = match.group(1)

                    if 'Atmosphere:' in review_text:
                        match = re.search(r'Atmosphere:\s*(\d+\.?\d*)', review_text)
                        if match:
                            metadata['atmosphere_rating'] = match.group(1)

                except Exception:
                    pass
            else:
                # Parse detail spans
                for span in detail_spans:
                    text = span.text.strip()

                    # Dining type detection
                    if "Dine in" in text or "Dine-in" in text:
                        metadata['dining_type'] = 'Dine-in'
                    elif "Takeout" in text or "Take out" in text:
                        metadata['dining_type'] = 'Takeout'
                    elif "Delivery" in text:
                        metadata['dining_type'] = 'Delivery'

                    # Price range detection
                    elif "$" in text and ("–" in text or "-" in text):
                        metadata['price_range'] = text

                    # Rating detection
                    elif "Food:" in text or "food:" in text.lower():
                        match = re.search(r'[:\s]+(\d+\.?\d*)', text)
                        if match:
                            metadata['food_rating'] = match.group(1)

                    elif "Service:" in text or "service:" in text.lower():
                        match = re.search(r'[:\s]+(\d+\.?\d*)', text)
                        if match:
                            metadata['service_rating'] = match.group(1)

                    elif "Atmosphere:" in text or "atmosphere:" in text.lower():
                        match = re.search(r'[:\s]+(\d+\.?\d*)', text)
                        if match:
                            metadata['atmosphere_rating'] = match.group(1)

                    # Recommended dishes - any text that doesn't match above patterns
                    elif not any(keyword in text.lower() for keyword in
                                ['food:', 'service:', 'atmosphere:', 'parking', 'delivery',
                                 'dine', 'takeout', 'take out', '$', '–', '-']):
                        if text and len(text) > 2:
                            if metadata['recommended_dishes']:
                                metadata['recommended_dishes'] += "; " + text
                            else:
                                metadata['recommended_dishes'] = text

        except Exception as e:
            print(f"  ⚠ Error extracting metadata: {str(e)}")

        return metadata

    def get_reviews(self, driver, restaurant_id, restaurant_name):
        """Get all reviews for a restaurant"""
        reviews_data = []
        processed_review_ids = set()

        try:
            if not self.click_reviews_tab(driver):
                return reviews_data

            # Get scrollable container with multiple selector attempts
            scrollable = None
            selectors = [
                "div.m6QErb.DxyBCb.kA9KIf.dS8AEf",
                "div.m6QErb.DxyBCb",
                "div[role='region']",
                "div.kA9KIf"
            ]

            for selector in selectors:
                try:
                    scrollable = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    if scrollable:
                        break
                except TimeoutException:
                    continue

            if not scrollable:
                print("  ✗ Could not find scrollable reviews container")
                return reviews_data

            print("  Scrolling to load reviews...")
            scroll_element(driver, scrollable)

            print("  Expanding review text...")
            expand_all_text(driver)

            # Get all review elements with multiple selectors
            review_selectors = [
                "div.jJc9Ad",
                "div[data-review-id]",
                "div.review-item"
            ]

            reviews = []
            for selector in review_selectors:
                reviews = driver.find_elements(By.CSS_SELECTOR, selector)
                if reviews:
                    break

            print(f"  Found {len(reviews)} reviews")

            for review in reviews:
                try:
                    # Get reviewer info
                    reviewer_button = review.find_element(By.CSS_SELECTOR, "button.al6Kxe")
                    reviewer_name = safe_get_text(reviewer_button, "div.d4r55", "Anonymous")

                    # Get reviewer URL directly from the button's data-href attribute
                    reviewer_url = reviewer_button.get_attribute("data-href") or ""

                    # Fallback: try to extract from jsaction or other attributes if data-href is missing
                    if not reviewer_url:
                        try:
                            # Try to click the button and get the resulting URL
                            jsaction = reviewer_button.get_attribute("jsaction")
                            # If jsaction contains reviewerLink, it's a valid reviewer link
                            if jsaction and "reviewerLink" in jsaction:
                                # Try to get from onclick or data attributes
                                onclick = reviewer_button.get_attribute("onclick")
                                if onclick and "contrib" in onclick:
                                    # Extract URL from onclick
                                    match = re.search(r'(https://www\.google\.com/maps/contrib/[^"\)]+)', onclick)
                                    if match:
                                        reviewer_url = match.group(1)
                        except Exception:
                            pass

                    user_id = self.get_user_id(reviewer_name, reviewer_url)

                    # Get rating
                    rating = extract_rating_from_stars(review)
                    if rating is None:
                        continue

                    # Get review date and text
                    date = safe_get_text(review, "span.rsqaWe", "")
                    text = safe_get_text(review, "span.wiI7pd", "")

                    # Create unique review identifier
                    review_id = f"{user_id}:{restaurant_id}:{text[:50]}"
                    if review_id in processed_review_ids:
                        continue

                    processed_review_ids.add(review_id)

                    # Extract enhanced metadata
                    metadata = self.extract_enhanced_metadata(review)

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

                    # Print review data for verification (check for null values)
                    review_num = len(reviews_data)
                    print(f"    [{review_num}] {reviewer_name[:20]:20s} | ⭐{rating} | URL: {'✓' if reviewer_url else '✗'} | "
                          f"Dining: {metadata.get('dining_type', 'N/A')[:8]:8s} | "
                          f"Food: {metadata.get('food_rating', '-'):>3s} | "
                          f"Dishes: {('✓' if metadata.get('recommended_dishes') else '✗')}")

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
        os.makedirs(os.path.dirname(self.reviews_file), exist_ok=True)

        with open(self.reviews_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=reviews_data[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows(reviews_data)

    def process_restaurants(self, start_row=0, end_row=None):
        """Process restaurants and collect reviews"""
        print("\n" + "="*60)
        print("STAGE 2 ENHANCED: RESTAURANT REVIEW SCRAPING")
        print("="*60)

        if not os.path.exists(self.input_csv_path):
            print(f"\n✗ Error: Input file not found: {self.input_csv_path}")
            print("  Please run Stage 1 first to collect restaurants.")
            return

        df = pd.read_csv(self.input_csv_path)
        total_restaurants = len(df)

        print(f"\nFound {total_restaurants} restaurants in dataset")
        print(f"Already processed: {len(self.processed_restaurant_ids)} restaurants")

        start_row = max(0, min(start_row, total_restaurants - 1))
        end_row = min(end_row if end_row is not None else total_restaurants, total_restaurants)

        if start_row >= end_row:
            print("\n✗ Invalid range: start_row must be less than end_row")
            return

        print(f"\nProcessing restaurants from row {start_row} to {end_row-1}")
        print("="*60 + "\n")

        driver = webdriver.Chrome()
        driver.maximize_window()

        batch_size = get_batch_size('restaurants')
        total_reviews_collected = 0

        try:
            for idx in range(start_row, end_row):
                restaurant_id = str(df.loc[idx, 'restaurant_id'])

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

                    reviews = self.get_reviews(driver, restaurant_id, restaurant_name)

                    if reviews:
                        self.save_reviews(reviews)
                        total_reviews_collected += len(reviews)
                        print(f"  ✓ Saved {len(reviews)} reviews")

                        # Show sample of extracted data
                        if reviews:
                            sample = reviews[0]
                            if sample['dining_type']:
                                print(f"    - Dining: {sample['dining_type']}")
                            if sample['food_rating']:
                                print(f"    - Food Rating: {sample['food_rating']}")
                            if sample['recommended_dishes']:
                                print(f"    - Dishes: {sample['recommended_dishes'][:50]}...")
                    else:
                        print(f"  ⚠ No reviews found")

                    self.save_processed_restaurant(restaurant_id)

                    if idx < end_row - 1:
                        delay = get_random_delay('between_restaurants')
                        print(f"  Waiting {delay:.1f}s...\n")
                        time.sleep(delay)

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
            print("STAGE 2 ENHANCED SUMMARY")
            print("="*60)
            try:
                print(f"Restaurants processed: {idx - start_row + 1}")
            except:
                print(f"Restaurants processed: 0")
            print(f"Total reviews collected: {total_reviews_collected}")
            print(f"Unique users discovered: {len(self.user_id_map)}")
            print(f"\nData saved to:")
            print(f"  Reviews: {self.reviews_file}")
            print(f"  User mapping: {self.user_mapping_file}")


def run_enhanced_restaurant_review_scraper(start_row=0, end_row=None):
    """Main function to run enhanced restaurant review scraper"""
    scraper = EnhancedRestaurantReviewScraper()
    scraper.process_restaurants(start_row, end_row)
    return scraper.reviews_file


if __name__ == "__main__":
    scraper = EnhancedRestaurantReviewScraper()

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
