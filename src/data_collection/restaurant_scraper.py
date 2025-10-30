"""
Stage 1: Restaurant Discovery (AUTOMATED)
Automatically scrapes Google Maps for Champaign-Urbana restaurants using keyword list

Features:
- Reads search keywords from search_keywords.txt
- Automatically searches, scrolls, and collects URLs for each keyword
- Deduplicates across all searches
- Scrapes detailed information for each restaurant
"""

import csv
import os
import time
import re
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from .config import PATHS, LOCATION, get_random_delay
from .scraper_utils import (
    wait_and_find, generate_restaurant_id, format_hours, safe_get_attribute
)


# Configuration for automated search
KEYWORDS_FILE = "search_keywords.txt"
MAX_SCROLLS = 30  # Maximum scrolls per search
SCROLL_PAUSE = 2.5  # Seconds between scrolls
BETWEEN_SEARCHES_DELAY = 5  # Seconds between different keyword searches


def load_search_keywords(filename):
    """
    Load search keywords from file

    Args:
        filename: Path to keywords file

    Returns:
        List of search keyword strings
    """
    keywords = []

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue

                keywords.append(line)

        return keywords

    except FileNotFoundError:
        print(f"✗ Error: Keywords file not found: {filename}")
        print(f"  Please create '{filename}' with search keywords (one per line)")
        return []


def search_google_maps(driver, search_query):
    """
    Automatically search Google Maps for a query

    Args:
        driver: Selenium WebDriver
        search_query: Search string

    Returns:
        True if search successful, False otherwise
    """
    try:
        # Navigate to Google Maps (only on first search)
        current_url = driver.current_url
        if 'google.com/maps' not in current_url:
            driver.get("https://www.google.com/maps")
            time.sleep(3)

        # Find the search box
        search_box = None
        selectors = [
            "input#searchboxinput",
            "input[name='q']",
            "input[aria-label*='Search']"
        ]

        for selector in selectors:
            try:
                search_box = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                if search_box:
                    break
            except TimeoutException:
                continue

        if not search_box:
            print("    ✗ Could not find search box")
            return False

        # Clear and enter search query
        search_box.clear()
        time.sleep(0.5)
        search_box.send_keys(search_query)
        time.sleep(1)
        search_box.send_keys(Keys.RETURN)

        # Wait for results to load
        time.sleep(4)

        return True

    except Exception as e:
        print(f"    ✗ Error during search: {str(e)}")
        return False


def scroll_results_panel(driver, max_scrolls=30, scroll_pause=2.5):
    """
    Automatically scroll the results panel to load ALL restaurants

    Args:
        driver: Selenium WebDriver
        max_scrolls: Maximum number of scroll attempts
        scroll_pause: Seconds to wait between scrolls

    Returns:
        Number of scrolls performed
    """
    try:
        # Find the scrollable results panel
        scrollable_selectors = [
            "div[role='feed']",
            "div.m6QErb[aria-label]",
            "div[aria-label*='Results']"
        ]

        scrollable_div = None
        for selector in scrollable_selectors:
            try:
                scrollable_div = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                if scrollable_div:
                    break
            except TimeoutException:
                continue

        if not scrollable_div:
            print("    ⚠ Could not find scrollable panel")
            return 0

        # Scroll the results
        scroll_count = 0
        last_height = driver.execute_script("return arguments[0].scrollHeight", scrollable_div)
        no_change_count = 0  # Track consecutive scrolls with no height change

        while scroll_count < max_scrolls:
            # Scroll to bottom
            driver.execute_script(
                "arguments[0].scrollTo(0, arguments[0].scrollHeight);",
                scrollable_div
            )

            scroll_count += 1

            # Wait for new results to load
            time.sleep(scroll_pause)

            # Check if new content loaded
            new_height = driver.execute_script("return arguments[0].scrollHeight", scrollable_div)

            if new_height == last_height:
                no_change_count += 1

                # If height hasn't changed for 3 consecutive scrolls, we're done
                if no_change_count >= 3:
                    break
            else:
                no_change_count = 0  # Reset counter

            last_height = new_height

        return scroll_count

    except Exception as e:
        print(f"    ✗ Error during scrolling: {str(e)}")
        return 0


def collect_restaurant_urls(driver):
    """
    Collect all restaurant URLs from the current results

    Args:
        driver: Selenium WebDriver

    Returns:
        List of restaurant URLs
    """
    try:
        # Find all restaurant links
        link_selector = "a.hfpxzc"

        # Wait a moment for all links to be present
        time.sleep(2)

        elements = driver.find_elements(By.CSS_SELECTOR, link_selector)

        if not elements:
            return []

        # Extract URLs
        urls = []
        for element in elements:
            url = element.get_attribute('href')
            if url and url.startswith('https://www.google.com/maps/place/'):
                urls.append(url)

        # Remove duplicates
        urls = list(set(urls))

        return urls

    except Exception as e:
        print(f"    ✗ Error collecting URLs: {str(e)}")
        return []


def click_reviews_tab(driver):
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
        return False


def scrape_restaurant_details(driver, url):
    """
    Scrape detailed information for a single restaurant

    Args:
        driver: Selenium WebDriver instance
        url: Restaurant Google Maps URL

    Returns:
        Dictionary of restaurant data
    """
    driver.get(url)
    time.sleep(2)

    # Generate unique restaurant ID
    restaurant_id = generate_restaurant_id(url)

    # Basic info
    name = wait_and_find(driver, "h1.DUwDvf")
    name = name.text if name else "N/A"

    address = wait_and_find(driver, "button[data-item-id^='address']")
    address = address.text if address else "N/A"

    # Rating and review count
    rating_element = wait_and_find(driver, "div.F7nice")
    if rating_element:
        rating_parts = rating_element.text.split()
        rating = rating_parts[0] if rating_parts else "N/A"
        reviews_count = rating_parts[1].strip("()") if len(rating_parts) > 1 else "N/A"
    else:
        rating = "N/A"
        reviews_count = "N/A"

    # Price range
    price_range = wait_and_find(driver, "span.mgr77e")
    price_range = price_range.text if price_range else "N/A"

    # Cuisine/category
    cuisine = wait_and_find(driver, "button[jsaction='pane.wfvdle15.category']")
    cuisine = cuisine.text if cuisine else "N/A"

    # Contact info
    phone = wait_and_find(driver, "button[data-item-id^='phone:tel:']")
    phone = phone.text if phone else "N/A"

    website_element = wait_and_find(driver, "a[data-item-id^='authority']")
    website = website_element.get_attribute('href') if website_element else "N/A"

    # Hours
    hours_element = wait_and_find(driver, "div.t39EBf")
    hours = "N/A"
    if hours_element:
        aria_label = hours_element.get_attribute('aria-label')
        if aria_label:
            hours_match = re.search(r"(.*?)\. Hide open hours for the week", aria_label)
            if hours_match:
                hours = hours_match.group(1)

    # Description
    description = wait_and_find(driver, ".PYvSYb")
    description = description.text if description else "N/A"

    # Menu link
    menu_link = wait_and_find(driver, "a[data-item-id='menu']")
    menu = menu_link.get_attribute('href') if menu_link else "N/A"

    # Top commented words (from reviews tab)
    top_commented_words = "N/A"
    reviews_clicked = click_reviews_tab(driver)

    if reviews_clicked:
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.m6QErb"))
            )

            # Try multiple selectors for comment buttons
            selectors = [
                "div.m6QErb button.e2moi",
                "div[jscontroller='GQXWge'] button.e2moi",
                "div.kZuvyf button.e2moi"
            ]

            comment_buttons = []
            for selector in selectors:
                comment_buttons = driver.find_elements(By.CSS_SELECTOR, selector)
                if comment_buttons:
                    break

            top_words = []
            # Skip first button (usually "All reviews")
            for button in comment_buttons[1:]:
                aria_label = button.get_attribute('aria-label')
                if aria_label:
                    match = re.match(r"(.*?),\s*mentioned in (\d+) reviews?", aria_label)
                    if match:
                        word, count = match.groups()
                        top_words.append(f"{word.strip()} ({count.strip()})")

            if top_words:
                top_commented_words = "; ".join(top_words)

        except Exception as e:
            pass

    return {
        'restaurant_id': restaurant_id,
        'name': name,
        'address': address,
        'rating': rating,
        'reviews_count': reviews_count,
        'price_range': price_range,
        'cuisine': cuisine,
        'phone': phone,
        'website': website,
        'hours': hours,
        'description': description,
        'menu': menu,
        'top_commented_words': top_commented_words,
        'url': url
    }


def run_restaurant_scraper():
    """
    Main function to run automated restaurant discovery scraper

    Returns:
        Path to output CSV file
    """
    print("\n" + "="*70)
    print("STAGE 1: AUTOMATED RESTAURANT DISCOVERY")
    print("="*70)

    # Load keywords
    print(f"\nLoading search keywords from: {KEYWORDS_FILE}")
    keywords = load_search_keywords(KEYWORDS_FILE)

    if not keywords:
        print("\n✗ No keywords found. Please create the keywords file.")
        print(f"  Create '{KEYWORDS_FILE}' with search keywords (one per line)")
        return None

    print(f"✓ Loaded {len(keywords)} search keywords\n")

    # Show keywords
    print("Search keywords to process:")
    for i, keyword in enumerate(keywords[:10], 1):
        print(f"  {i}. {keyword}")
    if len(keywords) > 10:
        print(f"  ... and {len(keywords) - 10} more keywords")

    print(f"\nConfiguration:")
    print(f"  Max scrolls per search: {MAX_SCROLLS}")
    print(f"  Scroll pause: {SCROLL_PAUSE}s")
    print(f"  Delay between searches: {BETWEEN_SEARCHES_DELAY}s")
    print(f"\nThis will automatically search, scroll, and collect restaurant URLs.")

    input("\nPress Enter to start automated scraping...")

    # Initialize browser
    print("\n" + "="*70)
    print("Opening Chrome browser...")
    print("="*70)
    driver = webdriver.Chrome()
    driver.maximize_window()

    all_urls = []  # Collect all URLs from all searches

    try:
        # Phase 1: Collect URLs from all keyword searches
        print("\n" + "="*70)
        print("PHASE 1: COLLECTING RESTAURANT URLs")
        print("="*70)

        for i, keyword in enumerate(keywords, 1):
            print(f"\n[{i}/{len(keywords)}] Processing: '{keyword}'")

            # Search
            if not search_google_maps(driver, keyword):
                print(f"  ✗ Search failed, skipping...")
                continue

            # Scroll
            print(f"  Scrolling to load all results...")
            scrolls = scroll_results_panel(driver, max_scrolls=MAX_SCROLLS, scroll_pause=SCROLL_PAUSE)
            print(f"  ✓ Scrolled {scrolls} times")

            # Collect URLs
            print(f"  Collecting URLs...")
            urls = collect_restaurant_urls(driver)

            if urls:
                print(f"  ✓ Collected {len(urls)} URLs")
                all_urls.extend(urls)
            else:
                print(f"  ⚠ No URLs collected")

            # Delay between searches (anti-detection)
            if i < len(keywords):
                time.sleep(BETWEEN_SEARCHES_DELAY)

        # Deduplicate
        unique_urls = list(set(all_urls))

        print(f"\n{'='*70}")
        print("URL COLLECTION SUMMARY")
        print("="*70)
        print(f"Total URLs collected (with duplicates): {len(all_urls)}")
        print(f"Unique restaurants: {len(unique_urls)}")
        print(f"Duplicates removed: {len(all_urls) - len(unique_urls)}")

        if not unique_urls:
            print("\n✗ No URLs were collected. Exiting...")
            return None

        # Phase 2: Scrape individual restaurant details
        print(f"\n{'='*70}")
        print("PHASE 2: SCRAPING RESTAURANT DETAILS")
        print("="*70)
        print(f"Scraping detailed information for {len(unique_urls)} restaurants...\n")

        scraped_data = []
        for index, url in enumerate(unique_urls, 1):
            print(f"[{index}/{len(unique_urls)}] Scraping restaurant...")

            try:
                data = scrape_restaurant_details(driver, url)
                scraped_data.append(data)
                print(f"  ✓ {data['name']}")

                # Random delay between restaurants
                if index < len(unique_urls):
                    delay = get_random_delay('between_restaurants')
                    print(f"  Waiting {delay:.1f}s...\n")
                    time.sleep(delay)

            except Exception as e:
                print(f"  ✗ Error: {str(e)}\n")
                continue

        # Save to CSV
        output_file = PATHS['stage1_restaurants']

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'restaurant_id', 'name', 'address', 'rating', 'reviews_count',
                'price_range', 'cuisine', 'phone', 'website', 'hours',
                'description', 'menu', 'top_commented_words', 'url'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for data in scraped_data:
                data['hours'] = format_hours(data['hours'])
                writer.writerow(data)

        print("\n" + "="*70)
        print("✓ STAGE 1 COMPLETE!")
        print("="*70)
        print(f"\nData saved to: {output_file}")
        print(f"Total restaurants scraped: {len(scraped_data)}")
        print(f"Unique URLs found: {len(unique_urls)}")
        print(f"Keywords processed: {len(keywords)}")
        print("\n✓ You can now proceed to Stage 2 (Restaurant Review Scraping)")

        return output_file

    except KeyboardInterrupt:
        print("\n\n⚠ Scraping interrupted by user")
        print(f"URLs collected so far: {len(set(all_urls))}")
        return None

    finally:
        driver.quit()


if __name__ == "__main__":
    run_restaurant_scraper()
