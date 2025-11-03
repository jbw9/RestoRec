"""
Enhanced Stage 1: Restaurant Discovery & Details Scraper
Improved scraper with better selectors, image extraction, and comprehensive data collection

Features:
- Multiple fallback selectors for unreliable Google Maps elements
- Thumbnail image extraction and local storage
- Comprehensive field extraction (cuisine, description, hours, dining options)
- Better handling of dynamic content
- Robust error handling with detailed logging
"""

import csv
import os
import time
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from .config import PATHS, LOCATION, get_random_delay
from .scraper_utils import (
    wait_and_find, generate_restaurant_id, format_hours, safe_get_attribute
)


KEYWORDS_FILE = "search_keywords.txt"
MAX_SCROLLS = 30
SCROLL_PAUSE = 2.5
BETWEEN_SEARCHES_DELAY = 5


def load_search_keywords(filename):
    """Load search keywords from file"""
    keywords = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                keywords.append(line)
        return keywords
    except FileNotFoundError:
        print(f"✗ Error: Keywords file not found: {filename}")
        return []


def search_google_maps(driver, search_query):
    """Automatically search Google Maps for a query"""
    try:
        current_url = driver.current_url
        if 'google.com/maps' not in current_url:
            driver.get("https://www.google.com/maps")
            time.sleep(3)

        # Multiple selector attempts
        selectors = [
            "input#searchboxinput",
            "input[name='q']",
            "input[aria-label*='Search']",
            "input[role='combobox']"
        ]

        search_box = None
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

        search_box.clear()
        time.sleep(0.5)
        search_box.send_keys(search_query)
        time.sleep(1)
        search_box.send_keys(Keys.RETURN)
        time.sleep(4)

        return True

    except Exception as e:
        print(f"    ✗ Error during search: {str(e)}")
        return False


def scroll_results_panel(driver, max_scrolls=30, scroll_pause=2.5):
    """Automatically scroll the results panel to load ALL restaurants"""
    try:
        scrollable_selectors = [
            "div[role='feed']",
            "div.m6QErb[aria-label]",
            "div[aria-label*='Results']",
            "div.DxyBCb"
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

        scroll_count = 0
        last_height = driver.execute_script("return arguments[0].scrollHeight", scrollable_div)
        no_change_count = 0

        while scroll_count < max_scrolls:
            driver.execute_script(
                "arguments[0].scrollTo(0, arguments[0].scrollHeight);",
                scrollable_div
            )
            scroll_count += 1
            time.sleep(scroll_pause)

            new_height = driver.execute_script("return arguments[0].scrollHeight", scrollable_div)

            if new_height == last_height:
                no_change_count += 1
                if no_change_count >= 3:
                    break
            else:
                no_change_count = 0

            last_height = new_height

        return scroll_count

    except Exception as e:
        print(f"    ✗ Error during scrolling: {str(e)}")
        return 0


def collect_restaurant_urls(driver):
    """Collect all restaurant URLs from the current results"""
    try:
        # Multiple selector attempts for restaurant links
        link_selectors = [
            "a.hfpxzc",
            "a[href*='/maps/place/']",
            "div[role='button'][data-cid]"
        ]

        time.sleep(2)
        urls = []

        for selector in link_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    if selector == "a.hfpxzc" or selector == "a[href*='/maps/place/']":
                        url = element.get_attribute('href')
                        if url and url.startswith('https://www.google.com/maps/place/'):
                            urls.append(url)
                    elif selector == "div[role='button'][data-cid]":
                        # Get the URL from data-cid or click to get URL
                        try:
                            element.click()
                            time.sleep(1)
                            url = driver.current_url
                            if 'google.com/maps/place/' in url:
                                urls.append(url)
                        except:
                            pass

                if urls:
                    break
            except Exception:
                continue

        urls = list(set(urls))  # Remove duplicates
        return urls

    except Exception as e:
        print(f"    ✗ Error collecting URLs: {str(e)}")
        return []


def click_reviews_tab(driver):
    """Click the Reviews tab on restaurant page"""
    try:
        # Multiple selector attempts
        selectors = [
            "//div[contains(@class, 'Gpq6kf') and contains(text(), 'Reviews')]",
            "//button[contains(text(), 'Reviews')]",
            "//div[@role='button' and contains(., 'Reviews')]"
        ]

        for xpath in selectors:
            try:
                reviews_tab = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, xpath))
                )
                reviews_tab.click()
                time.sleep(2)
                return True
            except TimeoutException:
                continue

        return False

    except Exception as e:
        print(f"  ⚠ Error clicking reviews tab: {str(e)}")
        return False


def extract_cuisine(driver):
    """Extract cuisine/category information"""
    try:
        # Try multiple selectors for cuisine
        selectors = [
            "button.DkEaL",  # Main cuisine button
            "button[jsaction*='category']",
            "div.fontBodyMedium button",
            "span.fontBodyMedium"
        ]

        for selector in selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    text = element.text.strip()
                    # Look for restaurant type keywords
                    if any(keyword in text.lower() for keyword in
                           ['restaurant', 'sushi', 'pizza', 'mexican', 'chinese',
                            'italian', 'japanese', 'thai', 'indian', 'cafe', 'bar',
                            'burger', 'seafood', 'steakhouse', 'asian', 'korean',
                            'vietnamese', 'french', 'mediterranean', 'bbq', 'bakery']):
                        return text
            except Exception:
                continue

        return "N/A"

    except Exception:
        return "N/A"


def extract_description(driver):
    """Extract restaurant description/about section"""
    try:
        selectors = [
            ".PYvSYb",
            "div.Kzrk1b",
            "div[data-section-id='OgQvcd']",
            "span.wiI7pd"
        ]

        for selector in selectors:
            try:
                element = driver.find_element(By.CSS_SELECTOR, selector)
                if element and element.text:
                    return element.text.strip()
            except Exception:
                continue

        # Try to find description text in the header area
        try:
            header = driver.find_element(By.CSS_SELECTOR, "div[role='button'].rh8kZd")
            parent = header.find_element(By.XPATH, "./ancestor::div[contains(@class, 'JqKjRe')]")
            text = parent.text.strip()
            if len(text) > 20:
                return text
        except Exception:
            pass

        return "N/A"

    except Exception:
        return "N/A"


def extract_dining_options(driver):
    """Extract dining options (dine-in, takeout, delivery)"""
    try:
        dining_options = []

        # Find all dining option divs - they have class "LTs0Rc" with role="group"
        # The aria-label contains text like "Serves dine-in", "Offers takeout", "Has no-contact delivery"
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, "div.LTs0Rc[role='group']")

            for elem in elements:
                aria_label = elem.get_attribute('aria-label') or ""

                if aria_label:
                    # Extract dining option from aria-label
                    # Examples: "Serves dine-in" -> "Dine-in"
                    #           "Offers takeout" -> "Takeout"
                    #           "Has no-contact delivery" -> "No-contact delivery"

                    if 'dine-in' in aria_label.lower() and 'dine-in' not in [d.lower() for d in dining_options]:
                        dining_options.append('Dine-in')
                    elif 'dine in' in aria_label.lower() and 'dine-in' not in [d.lower() for d in dining_options]:
                        dining_options.append('Dine-in')

                    if 'takeout' in aria_label.lower() and 'takeout' not in [d.lower() for d in dining_options]:
                        dining_options.append('Takeout')

                    if 'no-contact delivery' in aria_label.lower() and 'no-contact delivery' not in [d.lower() for d in dining_options]:
                        dining_options.append('No-contact delivery')
                    elif 'delivery' in aria_label.lower() and 'delivery' not in [d.lower() for d in dining_options]:
                        if 'no-contact' not in aria_label.lower():  # Don't add generic "delivery" if it's specifically no-contact
                            dining_options.append('Delivery')
        except Exception:
            pass

        return "; ".join(dining_options) if dining_options else "N/A"

    except Exception:
        return "N/A"


def extract_thumbnail_image_url(driver):
    """Extract thumbnail image URL (not downloading, just getting the URL)"""
    try:
        # Try to find the main image from hero image button
        image_selectors = [
            "button.aoRNLd img",  # Hero image button
            "img[src*='lh3.googleusercontent.com']",  # Google hosted images
            "button.RZ66Rb img",
            "img[decoding='async']",
            "img.Lyrzac"
        ]

        for selector in image_selectors:
            try:
                img_element = driver.find_element(By.CSS_SELECTOR, selector)

                # Try to get src
                src = img_element.get_attribute('src')
                if src and src.startswith('http'):
                    # Google image URLs may have parameters, clean them up
                    # Extract the base URL and standard size parameters
                    if 'lh3.googleusercontent.com' in src:
                        # Return the URL as-is or with standard size
                        return src
                    elif 'googleusercontent' in src:
                        return src

                # Try to get data-src (lazy loaded images)
                data_src = img_element.get_attribute('data-src')
                if data_src and data_src.startswith('http'):
                    return data_src

            except Exception:
                continue

        return "N/A"

    except Exception:
        return "N/A"


def scrape_restaurant_details(driver, url):
    """Scrape detailed information for a single restaurant"""
    driver.get(url)
    time.sleep(3)

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

    # Enhanced cuisine extraction
    cuisine = extract_cuisine(driver)

    # Contact info - phone
    phone = "N/A"
    try:
        phone_button = driver.find_element(By.CSS_SELECTOR, "button[data-item-id^='phone:tel:']")
        phone_text = phone_button.find_element(By.CSS_SELECTOR, ".Io6YTe")
        phone = phone_text.text.strip() if phone_text.text else "N/A"
    except Exception:
        try:
            # Fallback: try to get phone from any button with phone data-item-id
            phone_button = driver.find_element(By.XPATH, "//button[contains(@data-item-id, 'phone:tel:')]")
            phone = phone_button.text.strip() if phone_button.text else "N/A"
        except Exception:
            pass

    website_element = wait_and_find(driver, "a[data-item-id^='authority']")
    website = website_element.get_attribute('href') if website_element else "N/A"

    # Hours - try multiple approaches
    hours = "N/A"
    try:
        # First try: Look for the button/div that contains hours with aria-label
        try:
            hours_button = driver.find_element(By.CSS_SELECTOR, "button[jsaction*='openhours']")
            aria_label = hours_button.get_attribute('aria-label')
            if aria_label and ('AM' in aria_label or 'PM' in aria_label):
                # Extract the readable hours part, remove "Copy open hours" or similar
                hours = re.sub(r",\s*(Copy|Show|Hide).*", "", aria_label).strip()
        except Exception:
            # Fallback: Try the div.t39EBf approach
            hours_element = driver.find_element(By.CSS_SELECTOR, "div.t39EBf")
            aria_label = hours_element.get_attribute('aria-label')
            if aria_label:
                hours_match = re.search(r"(.*?)\. (?:Hide|Show|Confirm)", aria_label)
                if hours_match:
                    hours = hours_match.group(1)
    except Exception:
        pass

    # Enhanced description extraction
    description = extract_description(driver)

    # Menu link
    menu_link = wait_and_find(driver, "a[data-item-id='menu']")
    menu = menu_link.get_attribute('href') if menu_link else "N/A"

    # Enhanced dining options extraction
    dining_options = extract_dining_options(driver)

    # Thumbnail image URL
    thumbnail_url = extract_thumbnail_image_url(driver)

    # Top commented words from reviews
    top_commented_words = "N/A"
    if click_reviews_tab(driver):
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.m6QErb"))
            )

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
        'dining_options': dining_options,
        'menu': menu,
        'thumbnail_image_url': thumbnail_url,
        'top_commented_words': top_commented_words,
        'url': url
    }


def run_enhanced_restaurant_scraper():
    """Main function to run enhanced restaurant discovery scraper"""
    print("\n" + "="*70)
    print("STAGE 1 ENHANCED: AUTOMATED RESTAURANT DISCOVERY")
    print("="*70)

    # Load keywords
    print(f"\nLoading search keywords from: {KEYWORDS_FILE}")
    keywords = load_search_keywords(KEYWORDS_FILE)

    if not keywords:
        print("\n✗ No keywords found. Please create the keywords file.")
        return None

    print(f"✓ Loaded {len(keywords)} search keywords\n")

    print("Search keywords to process:")
    for i, keyword in enumerate(keywords[:10], 1):
        print(f"  {i}. {keyword}")
    if len(keywords) > 10:
        print(f"  ... and {len(keywords) - 10} more keywords")

    print(f"\nConfiguration:")
    print(f"  Max scrolls per search: {MAX_SCROLLS}")
    print(f"  Scroll pause: {SCROLL_PAUSE}s")
    print(f"  Delay between searches: {BETWEEN_SEARCHES_DELAY}s")
    print(f"  Image storage: {IMAGES_FOLDER}")

    input("\nPress Enter to start enhanced scraping...")

    # Initialize browser
    print("\n" + "="*70)
    print("Opening Chrome browser...")
    print("="*70)
    driver = webdriver.Chrome()
    driver.maximize_window()

    all_urls = []

    try:
        # Phase 1: Collect URLs
        print("\n" + "="*70)
        print("PHASE 1: COLLECTING RESTAURANT URLs")
        print("="*70)

        for i, keyword in enumerate(keywords, 1):
            print(f"\n[{i}/{len(keywords)}] Processing: '{keyword}'")

            if not search_google_maps(driver, keyword):
                print(f"  ✗ Search failed, skipping...")
                continue

            print(f"  Scrolling to load all results...")
            scrolls = scroll_results_panel(driver, max_scrolls=MAX_SCROLLS, scroll_pause=SCROLL_PAUSE)
            print(f"  ✓ Scrolled {scrolls} times")

            print(f"  Collecting URLs...")
            urls = collect_restaurant_urls(driver)

            if urls:
                print(f"  ✓ Collected {len(urls)} URLs")
                all_urls.extend(urls)
            else:
                print(f"  ⚠ No URLs collected")

            if i < len(keywords):
                time.sleep(BETWEEN_SEARCHES_DELAY)

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
        print("PHASE 2: SCRAPING RESTAURANT DETAILS (ENHANCED)")
        print("="*70)
        print(f"Scraping detailed information for {len(unique_urls)} restaurants...\n")

        scraped_data = []
        for index, url in enumerate(unique_urls, 1):
            print(f"[{index}/{len(unique_urls)}] Scraping restaurant...")

            try:
                data = scrape_restaurant_details(driver, url)
                scraped_data.append(data)
                print(f"  ✓ {data['name']}")
                print(f"    - Cuisine: {data['cuisine']}")
                print(f"    - Dining Options: {data['dining_options']}")
                if data['thumbnail_image_url'] != "N/A":
                    print(f"    - Image: ✓ Found")

                if index < len(unique_urls):
                    delay = get_random_delay('between_restaurants')
                    print(f"  Waiting {delay:.1f}s...\n")
                    time.sleep(delay)

            except Exception as e:
                print(f"  ✗ Error: {str(e)}\n")
                continue

        # Save to CSV
        output_file = PATHS['stage1_restaurants']
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'restaurant_id', 'name', 'address', 'rating', 'reviews_count',
                'price_range', 'cuisine', 'phone', 'website', 'hours',
                'description', 'dining_options', 'menu', 'thumbnail_image_url',
                'top_commented_words', 'url'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for data in scraped_data:
                data['hours'] = format_hours(data['hours'])
                writer.writerow(data)

        print("\n" + "="*70)
        print("✓ STAGE 1 ENHANCED COMPLETE!")
        print("="*70)
        print(f"\nData saved to: {output_file}")
        print(f"Total restaurants scraped: {len(scraped_data)}")
        print(f"Unique URLs found: {len(unique_urls)}")
        print(f"Keywords processed: {len(keywords)}")

        return output_file

    except KeyboardInterrupt:
        print("\n\n⚠ Scraping interrupted by user")
        return None

    finally:
        driver.quit()


def scrape_restaurants_by_urls(urls, output_csv_path=None):
    """
    Scrape specific restaurants by their Google Maps URLs.
    More efficient when you already have filtered URLs.
    """
    if output_csv_path is None:
        output_csv_path = PATHS['stage1_restaurants']

    print("\n" + "="*70)
    print("SCRAPING RESTAURANTS BY URL (FILTERED)")
    print("="*70)
    print(f"Will scrape {len(urls)} restaurants...\n")

    # Initialize driver
    options = webdriver.ChromeOptions()
    options.add_argument('--start-maximized')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    driver = webdriver.Chrome(options=options)

    try:
        scraped_data = []

        for index, url in enumerate(urls, 1):
            print(f"[{index}/{len(urls)}] Scraping restaurant...")

            try:
                data = scrape_restaurant_details(driver, url)
                scraped_data.append(data)
                print(f"  ✓ {data['name']}")
                print(f"    - Cuisine: {data['cuisine']}")
                print(f"    - Hours: {data['hours']}")
                print(f"    - Dining: {data['dining_options']}")
                print(f"    - Image: {'✓' if data['thumbnail_image_url'] != 'N/A' else '✗'}")

                if index < len(urls):
                    delay = get_random_delay('between_restaurants')
                    print(f"  Waiting {delay:.1f}s...\n")
                    time.sleep(delay)

            except Exception as e:
                print(f"  ✗ Error: {str(e)}\n")
                continue

        # Save to CSV
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'restaurant_id', 'name', 'address', 'rating', 'reviews_count',
                'price_range', 'cuisine', 'phone', 'website', 'hours',
                'description', 'dining_options', 'menu', 'thumbnail_image_url',
                'top_commented_words', 'url'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for data in scraped_data:
                data['hours'] = format_hours(data['hours'])
                writer.writerow(data)

        print("\n" + "="*70)
        print("✓ SCRAPING COMPLETE!")
        print("="*70)
        print(f"\nData saved to: {output_csv_path}")
        print(f"Total restaurants scraped: {len(scraped_data)}/{len(urls)}")
        print(f"Success rate: {len(scraped_data)/len(urls)*100:.1f}%")

        return output_csv_path

    except KeyboardInterrupt:
        print("\n\n⚠ Scraping interrupted by user")
        return None

    finally:
        driver.quit()


if __name__ == "__main__":
    run_enhanced_restaurant_scraper()
