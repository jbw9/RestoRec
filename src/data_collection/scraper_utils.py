"""
Shared utility functions for web scraping
"""

import time
import re
import hashlib
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from .config import ANTI_DETECTION, get_random_delay


def wait_and_find(driver, selector, timeout=None, multiple=False):
    """
    Wait for element(s) to be present and return them

    Args:
        driver: Selenium WebDriver instance
        selector: CSS selector string
        timeout: Timeout in seconds (uses config default if None)
        multiple: If True, return list of elements; if False, return single element

    Returns:
        Element, list of elements, or None/[] if not found
    """
    if timeout is None:
        timeout = ANTI_DETECTION['timeouts']['page_load']

    try:
        if multiple:
            elements = WebDriverWait(driver, timeout).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector))
            )
            return elements
        else:
            element = WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
            )
            return element
    except TimeoutException:
        return [] if multiple else None


def generate_restaurant_id(url):
    """
    Generate unique restaurant ID from Google Maps URL

    Args:
        url: Google Maps URL

    Returns:
        12-character unique ID string
    """
    # Try to extract place_id from URL
    match = re.search(r'!1s(0x[0-9a-f:]+)', url)
    if match:
        place_id = match.group(1)
        return hashlib.md5(place_id.encode()).hexdigest()[:12]

    # Fallback to URL hash
    return hashlib.md5(url.encode()).hexdigest()[:12]


def normalize_restaurant_name(name):
    """
    Normalize restaurant name for matching

    Args:
        name: Restaurant name string

    Returns:
        Normalized name string
    """
    name = name.lower()
    # Remove common suffixes
    name = re.sub(r'\s+(restaurant|cafe|bar|grill|kitchen|diner|bistro|eatery)$', '', name)
    # Remove punctuation
    name = re.sub(r'[^\w\s]', '', name)
    # Remove extra whitespace
    name = ' '.join(name.split())
    return name.strip()


def scroll_element(driver, element, max_scrolls=None):
    """
    Scroll an element to load all content

    Args:
        driver: Selenium WebDriver instance
        element: Element to scroll (or CSS selector string)
        max_scrolls: Maximum number of scroll attempts (uses config default if None)

    Returns:
        Number of scrolls performed
    """
    if max_scrolls is None:
        max_scrolls = ANTI_DETECTION['timeouts']['page_load']

    # If element is a string, find it first
    if isinstance(element, str):
        element = wait_and_find(driver, element)
        if not element:
            print(f"Could not find scrollable element: {element}")
            return 0

    scroll_count = 0
    last_height = driver.execute_script("return arguments[0].scrollHeight", element)

    while scroll_count < max_scrolls:
        # Scroll to bottom
        driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight;", element)

        # Random delay
        delay = get_random_delay('between_scrolls')
        time.sleep(delay)

        # Check if we've reached the bottom
        new_height = driver.execute_script("return arguments[0].scrollHeight", element)

        if new_height == last_height:
            print(f"Reached bottom after {scroll_count + 1} scrolls")
            break

        last_height = new_height
        scroll_count += 1

    return scroll_count


def expand_all_text(driver, button_selector="button.w8nwRe.kyuRq", max_attempts=3):
    """
    Click all 'More' buttons to expand truncated text

    Args:
        driver: Selenium WebDriver instance
        button_selector: CSS selector for 'More' buttons
        max_attempts: Maximum number of expansion attempts

    Returns:
        Total number of buttons clicked
    """
    total_clicked = 0

    for attempt in range(max_attempts):
        more_buttons = driver.find_elements(By.CSS_SELECTOR, button_selector)
        clicked = 0

        for button in more_buttons:
            try:
                if button.is_displayed():
                    driver.execute_script("arguments[0].click();", button)
                    clicked += 1
                    total_clicked += 1

                    # Small delay between clicks
                    delay = get_random_delay('between_clicks')
                    time.sleep(delay)
            except Exception:
                continue

        if clicked == 0:
            break

        # Small delay before next attempt
        time.sleep(1)

    return total_clicked


def extract_rating_from_stars(review_element):
    """
    Extract rating from aria-label attribute

    Args:
        review_element: Selenium element containing the review

    Returns:
        Integer rating (1-5) or None if not found
    """
    try:
        # Find the rating container with aria-label
        rating_span = review_element.find_element(By.CSS_SELECTOR, "span.kvMYJc[role='img']")

        if rating_span:
            # Get the aria-label attribute (e.g., "5 stars", "2 stars")
            aria_label = rating_span.get_attribute('aria-label')

            if aria_label:
                # Extract the number from "X stars" format
                match = re.match(r'(\d+)\s*star', aria_label, re.IGNORECASE)
                if match:
                    return int(match.group(1))

        return None
    except Exception:
        return None


def has_review_photos(review_element):
    """
    Check if review has photos attached

    Args:
        review_element: Selenium element containing the review

    Returns:
        Boolean - True if review has photos, False otherwise
    """
    try:
        # Look for photo/image buttons in the review
        photo_buttons = review_element.find_elements(By.CSS_SELECTOR, "button.Tya61d")
        return len(photo_buttons) > 0
    except Exception:
        return False


def extract_review_metadata(review_element):
    """
    Extract additional review metadata (dining type, ratings, etc.)

    Args:
        review_element: Selenium element containing the review

    Returns:
        Dictionary of metadata
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
        detail_spans = review_element.find_elements(By.CSS_SELECTOR, "span.RfDO5c")

        for span in detail_spans:
            text = span.text.strip()

            if "Dine in" in text or "Delivery" in text or "Takeout" in text:
                metadata['dining_type'] = text
            elif "$" in text and "–" in text:
                metadata['price_range'] = text
            elif "Food:" in text:
                metadata['food_rating'] = text.split(':')[1].strip()
            elif "Service:" in text:
                metadata['service_rating'] = text.split(':')[1].strip()
            elif "Atmosphere:" in text:
                metadata['atmosphere_rating'] = text.split(':')[1].strip()
            elif not any(keyword in text.lower() for keyword in
                        ['food:', 'service:', 'atmosphere:', 'parking', 'recommend', '$']):
                # Likely recommended dishes
                if metadata['recommended_dishes']:
                    metadata['recommended_dishes'] += "; " + text
                else:
                    metadata['recommended_dishes'] = text

    except Exception as e:
        print(f"Error extracting review metadata: {str(e)}")

    return metadata


def safe_get_text(element, selector, default="N/A"):
    """
    Safely get text from element using selector

    Args:
        element: Parent element
        selector: CSS selector string
        default: Default value if element not found

    Returns:
        Text content or default value
    """
    try:
        sub_element = element.find_element(By.CSS_SELECTOR, selector)
        return sub_element.text if sub_element.text else default
    except (NoSuchElementException, Exception):
        return default


def safe_get_attribute(element, selector, attribute, default="N/A"):
    """
    Safely get attribute from element using selector

    Args:
        element: Parent element
        selector: CSS selector string
        attribute: Attribute name to retrieve
        default: Default value if element/attribute not found

    Returns:
        Attribute value or default value
    """
    try:
        sub_element = element.find_element(By.CSS_SELECTOR, selector)
        value = sub_element.get_attribute(attribute)
        return value if value else default
    except (NoSuchElementException, Exception):
        return default


def format_hours(hours):
    """
    Format hours dictionary or string for CSV storage

    Args:
        hours: Hours data (dict or string)

    Returns:
        Formatted string
    """
    if isinstance(hours, dict):
        return '; '.join([f"{day}: {time}" for day, time in hours.items()])
    return hours if hours else "N/A"


def create_review_hash(user_id, restaurant_id, review_text):
    """
    Create unique hash for review deduplication

    Args:
        user_id: User ID
        restaurant_id: Restaurant ID
        review_text: Review text (first 100 chars used)

    Returns:
        MD5 hash string
    """
    text_sample = review_text[:100] if review_text else ''
    unique_string = f"{user_id}:{restaurant_id}:{text_sample}"
    return hashlib.md5(unique_string.encode()).hexdigest()


def print_progress(current, total, prefix='Progress', bar_length=50):
    """
    Print progress bar

    Args:
        current: Current progress
        total: Total items
        prefix: Prefix text
        bar_length: Length of progress bar
    """
    percent = 100 * (current / float(total))
    filled = int(bar_length * current // total)
    bar = '█' * filled + '-' * (bar_length - filled)

    print(f'\r{prefix}: |{bar}| {percent:.1f}% ({current}/{total})', end='', flush=True)

    if current == total:
        print()  # New line when complete
