"""
TEST SCRIPT: Automated Stage 1 - Restaurant Discovery with Multiple Keywords

This script reads search keywords from 'search_keywords.txt' and automatically:
1. Searches Google Maps for each keyword
2. Scrolls through all results
3. Collects restaurant URLs
4. Deduplicates and saves final list

Usage:
    python test_automated_stage1.py
"""

import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


# Configuration
KEYWORDS_FILE = "search_keywords.txt"
MAX_SCROLLS = 30  # Increased to ensure we get all results
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
        print("  Please create the file with search keywords (one per line)")
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
    print(f"\n  Searching for: '{search_query}'")

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
            print(f"    Scroll #{scroll_count}", end='')

            # Wait for new results to load
            time.sleep(scroll_pause)

            # Check if new content loaded
            new_height = driver.execute_script("return arguments[0].scrollHeight", scrollable_div)

            if new_height == last_height:
                no_change_count += 1
                print(f" (no new results - {no_change_count}/3)")

                # If height hasn't changed for 3 consecutive scrolls, we're done
                if no_change_count >= 3:
                    print(f"    ✓ Reached bottom (no new results after 3 attempts)")
                    break
            else:
                no_change_count = 0  # Reset counter
                print(" (loaded more results)")

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


def main():
    """Main test function"""
    print("\n" + "="*70)
    print("TEST: AUTOMATED STAGE 1 - MULTI-KEYWORD RESTAURANT DISCOVERY")
    print("="*70)

    # Load keywords
    print(f"\nLoading search keywords from: {KEYWORDS_FILE}")
    keywords = load_search_keywords(KEYWORDS_FILE)

    if not keywords:
        print("\n✗ No keywords found. Please check the keywords file.")
        return

    print(f"✓ Loaded {len(keywords)} search keywords\n")

    # Show keywords
    print("Search keywords to process:")
    for i, keyword in enumerate(keywords, 1):
        print(f"  {i}. {keyword}")

    print(f"\nConfiguration:")
    print(f"  Max scrolls per search: {MAX_SCROLLS}")
    print(f"  Scroll pause: {SCROLL_PAUSE}s")
    print(f"  Delay between searches: {BETWEEN_SEARCHES_DELAY}s")

    input("\nPress Enter to start automated scraping...")

    # Initialize browser
    print("\n" + "="*70)
    print("Opening Chrome browser...")
    print("="*70)
    driver = webdriver.Chrome()
    driver.maximize_window()

    all_urls = []  # Collect all URLs from all searches

    try:
        for i, keyword in enumerate(keywords, 1):
            print(f"\n{'='*70}")
            print(f"[{i}/{len(keywords)}] Processing: '{keyword}'")
            print("="*70)

            # Step 1: Search
            if not search_google_maps(driver, keyword):
                print(f"  ✗ Search failed for '{keyword}', skipping...")
                continue

            # Step 2: Scroll
            print(f"  Scrolling to load all results...")
            scrolls = scroll_results_panel(driver, max_scrolls=MAX_SCROLLS, scroll_pause=SCROLL_PAUSE)

            # Step 3: Collect URLs
            print(f"  Collecting restaurant URLs...")
            urls = collect_restaurant_urls(driver)

            if urls:
                print(f"  ✓ Collected {len(urls)} URLs from this search")
                all_urls.extend(urls)
            else:
                print(f"  ⚠ No URLs collected from this search")

            # Delay between searches (anti-detection)
            if i < len(keywords):
                print(f"  Waiting {BETWEEN_SEARCHES_DELAY}s before next search...")
                time.sleep(BETWEEN_SEARCHES_DELAY)

        # Deduplicate all URLs
        print(f"\n{'='*70}")
        print("PROCESSING RESULTS")
        print("="*70)

        unique_urls = list(set(all_urls))

        print(f"\nTotal URLs collected (with duplicates): {len(all_urls)}")
        print(f"Unique restaurants: {len(unique_urls)}")
        print(f"Duplicates removed: {len(all_urls) - len(unique_urls)}")

        if unique_urls:
            print(f"\n✓ SUCCESS! Found {len(unique_urls)} unique restaurants")

            # Show first 15 as sample
            print(f"\nSample restaurants (first 15):")
            for i, url in enumerate(unique_urls[:15], 1):
                try:
                    name_part = url.split('/place/')[1].split('/data')[0]
                    import urllib.parse
                    name = urllib.parse.unquote(name_part).replace('+', ' ')
                    print(f"  {i}. {name}")
                except:
                    print(f"  {i}. {url[:60]}...")

            if len(unique_urls) > 15:
                print(f"\n  ... and {len(unique_urls) - 15} more restaurants")

            # Save to file
            output_file = "test_restaurant_urls.txt"
            with open(output_file, 'w') as f:
                f.write(f"# Collected {len(unique_urls)} unique restaurants\n")
                f.write(f"# From {len(keywords)} different searches\n")
                f.write(f"# Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                for url in unique_urls:
                    f.write(url + '\n')

            print(f"\n✓ All URLs saved to: {output_file}")

            # Estimate for full Stage 1
            print(f"\n{'='*70}")
            print("NEXT STEPS")
            print("="*70)
            print(f"\nYou now have {len(unique_urls)} unique restaurant URLs!")
            print(f"\nTo proceed:")
            print(f"  1. Review the URLs in '{output_file}'")
            print(f"  2. Edit '{KEYWORDS_FILE}' to add/remove search terms")
            print(f"  3. Integrate this into main Stage 1 scraper")
            print(f"\nEstimated time to scrape details for {len(unique_urls)} restaurants:")
            print(f"  ~{len(unique_urls) * 0.3:.0f}-{len(unique_urls) * 0.5:.0f} minutes")

        else:
            print("\n✗ FAILED: No URLs collected from any search")

        print(f"\n{'='*70}")
        print("Browser will stay open for 10 seconds for you to inspect...")
        print("="*70)
        time.sleep(10)

    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
        print(f"URLs collected so far: {len(set(all_urls))}")

    except Exception as e:
        print(f"\n✗ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        print("\nClosing browser...")
        driver.quit()
        print("✓ Test complete!")


if __name__ == "__main__":
    main()
