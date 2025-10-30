"""
Configuration for web scraping with anti-detection measures
"""

import random

# Anti-detection configuration
ANTI_DETECTION = {
    'delays': {
        'between_restaurants': (5, 15),  # Random delay between scraping restaurants (seconds)
        'between_users': (3, 8),         # Random delay between scraping user profiles (seconds)
        'between_scrolls': (1, 3),       # Random delay between scroll actions (seconds)
        'between_clicks': (0.1, 0.3),    # Random delay between clicks (seconds)
        'batch_break': 120,              # Break duration after batch completion (seconds)
    },

    'batch_sizes': {
        'restaurants': 20,  # Process this many restaurants before taking a break
        'users': 50,        # Process this many users before taking a break
    },

    'user_agents': [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
    ],

    'timeouts': {
        'page_load': 10,      # Timeout for page elements to load (seconds)
        'scroll_wait': 2,     # Wait time after scroll (seconds)
        'click_wait': 0.5,    # Wait time after click (seconds)
    },

    'retry': {
        'max_retries': 3,     # Maximum retry attempts for failed operations
        'retry_delay': 5,     # Delay before retry (seconds)
    }
}

# Scraping limits
SCRAPING_LIMITS = {
    'max_restaurants': 100,           # Maximum restaurants to scrape
    'max_reviews_per_restaurant': 50, # Maximum reviews to collect per restaurant (set to None for unlimited)
    'max_scrolls': 30,                # Maximum scroll attempts
    'max_users_to_scrape': None,      # Maximum user profiles to scrape (None = all discovered users)
}

# Fuzzy matching configuration
FUZZY_MATCHING = {
    'name_match_threshold': 85,   # Minimum similarity score (0-100) for restaurant name matching
    'name_weight': 0.7,           # Weight for name similarity in combined score
    'address_weight': 0.3,        # Weight for address similarity in combined score
}

# File paths
PATHS = {
    'data_raw': 'data/raw',
    'data_processed': 'data/processed',
    'stage1_restaurants': 'data/raw/stage1_restaurants.csv',
    'stage2_reviews': 'data/raw/stage2_restaurant_reviews.csv',
    'stage2_user_mapping': 'data/raw/stage2_user_mapping.csv',
    'stage2_processed_restaurants': 'data/raw/stage2_processed_restaurants.txt',
    'users_to_scrape': 'data/raw/users_to_scrape.csv',
    'stage3_reviews': 'data/raw/stage3_user_profile_reviews.csv',
    'stage3_processed_users': 'data/raw/stage3_processed_users.txt',
    'stage4_merged': 'data/raw/stage4_merged_reviews.csv',
}

# Location configuration
LOCATION = {
    'search_query': 'restaurants champaign urbana illinois',
    'city': 'Champaign',
    'state': 'IL',
    'campus_coords': {
        'latitude': 40.1092,   # UIUC Union
        'longitude': -88.2273,
    }
}


def get_random_delay(delay_type):
    """
    Get random delay for anti-detection

    Args:
        delay_type: Type of delay from ANTI_DETECTION['delays']

    Returns:
        Random float delay in seconds
    """
    if delay_type not in ANTI_DETECTION['delays']:
        raise ValueError(f"Unknown delay type: {delay_type}")

    min_delay, max_delay = ANTI_DETECTION['delays'][delay_type]
    return random.uniform(min_delay, max_delay)


def get_random_user_agent():
    """
    Get random user agent for anti-detection

    Returns:
        Random user agent string
    """
    return random.choice(ANTI_DETECTION['user_agents'])


def get_batch_size(batch_type):
    """
    Get batch size for processing

    Args:
        batch_type: Type of batch from ANTI_DETECTION['batch_sizes']

    Returns:
        Batch size integer
    """
    if batch_type not in ANTI_DETECTION['batch_sizes']:
        raise ValueError(f"Unknown batch type: {batch_type}")

    return ANTI_DETECTION['batch_sizes'][batch_type]
