# RestoRec: Champaign-Urbana Restaurant Recommendation System

A hybrid recommendation system combining collaborative filtering and content-based filtering for restaurants in the Champaign-Urbana area.

## 🎯 Project Overview

**Goal**: Build a neural network-based hybrid recommendation system trained on real user-restaurant rating data from Google Maps, enhanced with Yelp metadata.

**Key Features**:
- **Option B Data Collection**: Complete user preference mapping via restaurant pages + user profile scraping
- **Fuzzy Matching**: Automatically filters for CU restaurants when scraping user profiles
- **Resume Capability**: All stages support stopping and resuming without data loss
- **Anti-Detection**: Built-in delays, batching, and breaks to avoid scraping blocks
- **Hybrid Model**: Combines collaborative filtering (user-restaurant embeddings) + content features (categories, price, location)

## 📁 Project Structure

```
RestoRec/
├── README.md                          # This file
├── requirements.txt                    # Python dependencies
├── run_all_stages.py                  # ⭐ MASTER SCRIPT - Run this!
├── CU_Restaurant_Recommender_Plan.md  # Full implementation plan
│
├── src/
│   ├── data_collection/
│   │   ├── config.py                  # Scraping configuration
│   │   ├── scraper_utils.py           # Shared utilities
│   │   ├── restaurant_scraper.py      # Stage 1: Restaurant discovery
│   │   ├── restaurant_review_scraper.py  # Stage 2: Review scraping
│   │   └── user_profile_scraper.py    # Stage 3: User profile scraping
│   │
│   └── data_processing/
│       └── merge_reviews.py           # Stage 4: Merge & deduplicate
│
├── scripts/
│   └── extract_users_for_scraping.py  # Extract users for Stage 3
│
└── data/
    ├── raw/                           # Raw scraped data
    │   ├── stage1_restaurants.csv
    │   ├── stage2_restaurant_reviews.csv
    │   ├── stage2_user_mapping.csv
    │   ├── stage3_user_profile_reviews.csv
    │   └── stage4_merged_reviews.csv  # ⭐ FINAL DATASET
    │
    └── processed/                     # Processed data (for future phases)
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd RestoRec

# Install dependencies
pip install -r requirements.txt
```

**Note**: You'll need Chrome browser and ChromeDriver installed for Selenium.

### 2. Run Data Collection Pipeline

**Option A: FULLY AUTOMATIC MODE (recommended!)**

```bash
python run_all_stages.py --auto
```

This runs ALL 4 stages back-to-back without any prompts. Perfect for overnight runs!
- Stage 1: Reads keywords from `search_keywords.txt` and scrapes automatically
- Stage 2-4: All run sequentially without user input
- Just start it and walk away!

**Option B: Interactive mode (with prompts between stages)**

```bash
python run_all_stages.py
```

This will guide you through all 4 stages with confirmation prompts between each stage.

**Option C: Run individual stages**

```bash
# Stage 1: Restaurant Discovery
python run_all_stages.py --stage 1

# Stage 2: Restaurant Review Scraping
python run_all_stages.py --stage 2 --start 0 --end 100

# Stage 3: User Profile Scraping
python run_all_stages.py --stage 3 --start 0 --end 50

# Stage 4: Merge & Deduplicate
python run_all_stages.py --stage 4
```

## 📋 Data Collection Stages

### Stage 1: Restaurant Discovery (FULLY AUTOMATED!)
**What it does**: Automatically collects restaurant URLs from Google Maps using keywords

**How it works**:
1. Reads search keywords from `search_keywords.txt` (100+ pre-loaded keywords!)
2. For each keyword:
   - Automatically searches Google Maps
   - Automatically scrolls through all results
   - Collects all restaurant URLs
3. Deduplicates across all searches
4. Scrapes detailed info for each unique restaurant

**Customize**: Edit `search_keywords.txt` to add/remove search terms

**Output**: `data/raw/stage1_restaurants.csv`

**Expected**: 200-500+ unique restaurants (with 100+ keywords)

**Time**: ~2-4 hours (depends on number of keywords and restaurants found)

---

### Stage 2: Restaurant Review Scraping
**What it does**: Scrapes all reviews from each restaurant page

**How it works**:
1. Visits each restaurant URL from Stage 1
2. Clicks Reviews tab
3. Scrolls to load all reviews
4. Extracts: user_id, rating, review_text, date, metadata
5. **Builds user-restaurant rating matrix**
6. Tracks user IDs across all restaurants

**Output**:
- `data/raw/stage2_restaurant_reviews.csv` (~5,000-10,000 reviews)
- `data/raw/stage2_user_mapping.csv` (user ID mappings)

**Time**: ~4-6 hours (for 100 restaurants with delays)

---

### Stage 3: User Profile Scraping (The Secret Sauce!)
**What it does**: Visits each user's profile to get ALL their CU restaurant reviews

**How it works**:
1. Extracts unique users from Stage 2
2. Visits each user's Google Maps profile
3. Scrolls through their reviews
4. **Fuzzy matches restaurant names** to filter ONLY CU restaurants
5. Extracts all CU reviews (often 10-20 per prolific user!)

**Output**: `data/raw/stage3_user_profile_reviews.csv` (~3,000-8,000 additional reviews)

**Why this matters**:
- Stage 2 only gets "top reviews" shown on restaurant pages
- Stage 3 gets the COMPLETE user preference history
- **Result**: ~50-100% more data per user!

**Time**: ~10-20 hours (for 500-1000 users with delays)

**Pro tip**: Run in batches of 50-100 users at a time

---

### Stage 4: Merge & Deduplicate
**What it does**: Combines Stage 2 + Stage 3, removes duplicates

**How it works**:
1. Loads both datasets
2. Creates hash for each review (user + restaurant + text sample)
3. Removes duplicates (keeps first occurrence from Stage 2)
4. Generates statistics

**Output**: `data/raw/stage4_merged_reviews.csv` (FINAL DATASET)

**Expected results**:
- ~7,000-15,000 unique reviews
- 500-1,000 unique users
- 100 unique restaurants
- Average 10-15 reviews per user (vs 5-10 without Stage 3!)
- Matrix sparsity: 85-90%

**Time**: ~1 minute

---

## 🔧 Configuration

Edit `src/data_collection/config.py` to customize:

**Anti-detection settings**:
```python
'delays': {
    'between_restaurants': (5, 15),  # Random 5-15 sec delay
    'between_users': (3, 8),         # Random 3-8 sec delay
    'batch_break': 120,              # 2-min break every batch
}
```

**Batch sizes**:
```python
'batch_sizes': {
    'restaurants': 20,  # Take break after 20 restaurants
    'users': 50,        # Take break after 50 users
}
```

**Fuzzy matching**:
```python
FUZZY_MATCHING = {
    'name_match_threshold': 85,  # Minimum similarity (0-100)
    'name_weight': 0.7,          # Weight for name matching
    'address_weight': 0.3,       # Weight for address matching
}
```

## 📊 Expected Data Output

After running all stages, you should have:

```
Final Dataset (stage4_merged_reviews.csv)
├─ 7,000-15,000 unique reviews
├─ 500-1,000 unique users
├─ 100 unique restaurants
├─ Columns:
│   ├─ restaurant_id           # Unique restaurant ID
│   ├─ restaurant_name          # Restaurant name
│   ├─ user_id                  # Unique user ID
│   ├─ rating                   # 1-5 stars ⭐ TARGET VARIABLE
│   ├─ review_date              # Date of review
│   ├─ review_text              # Full review text
│   ├─ dining_type              # Dine-in/Delivery/Takeout
│   ├─ food_rating              # 1-5 (if available)
│   ├─ service_rating           # 1-5 (if available)
│   ├─ atmosphere_rating        # 1-5 (if available)
│   ├─ recommended_dishes       # Dishes mentioned
│   └─ source                   # 'restaurant_page' or 'user_profile'
```

This data is READY for model training!

## 🎓 Next Steps (After Data Collection)

See `CU_Restaurant_Recommender_Plan.md` for full details:

1. **Phase 2: Data Processing** (Week 2)
   - Clean data, handle missing values
   - Merge with Yelp API metadata
   - Feature engineering (categories, price, location)
   - Train/val/test splits

2. **Phase 3: Baseline Models** (Week 2)
   - Pure Collaborative Filtering
   - Pure Content-Based

3. **Phase 4: Hybrid Neural Network** (Week 3-4)
   - User/restaurant embeddings + content features
   - PyTorch neural network
   - Hyperparameter tuning

4. **Phase 5: Evaluation** (Week 4)
   - RMSE, MAE metrics
   - Model comparison

5. **Phase 6: Inference System** (Week 5)
   - Recommendation engine
   - Cold start handler
   - Demo notebook

## ⚠️ Important Notes

### Ethical & Legal Considerations

**⚠️ WARNING**: Web scraping Google Maps may violate their Terms of Service. This project is for **educational purposes only**. Use at your own risk.

**Anti-Detection Best Practices**:
- Use random delays (already built-in)
- Process in batches with breaks (already built-in)
- Scrape during off-peak hours (1-6 AM or 2-4 PM)
- Don't run scraper 24/7
- Consider using residential proxies for large-scale scraping

### Resume Capability

All stages support resuming:
- **Stage 1**: Re-run to add more restaurants (no duplicates)
- **Stage 2**: Tracks processed restaurants in `stage2_processed_restaurants.txt`
- **Stage 3**: Tracks processed users in `stage3_processed_users.txt`
- **Stage 4**: Can re-run anytime to re-merge

If scraping is interrupted, just run the same command again!

### Troubleshooting

**"ChromeDriver not found"**:
```bash
# Install ChromeDriver (macOS)
brew install chromedriver

# Or download from:
# https://chromedriver.chromium.org/
```

**"Element not found" errors**:
- Google Maps layout may have changed
- Check CSS selectors in scraper files
- Try increasing timeouts in `config.py`

**Low review counts**:
- Some restaurants have few reviews
- Try scraping more restaurants in Stage 1
- Focus on popular restaurants

**Fuzzy matching not working**:
- Adjust `name_match_threshold` in `config.py`
- Some restaurant names vary significantly (e.g., "Kams" vs "Kam's")
- Check `stage1_restaurants.csv` for correct names

## 📝 Example Commands

```bash
# Run all stages (guided)
python run_all_stages.py

# Run just Stage 2, rows 0-50
python run_all_stages.py --stage 2 --start 0 --end 50

# Run just Stage 3, first 100 users
python run_all_stages.py --stage 3 --start 0 --end 100

# Re-merge data after additional Stage 3 scraping
python run_all_stages.py --stage 4
```

## 🤝 Contributing

This is a personal educational project. See `CU_Restaurant_Recommender_Plan.md` for the full roadmap.

## 📄 License

MIT License - See full plan for details

## 🙏 Acknowledgments

- Adapted from previous Google Maps scraping projects
- Uses Selenium, Pandas, FuzzyWuzzy
- Inspired by hybrid recommender system research

---

**Ready to build your restaurant recommendation system?**

**For fully automatic mode (recommended - just start and walk away!):**

```bash
python run_all_stages.py --auto
```

**Or for interactive mode (with prompts between stages):**

```bash
python run_all_stages.py
```

**Good luck! 🚀**
