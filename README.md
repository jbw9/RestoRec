# RestoRec: Champaign-Urbana Restaurant Recommendation System

A collaborative filtering recommendation system for restaurants in the Champaign-Urbana area, built on real user-restaurant rating data scraped from Google Maps.

## 🎯 Project Overview

**Goal**: Build a recommendation engine using collaborative filtering to predict user ratings and recommend restaurants.

**Current Status**: ✅ **Data Collection & Processing Complete**
- 768 restaurants scraped with complete details
- 63,888 reviews extracted
- 37,732 unique reviewers identified
- User-restaurant preference matrix constructed (99.76% sparsity)

## 📊 Final Dataset

| Metric | Value |
|--------|-------|
| **Restaurants** | 710 (filtered to 10+ reviews) |
| **Reviews** | 63,888 |
| **Unique Reviewers** | 37,732 |
| **Preference Matrix** | 37,732 × 710 |
| **Dense Matrix** | 1,871 × 710 (reviewers with 5+ reviews) |
| **Top Reviewer** | Ben Brenner (111 reviews) |
| **Top Restaurant** | Bobo's Barbecue (110 reviews) |

## 🏗️ Pipeline Architecture

The system uses a 3-stage automated pipeline:

### Stage 1: Restaurant Discovery & Scraping
- **Purpose**: Discover and scrape restaurant details from Google Maps
- **Input**: Existing restaurant list (874 restaurants)
- **Output**: `stage1_restaurants.csv` (768 restaurants with 10+ reviews)
- **Details Extracted per Restaurant**:
  - Basic info: name, address, rating, review count
  - Details: cuisine, dining options, hours, phone, website
  - Images & menus

**Key Features**:
- Filters restaurants by minimum review count (10+)
- Fixed CSS selectors for reliable cuisine/hours/dining_options extraction
- Handles dynamic Google Maps content
- Resume-capable (can continue from interruption)

### Stage 2: Review & Reviewer URL Extraction
- **Purpose**: Extract all reviews and identify unique reviewers
- **Input**: `stage1_restaurants.csv` (768 restaurants)
- **Output**:
  - `stage2_restaurant_reviews.csv` (63,888 reviews)
  - `stage2_user_mapping.csv` (reviewer info)

**Key Features**:
- Extracts reviewer URLs (Google Maps profile identifiers)
- Captures ratings, review text, and metadata
- Handles dynamic review loading
- Deduplicates reviewers across restaurants

### Stage 3: User-Restaurant Preference Matrix
- **Purpose**: Build sparse matrix for collaborative filtering
- **Input**: `stage2_restaurant_reviews.csv`
- **Output**:
  - `stage3_user_restaurant_matrix.csv` (37,732 × 710)
  - `stage3_user_restaurant_matrix_dense.csv` (1,871 × 710)
  - `stage3_reviewer_profiles.csv` (reviewer statistics)
  - `stage3_restaurant_profiles.csv` (restaurant statistics)

**Key Features**:
- Builds user-restaurant rating matrix
- Generates dense matrix for high-activity reviewers
- Calculates reviewer and restaurant profiles
- Quality analysis with sparsity metrics

## 🚀 Quick Start

### Installation

```bash
# Navigate to project directory
cd RestoRec

# Install dependencies
pip install -r requirements.txt
```

**Requirements**:
- Python 3.8+
- Chrome browser & ChromeDriver (for Selenium)
- See `requirements.txt` for full list

### Running the Pipeline

```bash
# Run complete 3-stage pipeline
python run_full_pipeline.py
```

The pipeline will:
1. Load existing restaurants and filter by quality (10+ reviews)
2. Scrape complete details for filtered restaurants
3. Extract all reviews and reviewer information
4. Build user-restaurant preference matrix

**Expected Runtime**: ~16 hours (varies by network speed)

### Individual Stages

You can also run individual stages:

```bash
# Stage 1 only: Restaurant scraping
python run_full_pipeline.py --stage 1

# Stage 2 only: Review extraction
python run_full_pipeline.py --stage 2

# Stage 3 only: Matrix building
python run_full_pipeline.py --stage 3
```

## 📁 Project Structure

```
RestoRec/
├── README.md                                    # This file
├── requirements.txt                             # Dependencies
├── run_full_pipeline.py                         # Main 3-stage pipeline
├── filter_restaurants_by_quality.py             # Quality filtering config
├── SCRAPING_PIPELINE.md                         # Pipeline documentation
├── ENHANCED_SCRAPER_GUIDE.md                    # Scraper implementation details
│
├── src/
│   ├── data_collection/
│   │   ├── config.py                            # Configuration & paths
│   │   ├── scraper_utils.py                     # Shared utilities
│   │   ├── restaurant_scraper_enhanced.py       # Stage 1: Restaurant scraping
│   │   ├── restaurant_review_scraper_enhanced.py # Stage 2: Review extraction
│   │   └── matrix_builder.py                    # Stage 3: Matrix building
│   │
│   └── data_processing/
│       └── __init__.py
│
└── data/
    └── raw/                                     # Raw scraped data
        ├── stage1_restaurants.csv               # Restaurants with details
        ├── stage2_restaurant_reviews.csv        # Reviews from restaurants
        ├── stage2_user_mapping.csv              # Reviewer information
        ├── stage3_user_restaurant_matrix.csv    # Full preference matrix
        ├── stage3_user_restaurant_matrix_dense.csv  # Dense matrix (5+ reviews)
        ├── stage3_reviewer_profiles.csv         # Reviewer statistics
        └── stage3_restaurant_profiles.csv       # Restaurant statistics
```

## 🔧 Configuration

Edit `src/data_collection/config.py` to customize behavior:

**PATHS**: File locations for input/output
**SCRAPING**: Anti-detection settings (delays, batch sizes)
**SELENIUM**: Browser options and timeouts

Edit `filter_restaurants_by_quality.py` for quality thresholds:

```python
MIN_REVIEWS = 10      # Minimum reviews per restaurant
MIN_RATING = None     # No minimum rating threshold
```

## 📖 Documentation

- **[SCRAPING_PIPELINE.md](SCRAPING_PIPELINE.md)** - Complete pipeline architecture and stage details
- **[ENHANCED_SCRAPER_GUIDE.md](ENHANCED_SCRAPER_GUIDE.md)** - Detailed scraper implementation guide

## ⚠️ Important Notes

### Data Collection Completed ✅

All stages have been successfully executed:
- **Stage 1**: 768 restaurants scraped with 100% data quality (cuisine, hours, dining options all populated)
- **Stage 2**: 63,888 reviews extracted from 710 restaurants with 100% reviewer URL capture
- **Stage 3**: Complete user-restaurant preference matrix constructed

The data is ready for:
- Collaborative filtering recommendations
- User similarity analysis
- Restaurant-based clustering
- Preference prediction

### Ethical & Legal Considerations

⚠️ **Disclaimer**: Web scraping Google Maps may violate their Terms of Service. This project is for **educational purposes only**.

**Anti-Detection Built-in**:
- Random delays between requests
- Batch processing with breaks
- Realistic user-agent strings
- Selenium WebDriver best practices

### Resume Capability

All stages support resuming from interruption:
- Stage 1: Tracks processed restaurants
- Stage 2: Tracks processed restaurants
- Stage 3: Re-runs from scratch (small data size)

If interrupted, simply re-run `run_full_pipeline.py` - it will skip already-processed items.

## 🎓 Next Steps

The preference matrix is ready for:

1. **Collaborative Filtering Models**
   - User-based nearest neighbors
   - Item-based filtering
   - Matrix factorization (SVD)
   - Neural collaborative filtering

2. **Hybrid Approaches**
   - Combine with restaurant content features (cuisine, price, location)
   - Use deep learning embeddings
   - Content-boosted collaborative filtering

3. **Evaluation & Deployment**
   - Recommendation accuracy metrics
   - A/B testing framework
   - Real-time recommendation API

## 🤝 Contributing

This is a personal educational project. All major development phases are complete.

## 📄 License

MIT License

## 🙏 Acknowledgments

- Uses Selenium for web scraping
- Pandas for data processing
- Collaborative filtering research foundations

---

**Pipeline Status**: ✅ Complete - Ready for model development

**Last Updated**: November 11, 2024
