# RestoRec Scraping Pipeline - Complete Overview

## Architecture

The recommendation system uses a 3-stage scraping and processing pipeline to build a user-restaurant preference matrix for collaborative filtering.

---

## Stage 1: Restaurant Discovery & Filtering

### Purpose
Discover restaurants in Champaign and filter by quality metrics to identify data-rich establishments.

### Files
- `restaurant_scraper_enhanced.py` - Main scraper
- `filter_restaurants_by_quality.py` - Quality filtering

### Process

1. **Discovery**: Search Google Maps for restaurants in Champaign using keywords
2. **Extraction**: For each restaurant, extract:
   - Basic info: name, address, rating, review count
   - Details: cuisine, dining options, phone, hours, website
   - Images: thumbnail URL
   - Menu links

3. **Filtering**: Keep only restaurants with:
   - ≥ 30 reviews (sufficient data)
   - ≥ 3.5 rating (quality threshold)

### Output
- `data/raw/stage1_restaurants_filtered.csv` (~658 restaurants)

### Key Metrics
```
Total restaurants discovered: 874
After filtering: 658 (75.3%)
Average reviews: 536
Average rating: 4.36⭐
Median reviews: 310
```

---

## Stage 2: Review & Reviewer Profile Extraction

### Purpose
Extract all reviews and identify unique reviewers using their Google Maps profile URLs.

### Files
- `restaurant_review_scraper_enhanced.py` - Review scraper
- `run_stage2_filtered.py` - Orchestrator

### Process

For each of the 658 filtered restaurants:

1. **Navigate** to restaurant's Google Maps page
2. **Load all reviews** by scrolling and pagination
3. **Extract for each review**:
   - Reviewer name
   - **Reviewer URL** (Google Maps profile link) ← KEY for linking preferences
   - Rating (1-5 stars)
   - Review date
   - Review text
   - Dining type (dine-in, takeout, delivery)
   - Food/Service/Atmosphere ratings (if available)
   - Recommended dishes

4. **Create unique user mapping** - Identify unique reviewers across all restaurants using reviewer URLs

### Output
- `data/raw/stage2_restaurant_reviews.csv` - All reviews with reviewer info
- `data/raw/stage2_user_mapping.csv` - Reviewer ID mapping

### Data Quality
```
Per review extraction includes:
✓ Reviewer name
✓ Reviewer URL (100% extraction)
✓ Star rating
✓ Review text
✓ Dining type
✓ Food/service/atmosphere ratings
✓ Recommended dishes
```

---

## Stage 3: User-Restaurant Preference Matrix

### Purpose
Build a preference matrix mapping reviewers to restaurants and their ratings for collaborative filtering.

### Files
- `matrix_builder.py` - Matrix construction

### Matrix Structure

```
              Restaurant1  Restaurant2  Restaurant3  ...
ReviewerA          4.0         5.0          3.5     ...
ReviewerB          5.0         N/A          4.0     ...
ReviewerC          N/A         4.5          4.5     ...
...
```

**Rows**: Unique reviewers (identified by Google Maps profile URL)
**Columns**: Restaurants
**Values**: Star ratings (1-5) or NaN if not reviewed

### Process

1. **Pivot** reviews into matrix format (reviewer × restaurant)
2. **Aggregate** multiple reviews by same reviewer (mean)
3. **Generate profiles**:
   - Reviewer profiles: # reviews, avg rating, restaurants rated
   - Restaurant profiles: # reviews, avg rating, reviewer count

### Outputs
- `stage3_user_restaurant_matrix.csv` - Full sparse matrix
- `stage3_user_restaurant_matrix_dense.csv` - Dense matrix (reviewers with 5+ reviews)
- `stage3_reviewer_profiles.csv` - Reviewer statistics
- `stage3_restaurant_profiles.csv` - Restaurant statistics

### Expected Matrix Properties
```
Matrix size: ~200-300 reviewers × 658 restaurants
Sparsity: ~95-98% (sparse data is normal and expected)
- Each reviewer typically rates 5-50 restaurants
- Each restaurant typically rated by 100-200+ reviewers

High-value for recommendations:
- Reviewers with 10+ reviews across different restaurants
- Restaurants with 50+ reviews from diverse reviewers
```

---

## Data Quality Checks

### Stage 1 Checks
- ✓ All restaurants have valid ratings
- ✓ All restaurants have review counts
- ✓ Cuisine extracted successfully
- ✓ Dining options populated

### Stage 2 Checks
- ✓ 100% of reviews have reviewer_url (Google Maps profile link)
- ✓ All reviews have valid ratings (1-5 stars)
- ✓ Review dates properly extracted
- ✓ No null reviewer names
- ✓ Dining type extracted where available

### Stage 3 Checks
- ✓ Matrix properly indexed by reviewer URL
- ✓ All ratings are 1-5 stars
- ✓ No duplicate review entries
- ✓ Sparse matrix properties validated

---

## Usage Example

### Run full pipeline:

```python
# 1. Filter restaurants by quality
python3 filter_restaurants_by_quality.py

# 2. Scrape reviews for filtered restaurants
python3 run_stage2_filtered.py

# 3. Build preference matrix
from src.data_collection.matrix_builder import build_user_restaurant_matrix
build_user_restaurant_matrix()
```

### Use matrix for recommendations:

```python
import pandas as pd

# Load matrix
matrix = pd.read_csv('data/raw/stage3_user_restaurant_matrix.csv', index_col=0)

# Find similar reviewers to a target user
target_reviewer = matrix.loc['https://www.google.com/maps/contrib/...']

# Calculate similarity (cosine, Pearson, etc.)
# Find restaurants rated 5⭐ by similar users but not by target
# Recommend to target user
```

---

## Key Design Decisions

1. **Reviewer URL as unique identifier**: Uses Google Maps profile URL instead of username
   - Advantage: Globally unique, persistent, links to reviewer's other reviews
   - Enables cross-restaurant recommendation

2. **Filtering by review count**: Ensures sufficient data points
   - 30-review minimum captures meaningful preference patterns
   - Filters out low-quality or temporarily-popular places

3. **Sparse matrix is expected**:
   - Not a problem - sparse data is normal for recommendation systems
   - Standard algorithms (SVD, NMF, etc.) handle sparsity well

4. **Detailed extraction during scraping**:
   - Real-time monitoring for data quality
   - Catch missing fields early
   - Verify reviewer URLs are captured

---

## Performance Metrics

| Stage | Time per Restaurant | Total Time (658) |
|-------|-------------------|-----------------|
| Stage 1 | 15-25 sec | ~2-3 hours |
| Stage 2 | 20-60 sec | ~3-10 hours |
| Stage 3 | N/A | <1 min |

---

## Next Steps

With the preference matrix built, you can:

1. **Implement collaborative filtering** (user-based or item-based)
2. **Build KNN recommender** - Find similar users, recommend their highly-rated restaurants
3. **Apply matrix factorization** (SVD, NMF) for dimensionality reduction
4. **Create content-based filters** - Combine with cuisine, price, dining options
5. **Build hybrid recommender** - Combine collaborative + content-based

---

## Troubleshooting

### Missing reviewer_url?
- Check if reviewer button (`.al6Kxe`) is on page
- Verify `data-href` attribute is being extracted
- Check fallback extraction from `onclick` attribute

### Low review extraction?
- Ensure proper scrolling in reviews section
- Check if "expand review text" is working
- Verify page loads fully before extraction

### Matrix sparsity too high?
- Lower minimum review threshold (e.g., 20 instead of 30)
- Include restaurants from adjacent areas
- Combine with content-based filtering

---
