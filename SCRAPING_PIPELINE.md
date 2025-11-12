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
   - ≥ 10 reviews (sufficient data)
   - No minimum rating threshold

### Output
- `data/raw/stage1_restaurants.csv` (768 restaurants with complete details)

### Key Metrics
```
Total restaurants discovered: 874
After filtering (10+ reviews): 768 (87.9%)
Average reviews per restaurant: ~83
Average rating: 4.38⭐
Cuisine field: 100% populated
Hours field: 100% populated
Dining options field: 100% populated
```

---

## Stage 2: Review & Reviewer Profile Extraction

### Purpose
Extract all reviews and identify unique reviewers using their Google Maps profile URLs.

### Files
- `restaurant_review_scraper_enhanced.py` - Review scraper
- `run_full_pipeline.py` - Orchestrator

### Process

For each of the 768 filtered restaurants (10+ reviews):

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
Full Matrix size: 37,732 reviewers × 710 restaurants
Dense Matrix size: 1,871 reviewers × 710 restaurants (5+ reviews)
Sparsity: 99.76% (expected for collaborative filtering)
- Each reviewer typically rates 1-20 restaurants
- Each restaurant typically rated by 50-110+ reviewers

High-value for recommendations:
- Reviewers with 5+ reviews (1,871 reviewers in dense matrix)
- Restaurants with 50+ reviews from diverse reviewers
- Top reviewer: 111 reviews (Ben Brenner)
- Top restaurant: 110 reviews (Bobo's Barbecue)
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

```bash
# Run all 3 stages automatically
python run_full_pipeline.py

# Or run individual stages:
python run_full_pipeline.py --stage 1
python run_full_pipeline.py --stage 2
python run_full_pipeline.py --stage 3
```

### Or run stages directly in Python:

```python
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
   - 10-review minimum captures meaningful preference patterns
   - Balances between data richness and restaurant diversity
   - Filters out new or inactive restaurants

3. **Sparse matrix is expected**:
   - Not a problem - sparse data is normal for recommendation systems
   - Standard algorithms (SVD, NMF, etc.) handle sparsity well

4. **Detailed extraction during scraping**:
   - Real-time monitoring for data quality
   - Catch missing fields early
   - Verify reviewer URLs are captured

---

## Performance Metrics

| Stage | Time per Restaurant | Total Time (768 restaurants) |
|-------|-------------------|-----------------|
| Stage 1 | 5-15 sec | ~1-2 hours |
| Stage 2 | 20-60 sec | ~4-8 hours |
| Stage 3 | N/A | <1 min |
| **Total** | - | **~16 hours** |

**Actual recorded execution**: 15.91 hours for complete pipeline on 768 restaurants generating 63,888 reviews

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
