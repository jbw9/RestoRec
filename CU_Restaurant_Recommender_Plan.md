# Champaign-Urbana Restaurant Recommendation System
## Comprehensive Implementation Plan

---

## 📋 Project Overview

**Goal**: Build a hybrid recommendation system combining collaborative filtering and content-based filtering for restaurants in the Champaign-Urbana area.

**Key Features**:
- Neural network hybrid model (CF + Content-based)
- Trained on scraped Google Maps user-restaurant ratings
- Enhanced with Yelp API restaurant metadata
- Personalized recommendations for new users
- Eventually: Web interface for UIUC students

**Timeline**: 4-6 weeks (adjustable based on your schedule)

---

## 🗂️ Project Structure

```
cu-restaurant-recommender/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                          # Raw scraped/API data
│   │   ├── google_maps_restaurants.json
│   │   ├── google_maps_reviews.json
│   │   ├── google_maps_users.json
│   │   └── yelp_restaurants.json
│   │
│   ├── processed/                    # Cleaned and merged data
│   │   ├── restaurants_master.csv
│   │   ├── user_restaurant_ratings.csv
│   │   ├── content_features.csv
│   │   └── data_statistics.json
│   │
│   └── splits/                       # Train/val/test splits
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA on scraped data
│   ├── 02_feature_engineering.ipynb  # Create content features
│   ├── 03_baseline_models.ipynb      # Simple CF, Content-based
│   ├── 04_hybrid_model.ipynb         # Neural network hybrid
│   ├── 05_model_comparison.ipynb     # Compare all approaches
│   └── 06_inference_demo.ipynb       # Demo recommendations
│
├── src/
│   ├── __init__.py
│   │
│   ├── data_collection/
│   │   ├── __init__.py
│   │   ├── google_maps_scraper.py    # Scrape restaurants & reviews
│   │   ├── yelp_api_client.py        # Fetch Yelp data
│   │   └── scraper_utils.py          # Helper functions
│   │
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── clean_data.py             # Data cleaning pipeline
│   │   ├── merge_data.py             # Merge Google + Yelp
│   │   ├── feature_engineering.py    # Create content features
│   │   └── train_test_split.py       # Split data
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── collaborative_filtering.py # Pure CF model
│   │   ├── content_based.py          # Pure content model
│   │   ├── hybrid_model.py           # Neural hybrid model
│   │   └── model_utils.py            # Training utilities
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                # RMSE, MAE, etc.
│   │   └── evaluator.py              # Evaluation pipeline
│   │
│   └── inference/
│       ├── __init__.py
│       ├── recommender.py            # Recommendation engine
│       └── cold_start_handler.py     # Handle new users
│
├── models/                           # Saved model checkpoints
│   ├── cf_model.pth
│   ├── content_model.pth
│   ├── hybrid_model_v1.pth
│   ├── hybrid_model_v2.pth
│   └── best_model.pth
│
├── config/
│   ├── config.yaml                   # Main configuration
│   └── model_config.yaml             # Model hyperparameters
│
├── scripts/
│   ├── run_scraper.py                # Execute scraping
│   ├── run_data_pipeline.py          # Process all data
│   ├── train_model.py                # Train models
│   └── evaluate_models.py            # Run evaluation
│
├── tests/                            # Unit tests (optional)
│   ├── test_scraper.py
│   ├── test_data_processing.py
│   └── test_models.py
│
└── results/                          # Outputs and visualizations
    ├── figures/
    │   ├── rating_distribution.png
    │   ├── category_distribution.png
    │   ├── training_curves.png
    │   └── model_comparison.png
    │
    └── reports/
        ├── data_analysis_report.md
        ├── model_performance_report.md
        └── recommendations_examples.txt
```

---

## 📅 Implementation Phases

### **Phase 1: Data Collection (Week 1)**

#### 1.1 Google Maps Scraping
**Goal**: Scrape ~100 restaurants and their reviews

**Tasks**:
- [ ] Set up Google Maps scraper (reuse + debug your existing code)
- [ ] Scrape restaurant list in Champaign-Urbana
- [ ] For each restaurant: get top 50 reviews
- [ ] For each reviewer: visit profile, scrape their other CU restaurant reviews
- [ ] Save raw data to JSON files

**Implementation Details**:

```python
# src/data_collection/google_maps_scraper.py

"""
Data to collect:
- Restaurants: name, address, rating, review_count, price_level, categories
- Reviews: user_id, restaurant_name, rating, text, date
- Users: user_id, username, total_reviews, other_cu_restaurant_reviews
"""

class GoogleMapsScraper:
    def scrape_cu_restaurants(self, max_restaurants=100):
        """Scrape restaurant list"""
        pass
    
    def scrape_restaurant_reviews(self, restaurant_url, max_reviews=50):
        """Get top reviews for a restaurant"""
        pass
    
    def scrape_user_profile(self, user_url):
        """Get user's other CU restaurant reviews"""
        pass
```

**Output Files**:
- `data/raw/google_maps_restaurants.json`
- `data/raw/google_maps_reviews.json`
- `data/raw/google_maps_users.json`

**Anti-Detection Tips**:
- Random delays between requests (5-15 seconds)
- Rotate user agents
- Use residential proxies if needed
- Scrape during off-peak hours
- Save progress frequently (in case of interruption)

---

#### 1.2 Yelp API Data Collection
**Goal**: Get restaurant metadata for content features

**Tasks**:
- [ ] Sign up for Yelp Places API (free trial)
- [ ] Match Google Maps restaurants to Yelp
- [ ] Fetch: categories, price, location, hours, transactions
- [ ] Save to JSON

**Implementation**:

```python
# src/data_collection/yelp_api_client.py

import requests

class YelpAPIClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.yelp.com/v3"
        
    def search_restaurants(self, location="Champaign, IL"):
        """Get all restaurants in CU"""
        pass
    
    def get_business_details(self, business_id):
        """Get detailed info for a restaurant"""
        pass
    
    def match_google_to_yelp(self, google_restaurant):
        """Match Google Maps restaurant to Yelp (by name/address)"""
        pass
```

**Output Files**:
- `data/raw/yelp_restaurants.json`

---

### **Phase 2: Data Processing (Week 1-2)**

#### 2.1 Data Cleaning

**Tasks**:
- [ ] Remove duplicates
- [ ] Handle missing values
- [ ] Standardize restaurant names
- [ ] Filter outliers (e.g., users with only 1 review)
- [ ] Create consistent user and restaurant IDs

**Implementation**:

```python
# src/data_processing/clean_data.py

def clean_google_reviews(raw_reviews):
    """
    - Remove reviews without ratings
    - Standardize date formats
    - Filter spam/fake reviews
    """
    pass

def deduplicate_restaurants(google_restos, yelp_restos):
    """
    Match restaurants across sources
    Handle name variations (e.g., "Kam's" vs "Kams")
    """
    pass

def filter_users(user_ratings, min_reviews=3):
    """
    Keep only users who reviewed at least 3 CU restaurants
    (for meaningful CF)
    """
    pass
```

---

#### 2.2 Data Merging

**Goal**: Create master datasets

**Tasks**:
- [ ] Merge Google Maps + Yelp data
- [ ] Create user-restaurant rating matrix
- [ ] Create restaurant content features table

**Output Files**:
- `data/processed/restaurants_master.csv`
  - Columns: `restaurant_id, name, google_rating, yelp_rating, categories, price, lat, lon, ...`
- `data/processed/user_restaurant_ratings.csv`
  - Columns: `user_id, restaurant_id, rating, timestamp`

---

#### 2.3 Feature Engineering

**Goal**: Create content feature vectors

**Tasks**:
- [ ] One-hot encode categories
- [ ] One-hot encode price levels
- [ ] Create location features (distance from campus)
- [ ] Normalize numerical features (rating, review_count)
- [ ] Create derived features (e.g., "open_late", "delivery_available")

**Implementation**:

```python
# src/data_processing/feature_engineering.py

def create_content_features(restaurant_df):
    """
    Returns: DataFrame with feature columns
    
    Features:
    - One-hot: [italian, chinese, mexican, american, indian, ...]
    - One-hot: [price_1, price_2, price_3, price_4]
    - Numerical: [rating_norm, review_count_log, distance_km]
    - Binary: [has_delivery, has_pickup, open_late]
    """
    
    # Example: Category one-hot encoding
    categories = ['italian', 'chinese', 'mexican', 'american', 'indian', 
                  'thai', 'japanese', 'korean', 'pizza', 'burgers', 
                  'coffee', 'bars', 'fast_food', 'breakfast']
    
    for cat in categories:
        restaurant_df[f'cat_{cat}'] = restaurant_df['categories'].apply(
            lambda x: 1 if cat in x else 0
        )
    
    # Distance from campus (e.g., Union)
    campus_lat, campus_lon = 40.1092, -88.2273
    restaurant_df['distance_km'] = calculate_distance(
        restaurant_df['latitude'], 
        restaurant_df['longitude'],
        campus_lat, 
        campus_lon
    )
    
    return restaurant_df
```

**Output Files**:
- `data/processed/content_features.csv`

---

#### 2.4 Train/Val/Test Split

**Tasks**:
- [ ] Split data: 80% train, 10% validation, 10% test
- [ ] Ensure no data leakage
- [ ] Stratify by user (each user appears in all sets if they have enough ratings)

**Implementation**:

```python
# src/data_processing/train_test_split.py

def split_data(ratings_df, train_ratio=0.8, val_ratio=0.1):
    """
    For each user, split their ratings:
    - 80% to train
    - 10% to validation
    - 10% to test
    """
    
    train_data = []
    val_data = []
    test_data = []
    
    for user_id in ratings_df['user_id'].unique():
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        
        n = len(user_ratings)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_data.append(user_ratings.iloc[:train_end])
        val_data.append(user_ratings.iloc[train_end:val_end])
        test_data.append(user_ratings.iloc[val_end:])
    
    return pd.concat(train_data), pd.concat(val_data), pd.concat(test_data)
```

---

### **Phase 3: Baseline Models (Week 2)**

#### 3.1 Pure Collaborative Filtering

**Goal**: Replicate your movie recommendation approach

**Tasks**:
- [ ] Implement matrix factorization with embeddings
- [ ] Train on user-restaurant ratings only
- [ ] Evaluate on test set

**Implementation**:

```python
# src/models/collaborative_filtering.py

import torch
import torch.nn as nn

class CollaborativeFilteringModel(nn.Module):
    def __init__(self, n_users, n_restaurants, embedding_dim=32):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.restaurant_embedding = nn.Embedding(n_restaurants, embedding_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.restaurant_bias = nn.Embedding(n_restaurants, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, user_ids, restaurant_ids):
        user_emb = self.user_embedding(user_ids)
        resto_emb = self.restaurant_embedding(restaurant_ids)
        
        user_b = self.user_bias(user_ids).squeeze()
        resto_b = self.restaurant_bias(restaurant_ids).squeeze()
        
        dot_product = (user_emb * resto_emb).sum(dim=1)
        prediction = dot_product + user_b + resto_b + self.global_bias
        
        # Clamp to [0.5, 5.0]
        prediction = torch.clamp(prediction, 0.5, 5.0)
        
        return prediction
```

**Notebook**: `notebooks/03_baseline_models.ipynb`

---

#### 3.2 Pure Content-Based

**Goal**: Recommendations based only on restaurant features

**Tasks**:
- [ ] Compute restaurant similarity using content features
- [ ] For a user, average their liked restaurants' features
- [ ] Recommend restaurants similar to user's profile

**Implementation**:

```python
# src/models/content_based.py

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ContentBasedModel:
    def __init__(self, content_features_df):
        self.features = content_features_df
        self.feature_matrix = content_features_df.values
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
        
    def get_user_profile(self, user_ratings, restaurant_ids):
        """
        Average the features of restaurants the user liked (rating >= 4)
        """
        liked_restos = user_ratings[user_ratings['rating'] >= 4]['restaurant_id']
        liked_features = self.features.loc[liked_restos].mean(axis=0)
        return liked_features
        
    def recommend(self, user_profile, top_k=10):
        """
        Find restaurants most similar to user profile
        """
        similarities = cosine_similarity([user_profile], self.feature_matrix)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return top_indices
```

---

### **Phase 4: Hybrid Neural Network Model (Week 3-4)**

#### 4.1 Model Architecture

**Goal**: Combine CF embeddings + content features in a neural network

**Tasks**:
- [ ] Design architecture
- [ ] Implement in PyTorch
- [ ] Add regularization (dropout, L2)

**Implementation**:

```python
# src/models/hybrid_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridRecommender(nn.Module):
    def __init__(self, n_users, n_restaurants, n_content_features, 
                 embedding_dim=32, hidden_dims=[64, 32]):
        super().__init__()
        
        # Collaborative Filtering embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.restaurant_embedding = nn.Embedding(n_restaurants, embedding_dim)
        
        # Content feature processing
        self.content_fc1 = nn.Linear(n_content_features, 32)
        self.content_fc2 = nn.Linear(32, 16)
        self.content_dropout = nn.Dropout(0.3)
        
        # Combination layers
        combined_dim = embedding_dim + embedding_dim + 16
        self.fc1 = nn.Linear(combined_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 1)
        
        self.dropout = nn.Dropout(0.2)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.restaurant_embedding.weight)
        
    def forward(self, user_ids, restaurant_ids, content_features):
        # CF path
        user_emb = self.user_embedding(user_ids)
        resto_emb = self.restaurant_embedding(restaurant_ids)
        
        # Content path
        content = F.relu(self.content_fc1(content_features))
        content = self.content_dropout(content)
        content = F.relu(self.content_fc2(content))
        
        # Concatenate all features
        combined = torch.cat([user_emb, resto_emb, content], dim=1)
        
        # Feed through network
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        output = self.fc3(x)
        
        # Scale to rating range [0.5, 5.0]
        rating = torch.sigmoid(output) * 4.5 + 0.5
        
        return rating
```

**Notebook**: `notebooks/04_hybrid_model.ipynb`

---

#### 4.2 Training Pipeline

**Implementation**:

```python
# src/models/model_utils.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class RestaurantDataset(Dataset):
    def __init__(self, ratings_df, content_features_df):
        self.user_ids = torch.LongTensor(ratings_df['user_id_encoded'].values)
        self.resto_ids = torch.LongTensor(ratings_df['restaurant_id_encoded'].values)
        self.ratings = torch.FloatTensor(ratings_df['rating'].values)
        
        # Get content features for each restaurant in the ratings
        self.content_features = torch.FloatTensor(
            content_features_df.loc[ratings_df['restaurant_id']].values
        )
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return {
            'user_id': self.user_ids[idx],
            'restaurant_id': self.resto_ids[idx],
            'content_features': self.content_features[idx],
            'rating': self.ratings[idx]
        }

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        user_ids = batch['user_id'].to(device)
        resto_ids = batch['restaurant_id'].to(device)
        content_features = batch['content_features'].to(device)
        ratings = batch['rating'].to(device)
        
        # Forward pass
        predictions = model(user_ids, resto_ids, content_features)
        loss = criterion(predictions, ratings)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            user_ids = batch['user_id'].to(device)
            resto_ids = batch['restaurant_id'].to(device)
            content_features = batch['content_features'].to(device)
            ratings = batch['rating'].to(device)
            
            predictions = model(user_ids, resto_ids, content_features)
            loss = criterion(predictions, ratings)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

**Training Script**:

```python
# scripts/train_model.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from src.models.hybrid_model import HybridRecommender
from src.models.model_utils import RestaurantDataset, train_epoch, validate

def main():
    # Load data
    train_df = pd.read_csv('data/splits/train.csv')
    val_df = pd.read_csv('data/splits/val.csv')
    content_features = pd.read_csv('data/processed/content_features.csv')
    
    # Create datasets
    train_dataset = RestaurantDataset(train_df, content_features)
    val_dataset = RestaurantDataset(val_df, content_features)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    
    # Initialize model
    n_users = train_df['user_id_encoded'].nunique()
    n_restaurants = train_df['restaurant_id_encoded'].nunique()
    n_content_features = content_features.shape[1]
    
    model = HybridRecommender(n_users, n_restaurants, n_content_features)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(100):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= 10:
            print("Early stopping triggered")
            break

if __name__ == '__main__':
    main()
```

---

#### 4.3 Hyperparameter Tuning

**Experiments to try**:
- Embedding dimensions: [16, 32, 64, 128]
- Hidden layer sizes: [64, 32], [128, 64, 32], [256, 128, 64]
- Dropout rates: [0.1, 0.2, 0.3, 0.5]
- Learning rates: [0.001, 0.0001]
- Weight decay: [1e-5, 1e-4, 1e-3]

**Track experiments**: Use a simple table or tool like Weights & Biases

---

### **Phase 5: Evaluation (Week 4)**

#### 5.1 Quantitative Metrics

**Tasks**:
- [ ] Implement RMSE, MAE
- [ ] Compare CF vs Content vs Hybrid
- [ ] Analyze error patterns

**Implementation**:

```python
# src/evaluation/metrics.py

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_rmse(predictions, actuals):
    return np.sqrt(mean_squared_error(actuals, predictions))

def calculate_mae(predictions, actuals):
    return mean_absolute_error(actuals, predictions)

def calculate_accuracy_within_threshold(predictions, actuals, threshold=0.5):
    """
    Percentage of predictions within threshold of actual rating
    """
    errors = np.abs(predictions - actuals)
    return (errors <= threshold).mean()
```

---

#### 5.2 Qualitative Evaluation

**Tasks**:
- [ ] Generate recommendations for sample users
- [ ] Manually check if they make sense
- [ ] Test cold start scenarios

**Notebook**: `notebooks/05_model_comparison.ipynb`

---

### **Phase 6: Inference System (Week 5)**

#### 6.1 Recommendation Engine

**Implementation**:

```python
# src/inference/recommender.py

import torch
import pandas as pd

class RestaurantRecommender:
    def __init__(self, model, restaurants_df, content_features_df):
        self.model = model
        self.restaurants_df = restaurants_df
        self.content_features_df = content_features_df
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
    def recommend_for_user(self, user_id, user_rated_restaurants, top_k=10):
        """
        Generate top-k recommendations for a user
        
        Args:
            user_id: User ID (encoded)
            user_rated_restaurants: List of restaurant IDs user has already rated
            top_k: Number of recommendations
        """
        # Get all restaurants user hasn't rated
        all_restaurants = self.restaurants_df['restaurant_id_encoded'].values
        unrated = [r for r in all_restaurants if r not in user_rated_restaurants]
        
        # Prepare inputs
        user_ids = torch.LongTensor([user_id] * len(unrated)).to(self.device)
        resto_ids = torch.LongTensor(unrated).to(self.device)
        content_features = torch.FloatTensor(
            self.content_features_df.iloc[unrated].values
        ).to(self.device)
        
        # Predict ratings
        with torch.no_grad():
            predictions = self.model(user_ids, resto_ids, content_features)
        
        # Get top-k
        top_k_indices = torch.argsort(predictions, descending=True)[:top_k]
        top_k_resto_ids = [unrated[i] for i in top_k_indices.cpu().numpy()]
        top_k_predictions = predictions[top_k_indices].cpu().numpy()
        
        # Get restaurant details
        recommendations = []
        for resto_id, pred_rating in zip(top_k_resto_ids, top_k_predictions):
            resto_info = self.restaurants_df[
                self.restaurants_df['restaurant_id_encoded'] == resto_id
            ].iloc[0]
            
            recommendations.append({
                'name': resto_info['name'],
                'predicted_rating': pred_rating,
                'actual_rating': resto_info['rating'],
                'categories': resto_info['categories'],
                'price': resto_info['price'],
                'address': resto_info['address']
            })
        
        return recommendations
```

---

#### 6.2 Cold Start Handler

**For new users (no ratings yet)**:

```python
# src/inference/cold_start_handler.py

class ColdStartHandler:
    def __init__(self, content_features_df, restaurants_df):
        self.content_features_df = content_features_df
        self.restaurants_df = restaurants_df
        
    def recommend_popular(self, top_k=10):
        """
        Recommend most popular restaurants (by rating + review count)
        """
        self.restaurants_df['popularity_score'] = (
            self.restaurants_df['rating'] * 
            np.log1p(self.restaurants_df['review_count'])
        )
        top_restaurants = self.restaurants_df.nlargest(top_k, 'popularity_score')
        return top_restaurants
    
    def recommend_by_preferences(self, preferred_categories, preferred_price, top_k=10):
        """
        Content-based recommendations based on user preferences
        """
        # Filter by preferences
        filtered = self.restaurants_df[
            (self.restaurants_df['categories'].apply(
                lambda x: any(cat in x for cat in preferred_categories)
            )) &
            (self.restaurants_df['price'] == preferred_price)
        ]
        
        # Sort by rating
        return filtered.nlargest(top_k, 'rating')
```

---

#### 6.3 Demo Interface (Notebook)

**Notebook**: `notebooks/06_inference_demo.ipynb`

**Example Usage**:

```python
# Load trained model
model = HybridRecommender(n_users, n_restaurants, n_content_features)
model.load_state_dict(torch.load('models/best_model.pth'))

recommender = RestaurantRecommender(model, restaurants_df, content_features_df)

# Simulate new user
print("Welcome! Please rate a few restaurants:")

# User rates 5 restaurants
user_ratings = {
    'Crane Alley': 5,
    'Maize': 4,
    'Kofusion': 5,
    'Nando Milano': 3,
    'Black Dog': 4
}

# Get recommendations
recommendations = recommender.recommend_for_user(
    user_id=0,  # New user
    user_rated_restaurants=list(user_ratings.keys()),
    top_k=10
)

# Display
for i, rec in enumerate(recommendations, 1):
    print(f"\n{i}. {rec['name']}")
    print(f"   Predicted Rating: {rec['predicted_rating']:.1f} ⭐")
    print(f"   Categories: {rec['categories']}")
    print(f"   Price: {rec['price']}")
```

---

## 📊 Expected Results

### Data Statistics (Estimated)
- **Restaurants**: ~100 in Champaign-Urbana
- **Users**: ~500-1000 (from Google Maps reviews)
- **Ratings**: ~5,000-10,000 (user-restaurant pairs)
- **Sparsity**: ~90-95% (most users only rate a few restaurants)

### Model Performance (Target)
- **RMSE**: < 0.8 (on 1-5 scale)
- **MAE**: < 0.6
- **Accuracy within 0.5 stars**: > 70%

### Comparison
| Model | RMSE | MAE | Training Time |
|-------|------|-----|---------------|
| Pure CF | 0.85 | 0.65 | 5 min |
| Content-Based | 0.90 | 0.70 | 1 min |
| Hybrid (Simple) | 0.80 | 0.60 | 8 min |
| Hybrid (Neural) | 0.75 | 0.55 | 15 min |

---

## ⚠️ Potential Challenges & Solutions

### Challenge 1: Google Maps Scraping Issues
**Problems**:
- CAPTCHAs
- IP bans
- Dynamic content loading

**Solutions**:
- Use Selenium/Playwright (handle JS rendering)
- Add random delays (5-15 seconds)
- Use residential proxies
- Scrape during off-peak hours
- Save progress frequently

---

### Challenge 2: Data Sparsity
**Problem**: Most users only review a few restaurants

**Solutions**:
- Filter users with minimum 3 reviews
- Use hybrid model (content features help with sparse data)
- Implement cold start handlers
- Consider implicit feedback (clicks, time spent)

---

### Challenge 3: Restaurant Matching
**Problem**: Same restaurant has different names in Google vs Yelp

**Solutions**:
- Use fuzzy string matching (fuzzywuzzy library)
- Match by address/location
- Manual verification for ambiguous cases

```python
from fuzzywuzzy import fuzz

def match_restaurants(google_name, google_address, yelp_restaurants):
    best_match = None
    best_score = 0
    
    for yelp_resto in yelp_restaurants:
        name_score = fuzz.ratio(google_name.lower(), yelp_resto['name'].lower())
        address_score = fuzz.ratio(google_address.lower(), yelp_resto['address'].lower())
        
        combined_score = 0.6 * name_score + 0.4 * address_score
        
        if combined_score > best_score and combined_score > 80:
            best_score = combined_score
            best_match = yelp_resto
    
    return best_match
```

---

### Challenge 4: Model Overfitting
**Problem**: Model performs well on train but poorly on test

**Solutions**:
- Add dropout layers (0.2-0.3)
- Use L2 regularization (weight_decay=1e-5)
- Reduce model complexity
- Increase training data
- Early stopping

---

### Challenge 5: Cold Start Problem
**Problem**: New users have no ratings

**Solutions**:
- Ask new users to rate 5-10 popular restaurants
- Use content-based recommendations initially
- Use demographic info (if available)
- Show most popular restaurants

---

## 🔧 Configuration Files

### `config/config.yaml`

```yaml
# Data paths
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  splits_dir: "data/splits"

# Scraping settings
scraping:
  max_restaurants: 100
  max_reviews_per_restaurant: 50
  delay_between_requests: [5, 15]  # Random delay in seconds
  user_agent: "Mozilla/5.0 ..."

# Yelp API
yelp:
  api_key: "YOUR_API_KEY_HERE"
  base_url: "https://api.yelp.com/v3"
  location: "Champaign, IL"

# Data processing
processing:
  min_reviews_per_user: 3
  min_reviews_per_restaurant: 5
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1

# Feature engineering
features:
  categories: [
    "italian", "chinese", "mexican", "american", "indian",
    "thai", "japanese", "korean", "pizza", "burgers",
    "coffee", "bars", "fast_food", "breakfast", "brunch"
  ]
  price_levels: ["$", "$$", "$$$", "$$$$"]
  campus_location: [40.1092, -88.2273]  # UIUC Union
```

### `config/model_config.yaml`

```yaml
# Model architecture
model:
  embedding_dim: 32
  hidden_dims: [64, 32]
  dropout: 0.2
  content_hidden_dims: [32, 16]
  content_dropout: 0.3

# Training
training:
  batch_size: 512
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 1e-5
  early_stopping_patience: 10
  reduce_lr_patience: 5
  reduce_lr_factor: 0.5

# Evaluation
evaluation:
  metrics: ["rmse", "mae", "accuracy_0.5"]
```

---

## 📝 README Template

```markdown
# Champaign-Urbana Restaurant Recommender

A hybrid recommendation system combining collaborative filtering and content-based filtering for restaurants in the Champaign-Urbana area.

## Features
- Neural network hybrid model
- Personalized recommendations based on user preferences
- Cold start handling for new users
- Trained on real Google Maps review data + Yelp metadata

## Installation

```bash
# Clone repo
git clone https://github.com/yourusername/cu-restaurant-recommender.git
cd cu-restaurant-recommender

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Data Collection
```bash
# Scrape Google Maps (WARNING: violates ToS, use at own risk)
python scripts/run_scraper.py

# Fetch Yelp data
python scripts/fetch_yelp_data.py
```

### 2. Data Processing
```bash
python scripts/run_data_pipeline.py
```

### 3. Train Model
```bash
python scripts/train_model.py
```

### 4. Get Recommendations
```python
from src.inference.recommender import RestaurantRecommender

# Load model and get recommendations
recommender = RestaurantRecommender.load('models/best_model.pth')
recommendations = recommender.recommend(user_id, top_k=10)
```

## Project Structure
See [Project Structure](#-project-structure) section above.

## Results
- RMSE: 0.75
- MAE: 0.55
- See `results/reports/model_performance_report.md` for details

## License
MIT

## Disclaimer
This project scrapes Google Maps for educational purposes only. Web scraping may violate Terms of Service. Use at your own risk.
```

---

## 🔄 Development Workflow

### Daily Tasks:
1. **Week 1**: Focus on data collection
2. **Week 2**: Data processing and baseline models
3. **Week 3-4**: Build and train hybrid model
4. **Week 5**: Evaluation and inference system
5. **Week 6**: Documentation and polish

### Git Workflow:
```bash
# Create feature branches
git checkout -b feature/google-scraper
git checkout -b feature/hybrid-model
git checkout -b feature/evaluation

# Commit frequently
git add .
git commit -m "Add Google Maps scraper"
git push origin feature/google-scraper
```

---

## 📚 Dependencies

### `requirements.txt`

```txt
# Core
numpy==1.24.3
pandas==2.0.2
scipy==1.10.1

# Web scraping
selenium==4.10.0
beautifulsoup4==4.12.2
requests==2.31.0

# Machine learning
torch==2.0.1
scikit-learn==1.3.0

# Data processing
fuzzywuzzy==0.18.0
python-Levenshtein==0.21.1

# Visualization
matplotlib==3.7.1
seaborn==0.12.2

# Utilities
pyyaml==6.0
tqdm==4.65.0
jupyter==1.0.0
ipykernel==6.23.1

# Optional
# wandb==0.15.4  # For experiment tracking
```

---

## 🎯 Success Criteria

### Minimum Viable Product (MVP):
- ✅ Successfully scrape 100 restaurants with reviews
- ✅ Build working hybrid model
- ✅ RMSE < 0.9 on test set
- ✅ Generate sensible recommendations (manual check)
- ✅ Demo notebook showing recommendations

### Stretch Goals:
- ⭐ Deploy web interface (Flask/Streamlit)
- ⭐ Collect real user ratings via Google Form
- ⭐ A/B test different model architectures
- ⭐ Add explainability ("Recommended because you liked X")
- ⭐ Implement diversity in recommendations

---

## 🚀 Next Steps (After MVP)

### Phase 7: Web Interface (Future)
- Build Flask/FastAPI backend
- Create React/Streamlit frontend
- Deploy on Heroku/Railway/Vercel
- Collect real user feedback

### Phase 8: Iteration
- Retrain with real user data
- Add new features (time of day, weather, etc.)
- Improve cold start handling
- Add social features (friends' recommendations)

---

## 📞 Questions or Issues?

If you encounter problems:
1. Check `results/reports/troubleshooting.md`
2. Review error logs in `logs/`
3. Open an issue on GitHub
4. Email: your-email@example.com

---

## 📖 Learning Resources

- **Collaborative Filtering**: [Towards Data Science Article](https://towardsdatascience.com/collaborative-filtering-explained-d5b7d6c71ce0)
- **Hybrid Recommenders**: [Google Developers Guide](https://developers.google.com/machine-learning/recommendation)
- **PyTorch Tutorial**: [Official PyTorch Docs](https://pytorch.org/tutorials/)

---

**Good luck with your project! 🎉**

---

**Estimated Total Time**: 4-6 weeks
- Week 1: Data collection (15-20 hours)
- Week 2: Data processing + baselines (15-20 hours)
- Week 3-4: Hybrid model + training (20-30 hours)
- Week 5: Evaluation + inference (10-15 hours)
- Week 6: Documentation + polish (5-10 hours)

**Total**: ~70-100 hours

---

## Appendix: Sample Code Snippets

### A. Encoding User/Restaurant IDs

```python
def encode_ids(df, column_name):
    """Create mapping from original IDs to continuous integers"""
    unique_ids = df[column_name].unique()
    id_to_idx = {id_val: idx for idx, id_val in enumerate(unique_ids)}
    idx_to_id = {idx: id_val for id_val, idx in id_to_idx.items()}
    
    df[f'{column_name}_encoded'] = df[column_name].map(id_to_idx)
    
    return id_to_idx, idx_to_id
```

### B. Distance Calculation

```python
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    """Calculate distance between two points on Earth (in km)"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    
    return km
```

### C. Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
```

---

**END OF PLAN**
