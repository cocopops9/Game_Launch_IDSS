"""
Improved Machine Learning Models for Game Success Prediction
Enhanced feature engineering and model architecture for better predictions
"""

import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import (
    GradientBoostingRegressor, 
    RandomForestRegressor, 
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import lightgbm as lgb
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')


def enhanced_feature_engineering(df):
    """
    Enhanced feature engineering with all available data
    """
    print("🔧 Enhanced Feature Engineering...")
    
    feature_cols = []
    
    # ============================================================================
    # 1. BASIC FEATURES
    # ============================================================================
    
    # Price features
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
        df['price_log'] = np.log1p(df['price'])
        df['price_squared'] = df['price'] ** 2
        df['is_free'] = (df['price'] == 0).astype(int)
        df['price_tier'] = pd.cut(df['price'], 
                                   bins=[0, 5, 10, 20, 40, 60, 100],
                                   labels=[0, 1, 2, 3, 4, 5]).fillna(2).astype(int)
        feature_cols.extend(['price', 'price_log', 'price_squared', 'is_free', 'price_tier'])
    
    # Age and time features
    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        reference_date = pd.Timestamp('2017-05-01')  # Approximate data collection date
        df['game_age_days'] = (reference_date - df['release_date']).dt.days
        df['game_age_days'] = df['game_age_days'].fillna(365).clip(0, 10000)
        df['game_age_years'] = df['game_age_days'] / 365
        df['game_age_log'] = np.log1p(df['game_age_days'])
        
        # Extract more time features
        df['release_year'] = df['release_date'].dt.year.fillna(2015)
        df['release_month'] = df['release_date'].dt.month.fillna(6)
        df['release_quarter'] = df['release_date'].dt.quarter.fillna(2)
        df['is_holiday_release'] = df['release_month'].isin([11, 12, 6, 7]).astype(int)
        
        feature_cols.extend(['game_age_days', 'game_age_years', 'game_age_log',
                           'release_year', 'release_month', 'release_quarter', 
                           'is_holiday_release'])
    
    # Required age
    if 'required_age' in df.columns:
        df['required_age'] = pd.to_numeric(df['required_age'], errors='coerce').fillna(0)
        df['is_mature'] = (df['required_age'] >= 18).astype(int)
        feature_cols.extend(['required_age', 'is_mature'])
    
    # ============================================================================
    # 2. ENGAGEMENT FEATURES
    # ============================================================================
    
    # Ratings features
    if 'positive_ratings' in df.columns and 'negative_ratings' in df.columns:
        df['positive_ratings'] = pd.to_numeric(df['positive_ratings'], errors='coerce').fillna(0)
        df['negative_ratings'] = pd.to_numeric(df['negative_ratings'], errors='coerce').fillna(0)
        
        df['total_ratings'] = df['positive_ratings'] + df['negative_ratings']
        df['total_ratings_log'] = np.log1p(df['total_ratings'])
        df['total_ratings_sqrt'] = np.sqrt(df['total_ratings'])
        
        # Review ratio with smoothing to avoid division by zero
        df['review_ratio'] = np.where(
            df['total_ratings'] > 0,
            df['positive_ratings'] / df['total_ratings'],
            0.7  # Default for games with no reviews
        )
        
        # Rating features
        df['has_reviews'] = (df['total_ratings'] > 0).astype(int)
        df['rating_volume_tier'] = pd.cut(df['total_ratings'],
                                         bins=[0, 10, 100, 1000, 10000, 100000],
                                         labels=[0, 1, 2, 3, 4]).fillna(0).astype(int)
        
        # Review sentiment features
        df['review_controversy'] = np.where(
            df['total_ratings'] > 0,
            np.minimum(df['positive_ratings'], df['negative_ratings']) / df['total_ratings'],
            0
        )
        
        feature_cols.extend(['total_ratings', 'total_ratings_log', 'total_ratings_sqrt',
                           'has_reviews', 'rating_volume_tier', 'review_controversy'])
    
    # Playtime features
    if 'average_playtime' in df.columns and 'median_playtime' in df.columns:
        df['average_playtime'] = pd.to_numeric(df['average_playtime'], errors='coerce').fillna(0)
        df['median_playtime'] = pd.to_numeric(df['median_playtime'], errors='coerce').fillna(0)
        
        df['avg_playtime_hours'] = df['average_playtime'] / 60
        df['median_playtime_hours'] = df['median_playtime'] / 60
        df['playtime_log'] = np.log1p(df['average_playtime'])
        
        # Engagement score combining reviews and playtime
        df['engagement_score'] = np.log1p(df['total_ratings']) * np.log1p(df['average_playtime'])
        
        # Playtime skewness (indicates hardcore vs casual games)
        df['playtime_skewness'] = np.where(
            df['median_playtime'] > 0,
            df['average_playtime'] / df['median_playtime'],
            1
        ).clip(0, 10)
        
        feature_cols.extend(['average_playtime', 'median_playtime', 'avg_playtime_hours',
                           'median_playtime_hours', 'playtime_log', 'engagement_score',
                           'playtime_skewness'])
    
    # Achievement features
    if 'achievements' in df.columns:
        df['achievements'] = pd.to_numeric(df['achievements'], errors='coerce').fillna(0)
        df['has_achievements'] = (df['achievements'] > 0).astype(int)
        df['achievements_log'] = np.log1p(df['achievements'])
        df['achievements_tier'] = pd.cut(df['achievements'],
                                        bins=[0, 1, 10, 50, 100, 1000],
                                        labels=[0, 1, 2, 3, 4]).fillna(0).astype(int)
        
        feature_cols.extend(['achievements', 'has_achievements', 'achievements_log', 
                           'achievements_tier'])
    
    # ============================================================================
    # 3. PLATFORM FEATURES
    # ============================================================================
    
    # Parse platforms properly
    if 'platforms' in df.columns:
        df['windows'] = df['platforms'].str.contains('windows', case=False, na=False).astype(int)
        df['mac'] = df['platforms'].str.contains('mac', case=False, na=False).astype(int)
        df['linux'] = df['platforms'].str.contains('linux', case=False, na=False).astype(int)
        
        df['platform_count'] = df['windows'] + df['mac'] + df['linux']
        df['is_cross_platform'] = (df['platform_count'] >= 2).astype(int)
        df['is_all_platforms'] = (df['platform_count'] == 3).astype(int)
        
        feature_cols.extend(['windows', 'mac', 'linux', 'platform_count', 
                           'is_cross_platform', 'is_all_platforms'])
    
    # Language feature
    if 'english' in df.columns:
        df['english'] = pd.to_numeric(df['english'], errors='coerce').fillna(1)
        feature_cols.append('english')
    
    # ============================================================================
    # 4. CATEGORICAL FEATURES (Categories)
    # ============================================================================
    
    if 'categories' in df.columns:
        # Count categories
        df['num_categories'] = df['categories'].fillna('').str.split(';').str.len()
        feature_cols.append('num_categories')
        
        # Important categories as binary features
        important_categories = [
            'Single-player', 'Multi-player', 'Online Multi-Player',
            'Steam Achievements', 'Steam Trading Cards', 'Steam Cloud',
            'Full controller support', 'Partial Controller Support',
            'Steam Leaderboards', 'Co-op', 'Online Co-op',
            'Shared/Split Screen', 'VR Support', 'Steam Workshop',
            'In-App Purchases', 'Includes level editor', 'Commentary available'
        ]
        
        for category in important_categories:
            col_name = f'cat_{category.lower().replace(" ", "_").replace("-", "_").replace("/", "_")}'
            df[col_name] = df['categories'].str.contains(category, case=False, na=False).astype(int)
            feature_cols.append(col_name)
    
    # ============================================================================
    # 5. GENRE FEATURES (All 27 genres)
    # ============================================================================
    
    if 'genres' in df.columns:
        # Count genres
        df['num_genres'] = df['genres'].fillna('').str.split(';').str.len()
        feature_cols.append('num_genres')
        
        # All available genres as binary features
        all_genres = [
            'Indie', 'Action', 'Casual', 'Adventure', 'Strategy', 'Simulation',
            'RPG', 'Early Access', 'Free to Play', 'Sports', 'Racing',
            'Massively Multiplayer', 'Violent', 'Gore', 'Nudity', 'Sexual Content',
            'Utilities', 'Design & Illustration', 'Animation & Modeling',
            'Education', 'Video Production', 'Software Training', 'Audio Production',
            'Web Publishing', 'Game Development', 'Photo Editing', 'Accounting'
        ]
        
        for genre in all_genres:
            col_name = f'genre_{genre.lower().replace(" ", "_").replace("&", "and")}'
            df[col_name] = df['genres'].str.contains(genre, case=False, na=False).astype(int)
            feature_cols.append(col_name)
    
    # ============================================================================
    # 6. TAG FEATURES (Top 100 most common tags)
    # ============================================================================
    
    if 'steamspy_tags' in df.columns:
        # Count tags
        df['num_tags'] = df['steamspy_tags'].fillna('').str.split(';').str.len()
        feature_cols.append('num_tags')
        
        # Top tags as binary features
        top_tags = [
            'Action', 'Casual', 'Adventure', 'Strategy', 'Simulation',
            'RPG', 'Early Access', 'Free to Play', 'Puzzle', 'VR',
            'Racing', 'Sports', 'Platformer', 'Point & Click', 'FPS',
            'Anime', 'Visual Novel', 'Horror', 'Hidden Object', 'Multiplayer',
            'Gore', 'Massively Multiplayer', 'Open World', 'Space', 'Shoot Em Up',
            'Pixel Graphics', 'Survival', 'RTS', 'Female Protagonist', 'Classic',
            'Arcade', 'Sci-fi', 'Turn-Based', 'Tower Defense', 'RPGMaker',
            'Singleplayer', 'Difficult', 'Fantasy', 'Roguelike', 'Comedy',
            'Sandbox', 'Story Rich', 'Atmospheric', '2D', 'Zombies',
            'Co-op', 'Great Soundtrack', 'Physics', 'Management', 'Tactical',
            'Building', 'Fighting', 'Retro', 'War', 'JRPG',
            'Hack and Slash', 'Stealth', 'Mystery', 'Medieval', 'Crafting'
        ]
        
        for tag in top_tags[:60]:  # Use top 60 to avoid too many features
            col_name = f'tag_{tag.lower().replace(" ", "_").replace("&", "and").replace("-", "_")}'
            df[col_name] = df['steamspy_tags'].str.contains(tag, case=False, na=False).astype(int)
            feature_cols.append(col_name)
    
    # ============================================================================
    # 7. DEVELOPER/PUBLISHER FEATURES
    # ============================================================================
    
    if 'developer' in df.columns:
        # Developer popularity (how many games they have)
        dev_counts = df['developer'].value_counts()
        df['developer_game_count'] = df['developer'].map(dev_counts).fillna(1)
        df['is_prolific_developer'] = (df['developer_game_count'] >= 5).astype(int)
        
        # Top developers as binary features
        top_developers = dev_counts.head(20).index.tolist()
        for i, dev in enumerate(top_developers):
            if pd.notna(dev):
                col_name = f'dev_top_{i+1}'
                df[col_name] = (df['developer'] == dev).astype(int)
                feature_cols.append(col_name)
        
        feature_cols.extend(['developer_game_count', 'is_prolific_developer'])
    
    if 'publisher' in df.columns:
        # Publisher popularity
        pub_counts = df['publisher'].value_counts()
        df['publisher_game_count'] = df['publisher'].map(pub_counts).fillna(1)
        df['is_major_publisher'] = (df['publisher_game_count'] >= 10).astype(int)
        
        # Self-published games
        df['is_self_published'] = (df['developer'] == df['publisher']).astype(int)
        
        feature_cols.extend(['publisher_game_count', 'is_major_publisher', 'is_self_published'])
    
    # ============================================================================
    # 8. INTERACTION FEATURES
    # ============================================================================
    
    # Price-quality interactions
    if 'price' in df.columns and 'total_ratings' in df.columns:
        df['price_per_rating'] = np.where(
            df['total_ratings'] > 0,
            df['price'] / np.log1p(df['total_ratings']),
            df['price']
        )
        df['value_score'] = df['review_ratio'] * np.log1p(df['average_playtime']) / np.log1p(df['price'] + 1)
        feature_cols.extend(['price_per_rating', 'value_score'])
    
    # Platform-genre interactions
    if 'is_cross_platform' in feature_cols and 'genre_indie' in feature_cols:
        df['indie_multiplatform'] = df['is_cross_platform'] * df.get('genre_indie', 0)
        feature_cols.append('indie_multiplatform')
    
    # Age-engagement interaction
    if 'game_age_days' in feature_cols and 'engagement_score' in feature_cols:
        df['age_engagement_interaction'] = df['game_age_log'] * df['engagement_score']
        feature_cols.append('age_engagement_interaction')
    
    print(f"✅ Created {len(feature_cols)} features")
    
    return df, feature_cols


def train_improved_models(df, feature_cols):
    """
    Train improved models with better architecture and hyperparameter tuning
    """
    print("\n🚀 Training Improved Models...")
    
    # Prepare features and targets
    X = df[feature_cols].copy()
    
    # Fill NaN values
    X = X.fillna(0)
    
    # For owners: use log transformation (good practice for wide range)
    df['log_owners'] = np.log1p(df['owners'])
    y_owners = df['owners']
    y_owners_log = df['log_owners']
    y_reviews = df['review_ratio']
    
    print(f"📊 Dataset: {X.shape[0]} games, {X.shape[1]} features")
    print(f"📊 Owners range: [{y_owners.min():,.0f} - {y_owners.max():,.0f}]")
    print(f"📊 Review ratio range: [{y_reviews.min():.3f} - {y_reviews.max():.3f}]")
    
    # Split data
    X_train, X_test, y_train_log, y_test_log, y_train_rev, y_test_rev = train_test_split(
        X, y_owners_log, y_reviews, test_size=0.2, random_state=42
    )
    
    # Also get actual owners for evaluation
    _, _, y_train_actual, y_test_actual, _, _ = train_test_split(
        X, y_owners, y_reviews, test_size=0.2, random_state=42
    )
    
    # Clean column names for LightGBM
    X_train.columns = X_train.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)
    X_test.columns = X_test.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)
    feature_cols_clean = X_train.columns.tolist()
    
    # ============================================================================
    # MODEL 1: OWNERS PREDICTION (using ensemble)
    # ============================================================================
    
    print("\n📈 Training Owners Model (Ensemble)...")
    
    # Try multiple models and pick the best
    models_to_try = {
        'XGBoost': XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            n_jobs=-1
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=40,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        'HistGradientBoosting': HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_depth=8,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=42
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
    }
    
    best_score = -np.inf
    best_model = None
    best_model_name = None
    
    for name, model in models_to_try.items():
        print(f"  Testing {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train_log, cv=5, scoring='r2', n_jobs=-1)
        cv_mean = cv_scores.mean()
        
        print(f"    CV R² = {cv_mean:.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        if cv_mean > best_score:
            best_score = cv_mean
            best_model = model
            best_model_name = name
    
    print(f"\n✅ Best model for Owners: {best_model_name} (CV R² = {best_score:.3f})")
    
    # Train best model
    best_model.fit(X_train, y_train_log)
    owners_model = best_model
    
    # Evaluate
    owners_pred_log = owners_model.predict(X_test)
    owners_pred = np.expm1(owners_pred_log)
    
    test_r2_owners = r2_score(y_test_log, owners_pred_log)
    test_mae_owners = mean_absolute_error(y_test_actual, owners_pred)
    test_rmse_owners = np.sqrt(mean_squared_error(y_test_actual, owners_pred))
    
    print(f"  Test R² (log scale): {test_r2_owners:.3f}")
    print(f"  Test MAE: {test_mae_owners:,.0f} owners")
    print(f"  Test RMSE: {test_rmse_owners:,.0f} owners")
    
    # ============================================================================
    # MODEL 2: REVIEW RATIO PREDICTION (with feature selection)
    # ============================================================================
    
    print("\n⭐ Training Review Model (with feature selection)...")
    
    # Feature selection for review model
    # Reviews might be influenced by different features than owners
    selector = SelectKBest(score_func=f_regression, k=100)  # Select top 100 features
    X_train_selected = selector.fit_transform(X_train, y_train_rev)
    X_test_selected = selector.transform(X_test)
    
    selected_features = X_train.columns[selector.get_support()].tolist()
    print(f"  Selected {len(selected_features)} features for review model")
    
    # Try different models for reviews
    review_models_to_try = {
        'XGBoost': XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=5,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=2,
            random_state=42,
            n_jobs=-1
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=31,
            min_child_samples=30,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=2,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_split=30,
            min_samples_leaf=15,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
    }
    
    best_review_score = -np.inf
    best_review_model = None
    best_review_model_name = None
    
    for name, model in review_models_to_try.items():
        print(f"  Testing {name}...")
        
        cv_scores = cross_val_score(model, X_train_selected, y_train_rev, cv=5, scoring='r2', n_jobs=-1)
        cv_mean = cv_scores.mean()
        
        print(f"    CV R² = {cv_mean:.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        if cv_mean > best_review_score:
            best_review_score = cv_mean
            best_review_model = model
            best_review_model_name = name
    
    print(f"\n✅ Best model for Reviews: {best_review_model_name} (CV R² = {best_review_score:.3f})")
    
    # Train best model
    best_review_model.fit(X_train_selected, y_train_rev)
    review_model = best_review_model
    
    # Evaluate
    reviews_pred = review_model.predict(X_test_selected)
    
    test_r2_reviews = r2_score(y_test_rev, reviews_pred)
    test_mae_reviews = mean_absolute_error(y_test_rev, reviews_pred)
    test_rmse_reviews = np.sqrt(mean_squared_error(y_test_rev, reviews_pred))
    
    print(f"  Test R² : {test_r2_reviews:.3f}")
    print(f"  Test MAE: {test_mae_reviews:.3f}")
    print(f"  Test RMSE: {test_rmse_reviews:.3f}")
    
    # ============================================================================
    # FEATURE IMPORTANCE ANALYSIS
    # ============================================================================
    
    print("\n📊 Analyzing Feature Importance...")
    
    # Get feature importance for owners model
    if hasattr(owners_model, 'feature_importances_'):
        owner_importance = pd.DataFrame({
            'feature': feature_cols_clean,
            'importance': owners_model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        owner_importance = pd.DataFrame()
    
    # Get feature importance for review model
    if hasattr(review_model, 'feature_importances_'):
        review_importance = pd.DataFrame({
            'feature': [feature_cols_clean[i] for i in selector.get_support().nonzero()[0]],
            'importance': review_model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        review_importance = pd.DataFrame()
    
    print("\nTop 10 Features for Owners:")
    if not owner_importance.empty:
        for idx, row in owner_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    print("\nTop 10 Features for Reviews:")
    if not review_importance.empty:
        for idx, row in review_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Return all model artifacts
    return {
        'owners_model': owners_model,
        'review_model': review_model,
        'selector': selector,
        'feature_cols': feature_cols_clean,
        'selected_features': selected_features,
        'owner_importance': owner_importance,
        'review_importance': review_importance,
        'X_train': X_train,
        'X_test': X_test,
        'y_train_log': y_train_log,
        'y_test_log': y_test_log,
        'y_train_rev': y_train_rev,
        'y_test_rev': y_test_rev,
        'y_train_actual': y_train_actual,
        'y_test_actual': y_test_actual,
        'test_metrics': {
            'owners_r2': test_r2_owners,
            'owners_mae': test_mae_owners,
            'owners_rmse': test_rmse_owners,
            'owners_model_name': best_model_name,
            'reviews_r2': test_r2_reviews,
            'reviews_mae': test_mae_reviews,
            'reviews_rmse': test_rmse_reviews,
            'reviews_model_name': best_review_model_name
        }
    }


def main():
    """
    Main function to run the improved model training
    """
    print("\n" + "="*60)
    print("🎮 GAME LAUNCH IDSS - IMPROVED MODEL TRAINING")
    print("="*60)
    
    # Load data
    steam_path = "/mnt/user-data/uploads/steam.csv"
    
    print("\n📂 Loading Steam data...")
    df = pd.read_csv(steam_path, quotechar='"', escapechar='\\', on_bad_lines='warn')
    print(f"✅ Loaded {len(df)} games")
    
    # Parse owners (critical step)
    print("\n🔄 Parsing owners data...")
    def parse_owners_range(owners_str):
        """Parse owners range string to numeric midpoint"""
        if pd.isna(owners_str):
            return 10000
        
        try:
            owners_str = str(owners_str).strip()
            
            if '-' in owners_str and not owners_str.startswith('-'):
                parts = owners_str.split('-')
                if len(parts) == 2:
                    lower = int(parts[0].replace(',', '').strip())
                    upper = int(parts[1].replace(',', '').strip())
                    return (lower + upper) / 2
            else:
                return int(owners_str.replace(',', ''))
        except:
            return 10000
    
    df['owners'] = df['owners'].apply(parse_owners_range)
    print(f"✅ Owners range: [{df['owners'].min():,.0f} - {df['owners'].max():,.0f}]")
    
    # Enhanced feature engineering
    df, feature_cols = enhanced_feature_engineering(df)
    
    # Train improved models
    results = train_improved_models(df, feature_cols)
    
    # Print final results
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    
    print(f"\n🎯 Owners Model ({results['test_metrics']['owners_model_name']}):")
    print(f"  - R² Score: {results['test_metrics']['owners_r2']:.3f}")
    print(f"  - MAE: {results['test_metrics']['owners_mae']:,.0f} owners")
    print(f"  - RMSE: {results['test_metrics']['owners_rmse']:,.0f} owners")
    
    print(f"\n⭐ Review Model ({results['test_metrics']['reviews_model_name']}):")
    print(f"  - R² Score: {results['test_metrics']['reviews_r2']:.3f}")
    print(f"  - MAE: {results['test_metrics']['reviews_mae']:.3f}")
    print(f"  - RMSE: {results['test_metrics']['reviews_rmse']:.3f}")
    
    print(f"\n📈 Improvement Summary:")
    print(f"  - Used {len(feature_cols)} features (vs 43 originally)")
    print(f"  - Tested 4 different model architectures")
    print(f"  - Applied feature selection for review model")
    print(f"  - Enhanced feature engineering with interactions")
    
    return results


if __name__ == "__main__":
    results = main()
