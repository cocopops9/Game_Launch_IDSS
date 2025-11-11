"""
Machine learning models for game success prediction and improvement analysis
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import lightgbm as lgb

from src.data_loader import SteamDataLoader


@st.cache_data
def load_steam_data():
    """Load actual Steam data from CSV files"""
    steam_path = "steam.csv" if os.path.exists("steam.csv") else "/mnt/user-data/uploads/steam.csv"
    tags_path = "steamspy_tag_data.csv" if os.path.exists("steamspy_tag_data.csv") else "/mnt/user-data/uploads/steamspy_tag_data.csv"

    if not os.path.exists(steam_path):
        st.error("❌ steam.csv not found! Please upload your Steam dataset.")
        st.stop()

    loader = SteamDataLoader(
        steam_csv_path=steam_path,
        tags_csv_path=tags_path if os.path.exists(tags_path) else None
    )

    df = loader.load_steam_data()
    st.success(f"✅ Loaded {len(df)} games from Steam dataset")
    return df, loader


@st.cache_resource
def train_models(df, _loader):
    """
    Train ML models focused on IMPROVEMENT ANALYSIS rather than prediction accuracy.
    Uses all available features to learn what IMPROVES owners and review ratios.
    Returns actionable insights on feature impacts and relative improvements.
    """

    print("  🎯 IMPROVEMENT-FOCUSED MODEL TRAINING")
    print("  🎯 Goal: Learn what improves outcomes, not predict exact numbers")

    # Extract ALL available features from the dataset (all 18 columns)
    feature_cols = []

    # Basic features
    for col in ['price', 'required_age', 'release_month', 'release_year', 'is_free']:
        if col in df.columns:
            feature_cols.append(col)

    # Platform features
    for col in ['windows', 'mac', 'linux']:
        if col in df.columns:
            feature_cols.append(col)

    # Engagement features (from positive/negative ratings)
    if 'positive_ratings' in df.columns and 'negative_ratings' in df.columns:
        df['total_ratings'] = df['positive_ratings'] + df['negative_ratings']
        df['engagement_score'] = np.log1p(df['total_ratings'])
        feature_cols.extend(['total_ratings', 'engagement_score'])
    elif 'positive_reviews' in df.columns and 'negative_reviews' in df.columns:
        df['total_ratings'] = df['positive_reviews'] + df['negative_reviews']
        df['engagement_score'] = np.log1p(df['total_ratings'])
        feature_cols.extend(['total_ratings', 'engagement_score'])

    # Playtime features
    for col in ['average_playtime', 'median_playtime']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            feature_cols.append(col)

    # Achievement feature
    if 'achievements' in df.columns:
        df['achievements'] = pd.to_numeric(df['achievements'], errors='coerce').fillna(0)
        df['has_achievements'] = (df['achievements'] > 0).astype(int)
        feature_cols.extend(['achievements', 'has_achievements'])

    # Game age (days since release)
    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        reference_date = pd.Timestamp('2017-01-01')
        df['game_age_days'] = (reference_date - df['release_date']).dt.days
        df['game_age_days'] = df['game_age_days'].fillna(365).clip(0, 10000)
        feature_cols.append('game_age_days')

    # Tag/genre features
    tag_cols = [col for col in df.columns if col.startswith('tag_')]
    feature_cols.extend(tag_cols)

    # Categories and genres
    if 'categories' in df.columns:
        df['num_categories'] = df['categories'].fillna('').str.split(';').str.len()
        feature_cols.append('num_categories')

    if 'genres' in df.columns and len(df['genres']) > 0 and isinstance(df['genres'].iloc[0], str):
        df['num_genres'] = df['genres'].fillna('').str.split(';').str.len()
        feature_cols.append('num_genres')

    print(f"  ✅ Using ALL {len(feature_cols)} features from dataset")

    # Prepare features and targets
    X = df[feature_cols].copy().fillna(0)

    # Use log transformation for owners (handles wide 10K-200M range)
    df['log_owners'] = np.log1p(df['owners'])
    y_owners = df['owners']
    y_owners_log = df['log_owners']
    y_reviews = df['review_ratio']

    print(f"  📊 Dataset: {X.shape[0]} games, {X.shape[1]} features")
    print(f"  📊 Owners range: [{y_owners.min():,.0f} - {y_owners.max():,.0f}]")
    print(f"  📊 Review ratio range: [{y_reviews.min():.2f} - {y_reviews.max():.2f}]")

    # ============================================================================
    # CALCULATE FEATURE IMPACTS - The core improvement analysis
    # ============================================================================
    print("  🔬 Analyzing feature impacts on outcomes...")

    feature_impacts = {}

    for feat in feature_cols:
        if feat not in df.columns:
            continue

        # Check if feature has variation
        if df[feat].nunique() <= 1:
            continue

        # For BINARY features (0/1), calculate improvement when feature is present
        if df[feat].dtype in ['int64', 'bool', 'int32'] and set(df[feat].unique()).issubset({0, 1}):
            with_feat = df[df[feat] == 1]
            without_feat = df[df[feat] == 0]

            if len(with_feat) > 10 and len(without_feat) > 10:  # Need enough samples
                # Owners improvement (use median to be robust to outliers)
                own_with = with_feat['owners'].median()
                own_without = without_feat['owners'].median()
                own_improvement = ((own_with - own_without) / own_without * 100) if own_without > 0 else 0

                # Review ratio improvement (use mean)
                rev_with = with_feat['review_ratio'].mean()
                rev_without = without_feat['review_ratio'].mean()
                rev_improvement = ((rev_with - rev_without) / rev_without * 100) if rev_without > 0 else 0

                feature_impacts[feat] = {
                    'type': 'binary',
                    'owners_improvement_pct': round(own_improvement, 1),
                    'reviews_improvement_pct': round(rev_improvement, 1),
                    'owners_with': int(own_with),
                    'owners_without': int(own_without),
                    'reviews_with': round(rev_with, 3),
                    'reviews_without': round(rev_without, 3),
                    'sample_with': len(with_feat),
                    'sample_without': len(without_feat)
                }

        # For CONTINUOUS features, calculate correlation and percentile impact
        elif pd.api.types.is_numeric_dtype(df[feat]):
            corr_owners = df[[feat, 'owners']].corr().iloc[0, 1]
            corr_reviews = df[[feat, 'review_ratio']].corr().iloc[0, 1]

            # Compare top 25% vs bottom 25%
            q75 = df[feat].quantile(0.75)
            q25 = df[feat].quantile(0.25)

            top_quartile = df[df[feat] >= q75]
            bottom_quartile = df[df[feat] <= q25]

            if len(top_quartile) > 10 and len(bottom_quartile) > 10:
                own_top = top_quartile['owners'].median()
                own_bottom = bottom_quartile['owners'].median()
                own_improvement = ((own_top - own_bottom) / own_bottom * 100) if own_bottom > 0 else 0

                rev_top = top_quartile['review_ratio'].mean()
                rev_bottom = bottom_quartile['review_ratio'].mean()
                rev_improvement = ((rev_top - rev_bottom) / rev_bottom * 100) if rev_bottom > 0 else 0

                feature_impacts[feat] = {
                    'type': 'continuous',
                    'correlation_owners': round(corr_owners, 3),
                    'correlation_reviews': round(corr_reviews, 3),
                    'owners_improvement_pct': round(own_improvement, 1),
                    'reviews_improvement_pct': round(rev_improvement, 1),
                    'owners_top25': int(own_top),
                    'owners_bottom25': int(own_bottom),
                    'reviews_top25': round(rev_top, 3),
                    'reviews_bottom25': round(rev_bottom, 3),
                    'q75_value': round(q75, 2),
                    'q25_value': round(q25, 2)
                }

    print(f"  ✅ Calculated impacts for {len(feature_impacts)} features")

    # ============================================================================
    # Train models for relative scoring (still useful for comparisons)
    # ============================================================================
    X_engineered = X.copy()

    # Add useful interaction features
    if 'price' in X.columns:
        X_engineered['price_log'] = np.log1p(X['price'])

    if 'windows' in X.columns and 'mac' in X.columns and 'linux' in X.columns:
        X_engineered['multi_platform'] = X['windows'] + X['mac'] + X['linux']
        X_engineered['cross_platform'] = (X_engineered['multi_platform'] >= 2).astype(int)

    if tag_cols:
        X_engineered['total_tags'] = X[tag_cols].sum(axis=1)

    # Split data
    X_train, X_test, y_train_log, y_test_log, y_train_rev, y_test_rev = train_test_split(
        X_engineered, y_owners_log, y_reviews, test_size=0.2, random_state=42
    )

    _, _, y_train_actual, y_test_actual, _, _ = train_test_split(
        X_engineered, y_owners, y_reviews, test_size=0.2, random_state=42
    )

    # Clean column names
    X_train.columns = X_train.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)
    X_test.columns = X_test.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)

    # Train models
    print("  🤖 Training models for relative scoring...")
    owners_model = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.08, max_depth=5,
        min_samples_split=15, min_samples_leaf=8,
        random_state=42, subsample=0.8, max_features='sqrt'
    )
    owners_model.fit(X_train, y_train_log)

    review_model = lgb.LGBMRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=6,
        num_leaves=31, min_child_samples=15, random_state=42,
        verbose=-1, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=5, reg_alpha=0.15, reg_lambda=0.15
    )
    review_model.fit(X_train, y_train_rev)

    # Feature importance
    feature_importance_owners = pd.DataFrame({
        'feature': X_train.columns,
        'importance': owners_model.feature_importances_
    }).sort_values('importance', ascending=False)

    feature_importance_reviews = pd.DataFrame({
        'feature': X_train.columns,
        'importance': review_model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Store comprehensive analysis
    st.session_state.data_analysis = {
        'feature_impacts': feature_impacts,  # NEW: Core improvement data
        'feature_importance_owners': feature_importance_owners,
        'feature_importance_reviews': feature_importance_reviews
    }

    print("  ✅ Model training complete!")
    print(f"  💡 TOP IMPROVEMENTS FOR OWNERS:")
    # Show top 3 features with biggest positive impact
    sorted_impacts = sorted(feature_impacts.items(),
                          key=lambda x: x[1].get('owners_improvement_pct', 0),
                          reverse=True)[:3]
    for feat, impact in sorted_impacts:
        pct = impact.get('owners_improvement_pct', 0)
        print(f"     - {feat}: +{pct}% improvement")

    return {
        'owners_model': owners_model,
        'review_model': review_model,
        'feature_cols': X_train.columns.tolist(),
        'feature_impacts': feature_impacts,  # NEW: Return improvement data
        'X_test': X_test,
        'y_owners_test': y_test_log,
        'y_owners_actual_test': y_test_actual,
        'y_reviews_test': y_test_rev,
        'uses_log_transform': True
    }
