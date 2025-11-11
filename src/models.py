"""
Machine learning models for game success prediction and improvement analysis
"""

import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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

    # Train models with cross-validation
    print("  🤖 Training models for relative scoring...")

    # Owners model with cross-validation
    owners_model = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.08, max_depth=5,
        min_samples_split=15, min_samples_leaf=8,
        random_state=42, subsample=0.8, max_features='sqrt'
    )

    print("  📊 Running 5-fold cross-validation for Owners model...")
    cv_scores_owners = cross_val_score(owners_model, X_train, y_train_log, cv=5,
                                       scoring='r2', n_jobs=-1)
    print(f"     CV R² scores: {cv_scores_owners}")
    print(f"     CV R² mean: {cv_scores_owners.mean():.3f} (+/- {cv_scores_owners.std() * 2:.3f})")

    owners_model.fit(X_train, y_train_log)

    # Review model with cross-validation
    review_model = lgb.LGBMRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=6,
        num_leaves=31, min_child_samples=15, random_state=42,
        verbose=-1, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=5, reg_alpha=0.15, reg_lambda=0.15
    )

    print("  📊 Running 5-fold cross-validation for Review model...")
    cv_scores_reviews = cross_val_score(review_model, X_train, y_train_rev, cv=5,
                                        scoring='r2', n_jobs=-1)
    print(f"     CV R² scores: {cv_scores_reviews}")
    print(f"     CV R² mean: {cv_scores_reviews.mean():.3f} (+/- {cv_scores_reviews.std() * 2:.3f})")

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

    # Compute correlation matrix for all features
    corr_matrix = df[feature_cols + ['owners', 'review_ratio']].corr()

    # Store comprehensive analysis
    st.session_state.data_analysis = {
        'feature_impacts': feature_impacts,  # NEW: Core improvement data
        'feature_importance_owners': feature_importance_owners,
        'feature_importance_reviews': feature_importance_reviews,
        'correlations': corr_matrix  # Add correlation matrix
    }

    # ============================================================================
    # EVALUATE ON TEST SET
    # ============================================================================
    print("  📈 Evaluating models on test set...")

    # Predict on test set
    owners_pred_log = owners_model.predict(X_test)
    owners_pred = np.expm1(owners_pred_log)  # Convert from log scale
    reviews_pred = review_model.predict(X_test)

    # Calculate test metrics for owners model
    test_r2_owners = r2_score(y_test_log, owners_pred_log)
    test_mae_owners = mean_absolute_error(y_test_actual, owners_pred)
    test_rmse_owners = np.sqrt(mean_squared_error(y_test_actual, owners_pred))
    test_smape_owners = np.mean(2 * np.abs(owners_pred - y_test_actual) /
                                 (np.abs(owners_pred) + np.abs(y_test_actual) + 1)) * 100

    # Calculate test metrics for review model
    test_r2_reviews = r2_score(y_test_rev, reviews_pred)
    test_mae_reviews = mean_absolute_error(y_test_rev, reviews_pred)
    test_rmse_reviews = np.sqrt(mean_squared_error(y_test_rev, reviews_pred))

    # Calculate training metrics for comparison
    train_owners_pred_log = owners_model.predict(X_train)
    train_owners_pred = np.expm1(train_owners_pred_log)
    train_reviews_pred = review_model.predict(X_train)

    train_r2_owners = r2_score(y_train_log, train_owners_pred_log)
    train_mae_owners = mean_absolute_error(y_train_actual, train_owners_pred)
    train_r2_reviews = r2_score(y_train_rev, train_reviews_pred)
    train_mae_reviews = mean_absolute_error(y_train_rev, train_reviews_pred)

    print(f"  📊 Owners Model Performance:")
    print(f"     Train R²: {train_r2_owners:.3f} | Test R²: {test_r2_owners:.3f}")
    print(f"     Test MAE: {test_mae_owners:,.0f} owners | Test SMAPE: {test_smape_owners:.1f}%")
    print(f"  📊 Review Model Performance:")
    print(f"     Train R²: {train_r2_reviews:.3f} | Test R²: {test_r2_reviews:.3f}")
    print(f"     Test MAE: {test_mae_reviews:.3f}")

    # ============================================================================
    # DETAILED MODEL ANALYSIS FOR TECHNICAL REPORT
    # ============================================================================

    # Residual analysis
    train_residuals_owners = y_train_actual - train_owners_pred
    test_residuals_owners = y_test_actual - owners_pred
    train_residuals_reviews = y_train_rev - train_reviews_pred
    test_residuals_reviews = y_test_rev - reviews_pred

    # Prediction distributions
    train_pred_stats_owners = {
        'min': float(train_owners_pred.min()),
        'max': float(train_owners_pred.max()),
        'mean': float(train_owners_pred.mean()),
        'median': float(np.median(train_owners_pred)),
        'std': float(train_owners_pred.std())
    }
    test_pred_stats_owners = {
        'min': float(owners_pred.min()),
        'max': float(owners_pred.max()),
        'mean': float(owners_pred.mean()),
        'median': float(np.median(owners_pred)),
        'std': float(owners_pred.std())
    }

    # Feature correlations with targets
    feature_target_corr_owners = []
    feature_target_corr_reviews = []
    for feat in X_train.columns[:20]:  # Top 20 features
        if feat in df.columns:
            corr_own = df[[feat, 'owners']].corr().iloc[0, 1]
            corr_rev = df[[feat, 'review_ratio']].corr().iloc[0, 1]
            feature_target_corr_owners.append({'feature': feat, 'correlation': float(corr_own)})
            feature_target_corr_reviews.append({'feature': feat, 'correlation': float(corr_rev)})

    # Sort by absolute correlation
    feature_target_corr_owners = sorted(feature_target_corr_owners,
                                        key=lambda x: abs(x['correlation']), reverse=True)[:10]
    feature_target_corr_reviews = sorted(feature_target_corr_reviews,
                                         key=lambda x: abs(x['correlation']), reverse=True)[:10]

    # Model complexity metrics
    owners_model_complexity = {
        'n_features_used': len(X_train.columns),
        'n_estimators': owners_model.n_estimators,
        'max_depth': owners_model.max_depth,
        'min_samples_split': owners_model.min_samples_split,
        'min_samples_leaf': owners_model.min_samples_leaf,
        'n_features_considered_per_split': 'sqrt',
        'total_trees': owners_model.n_estimators
    }

    review_model_complexity = {
        'n_features_used': len(X_train.columns),
        'n_estimators': review_model.n_estimators,
        'max_depth': review_model.max_depth,
        'num_leaves': review_model.num_leaves,
        'min_samples_per_leaf': review_model.min_child_samples
    }

    # ============================================================================
    # GENERATE DETAILED TRAINING REPORT
    # ============================================================================
    print("  📝 Generating detailed technical report...")

    report = {
        'metadata': {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'dataset_size': len(df),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'num_features': len(feature_cols),
            'target_variables': ['owners (log-transformed)', 'review_ratio']
        },
        'data_overview': {
            'owners_range': f"{y_owners.min():,.0f} - {y_owners.max():,.0f}",
            'owners_median': f"{y_owners.median():,.0f}",
            'review_ratio_range': f"{y_reviews.min():.3f} - {y_reviews.max():.3f}",
            'review_ratio_mean': f"{y_reviews.mean():.3f}"
        },
        'models': {
            'owners_model': {
                'type': 'GradientBoostingRegressor',
                'purpose': 'Predict game ownership (user reach)',
                'architecture': {
                    'n_estimators': 150,
                    'learning_rate': 0.08,
                    'max_depth': 5,
                    'min_samples_split': 15,
                    'min_samples_leaf': 8,
                    'subsample': 0.8,
                    'max_features': 'sqrt'
                },
                'target_transformation': 'log1p (handles 10K-200M range)',
                'cross_validation': {
                    'folds': 5,
                    'cv_r2_mean': float(cv_scores_owners.mean()),
                    'cv_r2_std': float(cv_scores_owners.std()),
                    'cv_r2_scores': cv_scores_owners.tolist()
                },
                'train_performance': {
                    'r2': float(train_r2_owners),
                    'mae': float(train_mae_owners)
                },
                'test_performance': {
                    'r2': float(test_r2_owners),
                    'mae': float(test_mae_owners),
                    'rmse': float(test_rmse_owners),
                    'smape': float(test_smape_owners)
                },
                'interpretation': f"Test R² of {test_r2_owners:.3f} means the model explains {test_r2_owners*100:.1f}% of variance in log(owners). SMAPE of {test_smape_owners:.1f}% indicates typical prediction error.",
                'model_complexity': owners_model_complexity,
                'residual_analysis': {
                    'train': {
                        'mean': float(train_residuals_owners.mean()),
                        'std': float(train_residuals_owners.std()),
                        'min': float(train_residuals_owners.min()),
                        'max': float(train_residuals_owners.max()),
                        'median': float(np.median(train_residuals_owners))
                    },
                    'test': {
                        'mean': float(test_residuals_owners.mean()),
                        'std': float(test_residuals_owners.std()),
                        'min': float(test_residuals_owners.min()),
                        'max': float(test_residuals_owners.max()),
                        'median': float(np.median(test_residuals_owners))
                    }
                },
                'prediction_distribution': {
                    'train': train_pred_stats_owners,
                    'test': test_pred_stats_owners,
                    'actual_train': {
                        'min': float(y_train_actual.min()),
                        'max': float(y_train_actual.max()),
                        'mean': float(y_train_actual.mean()),
                        'median': float(np.median(y_train_actual)),
                        'std': float(y_train_actual.std())
                    },
                    'actual_test': {
                        'min': float(y_test_actual.min()),
                        'max': float(y_test_actual.max()),
                        'mean': float(y_test_actual.mean()),
                        'median': float(np.median(y_test_actual)),
                        'std': float(y_test_actual.std())
                    }
                },
                'feature_target_correlations': feature_target_corr_owners
            },
            'review_model': {
                'type': 'LightGBM',
                'purpose': 'Predict review quality (user satisfaction)',
                'architecture': {
                    'n_estimators': 150,
                    'learning_rate': 0.05,
                    'max_depth': 6,
                    'num_leaves': 31,
                    'min_child_samples': 15,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'reg_alpha': 0.15,
                    'reg_lambda': 0.15
                },
                'cross_validation': {
                    'folds': 5,
                    'cv_r2_mean': float(cv_scores_reviews.mean()),
                    'cv_r2_std': float(cv_scores_reviews.std()),
                    'cv_r2_scores': cv_scores_reviews.tolist()
                },
                'train_performance': {
                    'r2': float(train_r2_reviews),
                    'mae': float(train_mae_reviews)
                },
                'test_performance': {
                    'r2': float(test_r2_reviews),
                    'mae': float(test_mae_reviews),
                    'rmse': float(test_rmse_reviews)
                },
                'interpretation': f"Test R² of {test_r2_reviews:.3f} means the model explains {test_r2_reviews*100:.1f}% of variance in review ratio. MAE of {test_mae_reviews:.3f} is the average prediction error.",
                'model_complexity': review_model_complexity,
                'residual_analysis': {
                    'train': {
                        'mean': float(train_residuals_reviews.mean()),
                        'std': float(train_residuals_reviews.std()),
                        'min': float(train_residuals_reviews.min()),
                        'max': float(train_residuals_reviews.max()),
                        'median': float(np.median(train_residuals_reviews))
                    },
                    'test': {
                        'mean': float(test_residuals_reviews.mean()),
                        'std': float(test_residuals_reviews.std()),
                        'min': float(test_residuals_reviews.min()),
                        'max': float(test_residuals_reviews.max()),
                        'median': float(np.median(test_residuals_reviews))
                    }
                },
                'prediction_distribution': {
                    'train': {
                        'min': float(train_reviews_pred.min()),
                        'max': float(train_reviews_pred.max()),
                        'mean': float(train_reviews_pred.mean()),
                        'median': float(np.median(train_reviews_pred)),
                        'std': float(train_reviews_pred.std())
                    },
                    'test': {
                        'min': float(reviews_pred.min()),
                        'max': float(reviews_pred.max()),
                        'mean': float(reviews_pred.mean()),
                        'median': float(np.median(reviews_pred)),
                        'std': float(reviews_pred.std())
                    },
                    'actual_train': {
                        'min': float(y_train_rev.min()),
                        'max': float(y_train_rev.max()),
                        'mean': float(y_train_rev.mean()),
                        'median': float(np.median(y_train_rev)),
                        'std': float(y_train_rev.std())
                    },
                    'actual_test': {
                        'min': float(y_test_rev.min()),
                        'max': float(y_test_rev.max()),
                        'mean': float(y_test_rev.mean()),
                        'median': float(np.median(y_test_rev)),
                        'std': float(y_test_rev.std())
                    }
                },
                'feature_target_correlations': feature_target_corr_reviews
            }
        },
        'feature_analysis': {
            'top_10_features_owners': feature_importance_owners.head(10).to_dict('records'),
            'top_10_features_reviews': feature_importance_reviews.head(10).to_dict('records'),
            'top_5_positive_impacts_owners': [
                {
                    'feature': feat,
                    'improvement_pct': impact.get('owners_improvement_pct', 0),
                    'type': impact['type']
                }
                for feat, impact in sorted(feature_impacts.items(),
                                         key=lambda x: x[1].get('owners_improvement_pct', 0),
                                         reverse=True)[:5]
            ],
            'top_5_positive_impacts_reviews': [
                {
                    'feature': feat,
                    'improvement_pct': impact.get('reviews_improvement_pct', 0),
                    'type': impact['type']
                }
                for feat, impact in sorted(feature_impacts.items(),
                                         key=lambda x: x[1].get('reviews_improvement_pct', 0),
                                         reverse=True)[:5]
            ]
        },
        'model_insights': {
            'what_models_learned': [
                "Owners model learned to predict game reach using platform support, genres, pricing, and game features",
                "Review model learned to predict user satisfaction based on game quality indicators and feature combinations",
                f"Both models use {len(feature_cols)} features from actual Steam data to make predictions"
            ],
            'why_this_approach': [
                "Log transformation for owners handles the wide 10K-200M range and makes predictions more stable",
                "Gradient Boosting for owners provides better handling of non-linear relationships",
                "LightGBM for reviews is faster and handles sparse features (tags) efficiently",
                "Cross-validation ensures models generalize well to unseen games",
                "80/20 train/test split validates real-world performance"
            ],
            'model_behavior': [
                f"Owners model: CV R² = {cv_scores_owners.mean():.3f}, Test R² = {test_r2_owners:.3f} (overfitting check: {'minimal' if abs(cv_scores_owners.mean() - test_r2_owners) < 0.05 else 'present'})",
                f"Review model: CV R² = {cv_scores_reviews.mean():.3f}, Test R² = {test_r2_reviews:.3f} (overfitting check: {'minimal' if abs(cv_scores_reviews.mean() - test_r2_reviews) < 0.05 else 'present'})",
                "Models focus on IMPROVEMENT ANALYSIS rather than exact prediction accuracy",
                "Feature impact analysis shows what changes improve outcomes, not just correlation"
            ],
            'recommendations_for_improvement': []
        }
    }

    # Add recommendations based on model performance
    if test_r2_owners < 0.3:
        report['model_insights']['recommendations_for_improvement'].append(
            "Owners model has low R². Consider: (1) adding more features, (2) feature engineering, (3) collecting more data"
        )
    if test_r2_reviews < 0.3:
        report['model_insights']['recommendations_for_improvement'].append(
            "Review model has low R². Reviews may be inherently unpredictable from available features."
        )
    if abs(train_r2_owners - test_r2_owners) > 0.1:
        report['model_insights']['recommendations_for_improvement'].append(
            "Owners model shows overfitting. Consider: (1) increasing regularization, (2) reducing model complexity"
        )
    if abs(train_r2_reviews - test_r2_reviews) > 0.1:
        report['model_insights']['recommendations_for_improvement'].append(
            "Review model shows overfitting. Consider: (1) increasing reg_alpha/reg_lambda, (2) reducing max_depth"
        )

    # Save report to data folder
    os.makedirs('data', exist_ok=True)
    report_path = 'data/training_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Also create a markdown version for readability
    markdown_report = f"""# Game Launch IDSS - Training Report

**Generated:** {report['metadata']['timestamp']}

## Dataset Overview
- **Total Games:** {report['metadata']['dataset_size']:,}
- **Training Set:** {report['metadata']['train_size']:,} games (80%)
- **Test Set:** {report['metadata']['test_size']:,} games (20%)
- **Features Used:** {report['metadata']['num_features']}

### Data Ranges
- **Owners:** {report['data_overview']['owners_range']} (median: {report['data_overview']['owners_median']})
- **Review Ratio:** {report['data_overview']['review_ratio_range']} (mean: {report['data_overview']['review_ratio_mean']})

---

## Model 1: Owners Prediction (GradientBoostingRegressor)

### Purpose
{report['models']['owners_model']['purpose']}

### Architecture
- **Estimators:** {report['models']['owners_model']['architecture']['n_estimators']}
- **Learning Rate:** {report['models']['owners_model']['architecture']['learning_rate']}
- **Max Depth:** {report['models']['owners_model']['architecture']['max_depth']}
- **Regularization:** subsample={report['models']['owners_model']['architecture']['subsample']}, min_samples_split={report['models']['owners_model']['architecture']['min_samples_split']}

### Performance

**Cross-Validation (5-fold):**
- Mean R²: {report['models']['owners_model']['cross_validation']['cv_r2_mean']:.3f} ± {report['models']['owners_model']['cross_validation']['cv_r2_std']:.3f}

**Training Set:**
- R²: {report['models']['owners_model']['train_performance']['r2']:.3f}
- MAE: {report['models']['owners_model']['train_performance']['mae']:,.0f} owners

**Test Set (Unseen Data):**
- R²: {report['models']['owners_model']['test_performance']['r2']:.3f}
- MAE: {report['models']['owners_model']['test_performance']['mae']:,.0f} owners
- RMSE: {report['models']['owners_model']['test_performance']['rmse']:,.0f} owners
- SMAPE: {report['models']['owners_model']['test_performance']['smape']:.1f}%

**Interpretation:**
{report['models']['owners_model']['interpretation']}

### Model Complexity
- **Features Used:** {report['models']['owners_model']['model_complexity']['n_features_used']}
- **Total Trees:** {report['models']['owners_model']['model_complexity']['total_trees']}
- **Max Tree Depth:** {report['models']['owners_model']['model_complexity']['max_depth']}
- **Min Samples per Split:** {report['models']['owners_model']['model_complexity']['min_samples_split']}
- **Min Samples per Leaf:** {report['models']['owners_model']['model_complexity']['min_samples_leaf']}
- **Features Considered per Split:** {report['models']['owners_model']['model_complexity']['n_features_considered_per_split']}

**Analysis:** This configuration creates {report['models']['owners_model']['model_complexity']['total_trees']} decision trees, each with max depth {report['models']['owners_model']['model_complexity']['max_depth']}. The model considers √n features at each split, providing good balance between model diversity and computational efficiency.

### Residual Analysis

**Training Set Residuals:**
- Mean: {report['models']['owners_model']['residual_analysis']['train']['mean']:,.0f} (should be ~0)
- Std Dev: {report['models']['owners_model']['residual_analysis']['train']['std']:,.0f}
- Range: [{report['models']['owners_model']['residual_analysis']['train']['min']:,.0f}, {report['models']['owners_model']['residual_analysis']['train']['max']:,.0f}]
- Median: {report['models']['owners_model']['residual_analysis']['train']['median']:,.0f}

**Test Set Residuals:**
- Mean: {report['models']['owners_model']['residual_analysis']['test']['mean']:,.0f} (should be ~0)
- Std Dev: {report['models']['owners_model']['residual_analysis']['test']['std']:,.0f}
- Range: [{report['models']['owners_model']['residual_analysis']['test']['min']:,.0f}, {report['models']['owners_model']['residual_analysis']['test']['max']:,.0f}]
- Median: {report['models']['owners_model']['residual_analysis']['test']['median']:,.0f}

**Analysis:** Mean residuals close to 0 indicate unbiased predictions. Test residual std dev of {report['models']['owners_model']['residual_analysis']['test']['std']:,.0f} shows typical prediction error magnitude.

### Prediction Distribution

**Training Set:**
- Predictions range: {report['models']['owners_model']['prediction_distribution']['train']['min']:,.0f} - {report['models']['owners_model']['prediction_distribution']['train']['max']:,.0f}
- Mean predicted: {report['models']['owners_model']['prediction_distribution']['train']['mean']:,.0f} (actual: {report['models']['owners_model']['prediction_distribution']['actual_train']['mean']:,.0f})
- Std dev predicted: {report['models']['owners_model']['prediction_distribution']['train']['std']:,.0f} (actual: {report['models']['owners_model']['prediction_distribution']['actual_train']['std']:,.0f})

**Test Set:**
- Predictions range: {report['models']['owners_model']['prediction_distribution']['test']['min']:,.0f} - {report['models']['owners_model']['prediction_distribution']['test']['max']:,.0f}
- Mean predicted: {report['models']['owners_model']['prediction_distribution']['test']['mean']:,.0f} (actual: {report['models']['owners_model']['prediction_distribution']['actual_test']['mean']:,.0f})
- Std dev predicted: {report['models']['owners_model']['prediction_distribution']['test']['std']:,.0f} (actual: {report['models']['owners_model']['prediction_distribution']['actual_test']['std']:,.0f})

**Analysis:** Prediction distribution should match actual distribution. Significant differences indicate model bias or inability to capture full variance.

### Top 10 Features by Importance (Model-Based)
"""

    for i, feat in enumerate(report['feature_analysis']['top_10_features_owners'][:10], 1):
        markdown_report += f"{i}. **{feat['feature']}**: {feat['importance']:.4f}\n"

    markdown_report += """

### Top 10 Features by Target Correlation
"""

    for i, feat in enumerate(report['models']['owners_model']['feature_target_correlations'][:10], 1):
        markdown_report += f"{i}. **{feat['feature']}**: {feat['correlation']:.3f}\n"

    markdown_report += f"""

**Analysis:** Feature importance (model-based) shows which features the model uses most for predictions. Target correlation shows linear relationship with outcome. Both metrics are important for understanding model behavior.

---

## Model 2: Review Ratio Prediction (LightGBM)

### Purpose
{report['models']['review_model']['purpose']}

### Architecture
- **Estimators:** {report['models']['review_model']['architecture']['n_estimators']}
- **Learning Rate:** {report['models']['review_model']['architecture']['learning_rate']}
- **Max Depth:** {report['models']['review_model']['architecture']['max_depth']}
- **Num Leaves:** {report['models']['review_model']['architecture']['num_leaves']}
- **Regularization:** L1={report['models']['review_model']['architecture']['reg_alpha']}, L2={report['models']['review_model']['architecture']['reg_lambda']}

### Performance

**Cross-Validation (5-fold):**
- Mean R²: {report['models']['review_model']['cross_validation']['cv_r2_mean']:.3f} ± {report['models']['review_model']['cross_validation']['cv_r2_std']:.3f}

**Training Set:**
- R²: {report['models']['review_model']['train_performance']['r2']:.3f}
- MAE: {report['models']['review_model']['train_performance']['mae']:.3f}

**Test Set (Unseen Data):**
- R²: {report['models']['review_model']['test_performance']['r2']:.3f}
- MAE: {report['models']['review_model']['test_performance']['mae']:.3f}
- RMSE: {report['models']['review_model']['test_performance']['rmse']:.3f}

**Interpretation:**
{report['models']['review_model']['interpretation']}

### Model Complexity
- **Features Used:** {report['models']['review_model']['model_complexity']['n_features_used']}
- **Total Trees:** {report['models']['review_model']['model_complexity']['n_estimators']}
- **Max Tree Depth:** {report['models']['review_model']['model_complexity']['max_depth']}
- **Num Leaves per Tree:** {report['models']['review_model']['model_complexity']['num_leaves']}
- **Min Samples per Leaf:** {report['models']['review_model']['model_complexity']['min_samples_per_leaf']}

**Analysis:** LightGBM uses {report['models']['review_model']['model_complexity']['n_estimators']} leaf-wise trees with up to {report['models']['review_model']['model_complexity']['num_leaves']} leaves each. Leaf-wise growth is more efficient than depth-wise growth for complex patterns.

### Residual Analysis

**Training Set Residuals:**
- Mean: {report['models']['review_model']['residual_analysis']['train']['mean']:.4f} (should be ~0)
- Std Dev: {report['models']['review_model']['residual_analysis']['train']['std']:.4f}
- Range: [{report['models']['review_model']['residual_analysis']['train']['min']:.4f}, {report['models']['review_model']['residual_analysis']['train']['max']:.4f}]
- Median: {report['models']['review_model']['residual_analysis']['train']['median']:.4f}

**Test Set Residuals:**
- Mean: {report['models']['review_model']['residual_analysis']['test']['mean']:.4f} (should be ~0)
- Std Dev: {report['models']['review_model']['residual_analysis']['test']['std']:.4f}
- Range: [{report['models']['review_model']['residual_analysis']['test']['min']:.4f}, {report['models']['review_model']['residual_analysis']['test']['max']:.4f}]
- Median: {report['models']['review_model']['residual_analysis']['test']['median']:.4f}

**Analysis:** Mean residuals close to 0 indicate unbiased predictions. Review ratio residuals are on 0-1 scale.

### Prediction Distribution

**Training Set:**
- Predictions range: {report['models']['review_model']['prediction_distribution']['train']['min']:.3f} - {report['models']['review_model']['prediction_distribution']['train']['max']:.3f}
- Mean predicted: {report['models']['review_model']['prediction_distribution']['train']['mean']:.3f} (actual: {report['models']['review_model']['prediction_distribution']['actual_train']['mean']:.3f})
- Std dev predicted: {report['models']['review_model']['prediction_distribution']['train']['std']:.3f} (actual: {report['models']['review_model']['prediction_distribution']['actual_train']['std']:.3f})

**Test Set:**
- Predictions range: {report['models']['review_model']['prediction_distribution']['test']['min']:.3f} - {report['models']['review_model']['prediction_distribution']['test']['max']:.3f}
- Mean predicted: {report['models']['review_model']['prediction_distribution']['test']['mean']:.3f} (actual: {report['models']['review_model']['prediction_distribution']['actual_test']['mean']:.3f})
- Std dev predicted: {report['models']['review_model']['prediction_distribution']['test']['std']:.3f} (actual: {report['models']['review_model']['prediction_distribution']['actual_test']['std']:.3f})

**Analysis:** Prediction distribution alignment with actual values indicates model's ability to capture the full range of outcomes.

### Top 10 Features by Importance (Model-Based)
"""

    for i, feat in enumerate(report['feature_analysis']['top_10_features_reviews'][:10], 1):
        markdown_report += f"{i}. **{feat['feature']}**: {feat['importance']:.4f}\n"

    markdown_report += """

### Top 10 Features by Target Correlation
"""

    for i, feat in enumerate(report['models']['review_model']['feature_target_correlations'][:10], 1):
        markdown_report += f"{i}. **{feat['feature']}**: {feat['correlation']:.3f}\n"

    markdown_report += f"""

**Analysis:** Comparing model-based importance with direct correlation helps identify non-linear relationships that the model captures.

---

## Feature Impact Analysis

### Top 5 Features That Improve Owners
"""

    for item in report['feature_analysis']['top_5_positive_impacts_owners']:
        markdown_report += f"- **{item['feature']}**: +{item['improvement_pct']:.1f}% improvement\n"

    markdown_report += """

### Top 5 Features That Improve Review Ratio
"""

    for item in report['feature_analysis']['top_5_positive_impacts_reviews']:
        markdown_report += f"- **{item['feature']}**: +{item['improvement_pct']:.1f}% improvement\n"

    markdown_report += """

---

## Model Insights

### What the Models Learned
"""

    for insight in report['model_insights']['what_models_learned']:
        markdown_report += f"- {insight}\n"

    markdown_report += """

### Why This Approach
"""

    for reason in report['model_insights']['why_this_approach']:
        markdown_report += f"- {reason}\n"

    markdown_report += """

### Model Behavior
"""

    for behavior in report['model_insights']['model_behavior']:
        markdown_report += f"- {behavior}\n"

    if report['model_insights']['recommendations_for_improvement']:
        markdown_report += """

### Recommendations for Improvement
"""
        for rec in report['model_insights']['recommendations_for_improvement']:
            markdown_report += f"- ⚠️ {rec}\n"

    markdown_report += """

---

## Conclusion

These models analyze actual Steam data to provide:
1. **Predictive insights** - estimate potential owners and review quality
2. **Improvement analysis** - show what features improve outcomes
3. **Data-driven recommendations** - suggest optimal game configurations

The models prioritize **understanding what improves game success** over perfect prediction accuracy.
"""

    markdown_path = 'data/training_report.md'
    with open(markdown_path, 'w') as f:
        f.write(markdown_report)

    print(f"  ✅ Training report saved to {report_path} and {markdown_path}")
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
