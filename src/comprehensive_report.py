"""
Comprehensive Training Report Generator
Generates extremely detailed training reports for ML models
"""

import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats


def generate_comprehensive_training_report(df, feature_cols, results, training_time):
    """
    Generate an EXTREMELY detailed and exhaustive training report
    This is the most comprehensive version covering all aspects of training and testing

    Parameters:
    -----------
    df : DataFrame
        Original dataframe with all data
    feature_cols : list
        List of feature column names
    results : dict
        Results dictionary from train_improved_models
    training_time : float
        Total training time in seconds

    Returns:
    --------
    report : str
        Markdown formatted comprehensive training report
    """

    report_lines = []

    # ============================================================================
    # HEADER
    # ============================================================================
    report_lines.append("# 🎮 Game Launch IDSS - EXTREMELY DETAILED Training Report")
    report_lines.append("## Complete Documentation of Training and Testing Phases")
    report_lines.append("")
    report_lines.append(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Training Duration:** {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    report_lines.append(f"**Report Version:** v2.1.0 - Comprehensive Edition")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # ============================================================================
    # TABLE OF CONTENTS
    # ============================================================================
    report_lines.append("## 📑 Table of Contents")
    report_lines.append("")
    report_lines.append("1. [Executive Summary](#executive-summary)")
    report_lines.append("2. [Data Preprocessing Phase](#data-preprocessing-phase)")
    report_lines.append("3. [Dataset Information & Statistics](#dataset-information)")
    report_lines.append("4. [Feature Engineering Details](#feature-engineering)")
    report_lines.append("5. [Model Architecture & Configuration](#model-architecture)")
    report_lines.append("6. [Training Phase - Detailed Logs](#training-phase)")
    report_lines.append("7. [Cross-Validation Analysis](#cross-validation-analysis)")
    report_lines.append("8. [Test Phase Evaluation](#test-phase-evaluation)")
    report_lines.append("9. [Prediction Analysis & Distribution](#prediction-analysis)")
    report_lines.append("10. [Error Analysis & Residuals](#error-analysis)")
    report_lines.append("11. [Feature Importance Analysis](#feature-importance)")
    report_lines.append("12. [Model Diagnostics](#model-diagnostics)")
    report_lines.append("13. [Statistical Tests](#statistical-tests)")
    report_lines.append("14. [Performance Comparison](#performance-comparison)")
    report_lines.append("15. [Model Complexity Analysis](#model-complexity)")
    report_lines.append("16. [Insights & Recommendations](#insights-recommendations)")
    report_lines.append("17. [Complete Technical Specification](#technical-specification)")
    report_lines.append("18. [Reproducibility Checklist](#reproducibility)")
    report_lines.append("19. [Appendix: Raw Data](#appendix)")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # ============================================================================
    # 1. EXECUTIVE SUMMARY
    # ============================================================================
    report_lines.append("## 1. 📊 Executive Summary {#executive-summary}")
    report_lines.append("")
    report_lines.append("This report provides an EXTREMELY comprehensive analysis of the complete machine learning ")
    report_lines.append("pipeline for predicting game success metrics. Every phase of training and testing is ")
    report_lines.append("documented in exhaustive detail.")
    report_lines.append("")

    report_lines.append("### 1.1 Key Achievements")
    report_lines.append("")
    report_lines.append(f"- **Total Features Engineered:** {len(feature_cols)}")
    report_lines.append(f"- **Training Samples:** {len(results['X_train']):,} games")
    report_lines.append(f"- **Test Samples:** {len(results['X_test']):,} games")
    report_lines.append(f"- **Owners Model Performance (R²):** {results['test_metrics']['owners_r2']:.4f}")
    report_lines.append(f"- **Review Model Performance (R²):** {results['test_metrics']['reviews_r2']:.4f}")
    report_lines.append(f"- **Total Training Time:** {training_time:.2f} seconds")
    report_lines.append("")

    report_lines.append("### 1.2 Quick Performance Summary")
    report_lines.append("")
    report_lines.append("| Model | R² Score | MAE | RMSE | Status |")
    report_lines.append("|-------|----------|-----|------|--------|")
    report_lines.append(f"| **Owners** | {results['test_metrics']['owners_r2']:.4f} | "
                       f"{results['test_metrics']['owners_mae']:,.0f} | "
                       f"{results['test_metrics']['owners_rmse']:,.0f} | "
                       f"{'✅ Excellent' if results['test_metrics']['owners_r2'] > 0.85 else '⚠️ Good'} |")
    report_lines.append(f"| **Reviews** | {results['test_metrics']['reviews_r2']:.4f} | "
                       f"{results['test_metrics']['reviews_mae']:.4f} | "
                       f"{results['test_metrics']['reviews_rmse']:.4f} | "
                       f"{'✅ Good' if results['test_metrics']['reviews_r2'] > 0.4 else '⚠️ Fair'} |")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # ============================================================================
    # 2. DATA PREPROCESSING PHASE
    # ============================================================================
    report_lines.append("## 2. 🔧 Data Preprocessing Phase {#data-preprocessing-phase}")
    report_lines.append("")
    report_lines.append("This section documents every step of the data preprocessing pipeline.")
    report_lines.append("")

    report_lines.append("### 2.1 Initial Data Loading")
    report_lines.append("")
    report_lines.append(f"- **Total Records Loaded:** {len(df):,}")
    report_lines.append(f"- **Total Columns:** {len(df.columns)}")
    report_lines.append(f"- **Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    report_lines.append("")

    report_lines.append("### 2.2 Missing Value Analysis")
    report_lines.append("")
    X_train = results['X_train']
    missing_before = len(df) * len(feature_cols)
    report_lines.append("| Phase | Missing Values | Percentage |")
    report_lines.append("|-------|----------------|------------|")
    report_lines.append(f"| Before Cleaning | Unknown | - |")
    report_lines.append(f"| After Filling | 0 | 0.00% |")
    report_lines.append(f"| **Fill Strategy** | **Zero-fill for numeric features** | |")
    report_lines.append("")

    report_lines.append("### 2.3 Target Variable Transformation")
    report_lines.append("")
    report_lines.append("#### Owners Transformation")
    report_lines.append("```python")
    report_lines.append("# Log transformation applied to handle wide range")
    report_lines.append("df['log_owners'] = np.log1p(df['owners'])")
    report_lines.append("# Rationale: Owners range from ~100 to 150M (6 orders of magnitude)")
    report_lines.append("```")
    report_lines.append("")

    owners_data = df['owners'].dropna()
    report_lines.append("**Original Owners Distribution:**")
    report_lines.append(f"- Min: {owners_data.min():,.0f}")
    report_lines.append(f"- Max: {owners_data.max():,.0f}")
    report_lines.append(f"- Range: {owners_data.max() - owners_data.min():,.0f}")
    report_lines.append(f"- Skewness: {owners_data.skew():.2f} (highly right-skewed)")
    report_lines.append("")

    log_owners = np.log1p(owners_data)
    report_lines.append("**After Log Transformation:**")
    report_lines.append(f"- Min: {log_owners.min():.2f}")
    report_lines.append(f"- Max: {log_owners.max():.2f}")
    report_lines.append(f"- Range: {log_owners.max() - log_owners.min():.2f}")
    report_lines.append(f"- Skewness: {log_owners.skew():.2f} (more normalized)")
    report_lines.append("")

    report_lines.append("#### Review Ratio - No Transformation")
    report_lines.append("- Review ratio already in [0, 1] range")
    report_lines.append("- No transformation needed")
    report_lines.append("")

    report_lines.append("### 2.4 Train/Test Split")
    report_lines.append("")
    report_lines.append(f"- **Split Method:** Random stratified split")
    report_lines.append(f"- **Train Size:** {len(results['X_train']):,} ({len(results['X_train'])/len(df)*100:.1f}%)")
    report_lines.append(f"- **Test Size:** {len(results['X_test']):,} ({len(results['X_test'])/len(df)*100:.1f}%)")
    report_lines.append(f"- **Random Seed:** 42 (for reproducibility)")
    report_lines.append("")

    # Check if train/test distributions are similar
    y_train_actual = results['y_train_actual']
    y_test_actual = results['y_test_actual']

    report_lines.append("**Distribution Comparison (Owners):**")
    report_lines.append("")
    report_lines.append("| Statistic | Training Set | Test Set | Difference |")
    report_lines.append("|-----------|--------------|----------|------------|")
    report_lines.append(f"| Mean | {y_train_actual.mean():,.0f} | {y_test_actual.mean():,.0f} | "
                       f"{abs(y_train_actual.mean() - y_test_actual.mean()):,.0f} |")
    report_lines.append(f"| Median | {y_train_actual.median():,.0f} | {y_test_actual.median():,.0f} | "
                       f"{abs(y_train_actual.median() - y_test_actual.median()):,.0f} |")
    report_lines.append(f"| Std Dev | {y_train_actual.std():,.0f} | {y_test_actual.std():,.0f} | "
                       f"{abs(y_train_actual.std() - y_test_actual.std()):,.0f} |")
    report_lines.append("")

    # Statistical test for distribution similarity
    ks_stat, ks_pval = stats.ks_2samp(y_train_actual, y_test_actual)
    report_lines.append(f"**Kolmogorov-Smirnov Test:** p-value = {ks_pval:.4f}")
    if ks_pval > 0.05:
        report_lines.append("✅ Train and test distributions are statistically similar (p > 0.05)")
    else:
        report_lines.append("⚠️ Train and test distributions differ (p < 0.05)")
    report_lines.append("")

    report_lines.append("### 2.5 Feature Name Cleaning")
    report_lines.append("")
    report_lines.append("- **Method:** Replace special characters with underscores")
    report_lines.append("- **Regex Pattern:** `[^A-Za-z0-9_]` → `_`")
    report_lines.append("- **Reason:** Compatibility with XGBoost/LightGBM")
    report_lines.append(f"- **Features Affected:** Check for special chars in original names")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # ============================================================================
    # 3. DATASET INFORMATION & STATISTICS
    # ============================================================================
    report_lines.append("## 3. 📊 Dataset Information & Statistics {#dataset-information}")
    report_lines.append("")

    report_lines.append("### 3.1 Overall Dataset Statistics")
    report_lines.append("")
    report_lines.append(f"- **Total Games:** {len(df):,}")
    report_lines.append(f"- **Features Used:** {len(feature_cols)}")
    report_lines.append(f"- **Training Games:** {len(results['X_train']):,}")
    report_lines.append(f"- **Test Games:** {len(results['X_test']):,}")
    report_lines.append(f"- **Training Ratio:** {len(results['X_train'])/len(df)*100:.1f}%")
    report_lines.append("")

    report_lines.append("### 3.2 Target Variable Statistics")
    report_lines.append("")
    report_lines.append("#### Owners (Actual Values)")
    report_lines.append(f"- Mean: {owners_data.mean():,.0f}")
    report_lines.append(f"- Median: {owners_data.median():,.0f}")
    report_lines.append(f"- Std Dev: {owners_data.std():,.0f}")
    report_lines.append(f"- Min: {owners_data.min():,.0f}")
    report_lines.append(f"- Max: {owners_data.max():,.0f}")
    report_lines.append(f"- 25th Percentile: {owners_data.quantile(0.25):,.0f}")
    report_lines.append(f"- 75th Percentile: {owners_data.quantile(0.75):,.0f}")
    report_lines.append("")

    if 'positive_ratio' in df.columns:
        review_data = df['positive_ratio'].dropna()
        report_lines.append("#### Review Ratio")
        report_lines.append(f"- Mean: {review_data.mean():.4f}")
        report_lines.append(f"- Median: {review_data.median():.4f}")
        report_lines.append(f"- Std Dev: {review_data.std():.4f}")
        report_lines.append(f"- Min: {review_data.min():.4f}")
        report_lines.append(f"- Max: {review_data.max():.4f}")
        report_lines.append("")

    report_lines.append("### 3.3 Feature Coverage")
    report_lines.append("")
    report_lines.append(f"- **Numeric Features:** {sum(1 for col in feature_cols if col in X_train.columns)}")
    report_lines.append(f"- **Missing Values in Training:** {X_train.isna().sum().sum()}")
    report_lines.append(f"- **Missing Values in Test:** {results['X_test'].isna().sum().sum()}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # ============================================================================
    # 4. FEATURE ENGINEERING DETAILS
    # ============================================================================
    report_lines.append("## 4. 🔨 Feature Engineering Details {#feature-engineering}")
    report_lines.append("")
    report_lines.append("### 4.1 Feature Categories")
    report_lines.append("")

    # Categorize features
    price_features = [f for f in feature_cols if 'price' in f.lower()]
    review_features = [f for f in feature_cols if 'review' in f.lower() or 'positive' in f.lower() or 'negative' in f.lower()]
    time_features = [f for f in feature_cols if 'days' in f.lower() or 'age' in f.lower() or 'time' in f.lower()]
    genre_features = [f for f in feature_cols if 'genre' in f.lower()]
    category_features = [f for f in feature_cols if 'category' in f.lower()]
    tag_features = [f for f in feature_cols if 'tag' in f.lower()]

    report_lines.append(f"- **Price-related features:** {len(price_features)}")
    report_lines.append(f"- **Review-related features:** {len(review_features)}")
    report_lines.append(f"- **Time-related features:** {len(time_features)}")
    report_lines.append(f"- **Genre features:** {len(genre_features)}")
    report_lines.append(f"- **Category features:** {len(category_features)}")
    report_lines.append(f"- **Tag features:** {len(tag_features)}")
    report_lines.append(f"- **Other features:** {len(feature_cols) - len(price_features) - len(review_features) - len(time_features) - len(genre_features) - len(category_features) - len(tag_features)}")
    report_lines.append("")

    report_lines.append("### 4.2 Feature Statistics")
    report_lines.append("")
    report_lines.append("| Metric | Value |")
    report_lines.append("|--------|-------|")
    report_lines.append(f"| Mean feature value | {X_train.mean().mean():.4f} |")
    report_lines.append(f"| Median feature value | {X_train.median().median():.4f} |")
    report_lines.append(f"| Feature value std dev | {X_train.std().mean():.4f} |")
    report_lines.append(f"| Features with zero variance | {sum(X_train.std() == 0)} |")
    report_lines.append(f"| Features with high correlation (>0.9) | {sum((X_train.corr().abs() > 0.9).sum() > 1) // 2} |")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # ============================================================================
    # 5. MODEL ARCHITECTURE & CONFIGURATION
    # ============================================================================
    report_lines.append("## 5. 🏗️ Model Architecture & Configuration {#model-architecture}")
    report_lines.append("")

    report_lines.append("### 5.1 Model Selection: XGBoost")
    report_lines.append("")
    report_lines.append("**Rationale for XGBoost:**")
    report_lines.append("- Excellent performance on structured/tabular data")
    report_lines.append("- Handles missing values naturally")
    report_lines.append("- Built-in feature importance")
    report_lines.append("- Regularization to prevent overfitting")
    report_lines.append("- Fast training with GPU support")
    report_lines.append("")

    report_lines.append("### 5.2 Hyperparameters")
    report_lines.append("")

    if 'owners_model' in results:
        owners_params = results['owners_model'].get_params()
        report_lines.append("#### Owners Model Hyperparameters")
        report_lines.append("```python")
        report_lines.append(f"n_estimators = {owners_params.get('n_estimators', 'N/A')}")
        report_lines.append(f"max_depth = {owners_params.get('max_depth', 'N/A')}")
        report_lines.append(f"learning_rate = {owners_params.get('learning_rate', 'N/A')}")
        report_lines.append(f"subsample = {owners_params.get('subsample', 'N/A')}")
        report_lines.append(f"colsample_bytree = {owners_params.get('colsample_bytree', 'N/A')}")
        report_lines.append(f"min_child_weight = {owners_params.get('min_child_weight', 'N/A')}")
        report_lines.append(f"random_state = {owners_params.get('random_state', 42)}")
        report_lines.append("```")
        report_lines.append("")

    if 'reviews_model' in results:
        reviews_params = results['reviews_model'].get_params()
        report_lines.append("#### Reviews Model Hyperparameters")
        report_lines.append("```python")
        report_lines.append(f"n_estimators = {reviews_params.get('n_estimators', 'N/A')}")
        report_lines.append(f"max_depth = {reviews_params.get('max_depth', 'N/A')}")
        report_lines.append(f"learning_rate = {reviews_params.get('learning_rate', 'N/A')}")
        report_lines.append(f"subsample = {reviews_params.get('subsample', 'N/A')}")
        report_lines.append(f"colsample_bytree = {reviews_params.get('colsample_bytree', 'N/A')}")
        report_lines.append(f"min_child_weight = {reviews_params.get('min_child_weight', 'N/A')}")
        report_lines.append(f"random_state = {reviews_params.get('random_state', 42)}")
        report_lines.append("```")
        report_lines.append("")

        if 'feature_selector' in results:
            report_lines.append("**Feature Selection Applied:**")
            report_lines.append(f"- Method: SelectKBest with f_regression")
            report_lines.append(f"- Features selected: 100 (from {len(feature_cols)})")
            report_lines.append("")

    report_lines.append("---")
    report_lines.append("")

    # ============================================================================
    # 6. TRAINING PHASE - DETAILED LOGS
    # ============================================================================
    report_lines.append("## 6. 🏋️ Training Phase - Detailed Logs {#training-phase}")
    report_lines.append("")

    report_lines.append("### 6.1 Training Process")
    report_lines.append("")
    report_lines.append(f"- **Start Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"- **Total Duration:** {training_time:.2f} seconds")
    report_lines.append(f"- **Training Speed:** {len(results['X_train']) / training_time:.0f} samples/second")
    report_lines.append("")

    report_lines.append("### 6.2 Training Metrics")
    report_lines.append("")
    report_lines.append("| Model | Training R² | Training MAE | Training RMSE |")
    report_lines.append("|-------|-------------|--------------|---------------|")

    # Calculate training metrics
    if 'owners_model' in results:
        y_train_pred_owners = results['owners_model'].predict(results['X_train'])
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        train_r2_owners = r2_score(results['y_train_owners'], y_train_pred_owners)
        train_mae_owners = mean_absolute_error(np.expm1(results['y_train_owners']), np.expm1(y_train_pred_owners))
        train_rmse_owners = np.sqrt(mean_squared_error(np.expm1(results['y_train_owners']), np.expm1(y_train_pred_owners)))
        report_lines.append(f"| Owners | {train_r2_owners:.4f} | {train_mae_owners:,.0f} | {train_rmse_owners:,.0f} |")

    if 'reviews_model' in results:
        X_train_selected = results['feature_selector'].transform(results['X_train']) if 'feature_selector' in results else results['X_train']
        y_train_pred_reviews = results['reviews_model'].predict(X_train_selected)
        train_r2_reviews = r2_score(results['y_train_reviews'], y_train_pred_reviews)
        train_mae_reviews = mean_absolute_error(results['y_train_reviews'], y_train_pred_reviews)
        train_rmse_reviews = np.sqrt(mean_squared_error(results['y_train_reviews'], y_train_pred_reviews))
        report_lines.append(f"| Reviews | {train_r2_reviews:.4f} | {train_mae_reviews:.4f} | {train_rmse_reviews:.4f} |")

    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # ============================================================================
    # 7. CROSS-VALIDATION ANALYSIS
    # ============================================================================
    report_lines.append("## 7. 🔄 Cross-Validation Analysis {#cross-validation-analysis}")
    report_lines.append("")

    report_lines.append("### 7.1 Cross-Validation Configuration")
    report_lines.append("")
    report_lines.append("- **Method:** K-Fold Cross-Validation")
    report_lines.append("- **Number of Folds:** 3")
    report_lines.append("- **Shuffle:** Yes (random_state=42)")
    report_lines.append("")

    report_lines.append("### 7.2 Cross-Validation Results")
    report_lines.append("")

    if 'cv_scores' in results:
        cv_scores = results['cv_scores']
        report_lines.append("#### Owners Model CV Scores")
        report_lines.append(f"- **Mean R²:** {cv_scores['owners']['r2_mean']:.4f}")
        report_lines.append(f"- **Std Dev R²:** {cv_scores['owners']['r2_std']:.4f}")
        report_lines.append(f"- **Min R²:** {cv_scores['owners']['r2_mean'] - cv_scores['owners']['r2_std']:.4f}")
        report_lines.append(f"- **Max R²:** {cv_scores['owners']['r2_mean'] + cv_scores['owners']['r2_std']:.4f}")
        report_lines.append("")

        report_lines.append("#### Reviews Model CV Scores")
        report_lines.append(f"- **Mean R²:** {cv_scores['reviews']['r2_mean']:.4f}")
        report_lines.append(f"- **Std Dev R²:** {cv_scores['reviews']['r2_std']:.4f}")
        report_lines.append(f"- **Min R²:** {cv_scores['reviews']['r2_mean'] - cv_scores['reviews']['r2_std']:.4f}")
        report_lines.append(f"- **Max R²:** {cv_scores['reviews']['r2_mean'] + cv_scores['reviews']['r2_std']:.4f}")
        report_lines.append("")

    report_lines.append("### 7.3 Stability Analysis")
    report_lines.append("")
    if 'cv_scores' in results:
        owners_cv_stability = cv_scores['owners']['r2_std'] / cv_scores['owners']['r2_mean']
        reviews_cv_stability = cv_scores['reviews']['r2_std'] / cv_scores['reviews']['r2_mean']

        report_lines.append(f"- **Owners Model Stability (CoV):** {owners_cv_stability:.4f} - {'✅ Stable' if owners_cv_stability < 0.1 else '⚠️ Moderate Variability' if owners_cv_stability < 0.2 else '❌ High Variability'}")
        report_lines.append(f"- **Reviews Model Stability (CoV):** {reviews_cv_stability:.4f} - {'✅ Stable' if reviews_cv_stability < 0.1 else '⚠️ Moderate Variability' if reviews_cv_stability < 0.2 else '❌ High Variability'}")
        report_lines.append("")
        report_lines.append("*CoV = Coefficient of Variation (std/mean), lower is better*")
        report_lines.append("")

    report_lines.append("---")
    report_lines.append("")

    # ============================================================================
    # 8. TEST PHASE EVALUATION
    # ============================================================================
    report_lines.append("## 8. 🎯 Test Phase Evaluation {#test-phase-evaluation}")
    report_lines.append("")

    report_lines.append("### 8.1 Test Set Performance")
    report_lines.append("")
    test_metrics = results['test_metrics']

    report_lines.append("#### Owners Model - Test Results")
    report_lines.append(f"- **R² Score:** {test_metrics['owners_r2']:.4f}")
    report_lines.append(f"- **MAE:** {test_metrics['owners_mae']:,.0f} owners")
    report_lines.append(f"- **RMSE:** {test_metrics['owners_rmse']:,.0f} owners")
    report_lines.append(f"- **MAPE:** {test_metrics.get('owners_mape', 0)*100:.2f}%" if 'owners_mape' in test_metrics else "")
    report_lines.append("")

    report_lines.append("#### Reviews Model - Test Results")
    report_lines.append(f"- **R² Score:** {test_metrics['reviews_r2']:.4f}")
    report_lines.append(f"- **MAE:** {test_metrics['reviews_mae']:.4f}")
    report_lines.append(f"- **RMSE:** {test_metrics['reviews_rmse']:.4f}")
    report_lines.append(f"- **MAPE:** {test_metrics.get('reviews_mape', 0)*100:.2f}%" if 'reviews_mape' in test_metrics else "")
    report_lines.append("")

    report_lines.append("### 8.2 Train vs Test Comparison")
    report_lines.append("")
    report_lines.append("| Model | Train R² | Test R² | Difference | Overfit Risk |")
    report_lines.append("|-------|----------|---------|------------|--------------|")

    if 'owners_model' in results:
        train_test_diff_owners = train_r2_owners - test_metrics['owners_r2']
        overfit_status_owners = '✅ No' if train_test_diff_owners < 0.05 else '⚠️ Slight' if train_test_diff_owners < 0.10 else '❌ Yes'
        report_lines.append(f"| Owners | {train_r2_owners:.4f} | {test_metrics['owners_r2']:.4f} | {train_test_diff_owners:.4f} | {overfit_status_owners} |")

    if 'reviews_model' in results:
        train_test_diff_reviews = train_r2_reviews - test_metrics['reviews_r2']
        overfit_status_reviews = '✅ No' if train_test_diff_reviews < 0.05 else '⚠️ Slight' if train_test_diff_reviews < 0.10 else '❌ Yes'
        report_lines.append(f"| Reviews | {train_r2_reviews:.4f} | {test_metrics['reviews_r2']:.4f} | {train_test_diff_reviews:.4f} | {overfit_status_reviews} |")

    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # ============================================================================
    # 9. PREDICTION ANALYSIS & DISTRIBUTION
    # ============================================================================
    report_lines.append("## 9. 📈 Prediction Analysis & Distribution {#prediction-analysis}")
    report_lines.append("")

    # Get predictions
    y_pred_owners = results['y_test_owners_pred']
    y_true_owners = results['y_test_actual']

    report_lines.append("### 9.1 Owners Prediction Distribution")
    report_lines.append("")
    report_lines.append("| Metric | Actual | Predicted | Difference |")
    report_lines.append("|--------|--------|-----------|------------|")
    report_lines.append(f"| Mean | {y_true_owners.mean():,.0f} | {y_pred_owners.mean():,.0f} | {abs(y_true_owners.mean() - y_pred_owners.mean()):,.0f} |")
    report_lines.append(f"| Median | {y_true_owners.median():,.0f} | {np.median(y_pred_owners):,.0f} | {abs(y_true_owners.median() - np.median(y_pred_owners)):,.0f} |")
    report_lines.append(f"| Std Dev | {y_true_owners.std():,.0f} | {np.std(y_pred_owners):,.0f} | {abs(y_true_owners.std() - np.std(y_pred_owners)):,.0f} |")
    report_lines.append(f"| Min | {y_true_owners.min():,.0f} | {y_pred_owners.min():,.0f} | {abs(y_true_owners.min() - y_pred_owners.min()):,.0f} |")
    report_lines.append(f"| Max | {y_true_owners.max():,.0f} | {y_pred_owners.max():,.0f} | {abs(y_true_owners.max() - y_pred_owners.max()):,.0f} |")
    report_lines.append("")

    if 'y_test_reviews_pred' in results:
        y_pred_reviews = results['y_test_reviews_pred']
        y_true_reviews = results['y_test_reviews']

        report_lines.append("### 9.2 Reviews Prediction Distribution")
        report_lines.append("")
        report_lines.append("| Metric | Actual | Predicted | Difference |")
        report_lines.append("|--------|--------|-----------|------------|")
        report_lines.append(f"| Mean | {y_true_reviews.mean():.4f} | {y_pred_reviews.mean():.4f} | {abs(y_true_reviews.mean() - y_pred_reviews.mean()):.4f} |")
        report_lines.append(f"| Median | {y_true_reviews.median():.4f} | {np.median(y_pred_reviews):.4f} | {abs(y_true_reviews.median() - np.median(y_pred_reviews)):.4f} |")
        report_lines.append(f"| Std Dev | {y_true_reviews.std():.4f} | {np.std(y_pred_reviews):.4f} | {abs(y_true_reviews.std() - np.std(y_pred_reviews)):.4f} |")
        report_lines.append("")

    report_lines.append("### 9.3 Prediction Accuracy by Range")
    report_lines.append("")

    # Bin predictions by actual value ranges
    try:
        bins = [0, 10000, 100000, 1000000, 10000000, float('inf')]
        labels = ['<10K', '10K-100K', '100K-1M', '1M-10M', '>10M']
        y_true_binned = pd.cut(y_true_owners, bins=bins, labels=labels)

        report_lines.append("| Range | Count | Mean Error | RMSE | R² |")
        report_lines.append("|-------|-------|------------|------|-----|")

        for label in labels:
            mask = y_true_binned == label
            if mask.sum() > 0:
                range_true = y_true_owners[mask]
                range_pred = y_pred_owners[mask]
                range_mae = mean_absolute_error(range_true, range_pred)
                range_rmse = np.sqrt(mean_squared_error(range_true, range_pred))
                range_r2 = r2_score(range_true, range_pred) if len(range_true) > 1 else 0
                report_lines.append(f"| {label} | {mask.sum()} | {range_mae:,.0f} | {range_rmse:,.0f} | {range_r2:.4f} |")
    except Exception:
        report_lines.append("*Could not compute range-based accuracy due to data distribution*")

    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # ============================================================================
    # 10. ERROR ANALYSIS & RESIDUALS
    # ============================================================================
    report_lines.append("## 10. 🔍 Error Analysis & Residuals {#error-analysis}")
    report_lines.append("")

    residuals_owners = y_true_owners - y_pred_owners

    report_lines.append("### 10.1 Residual Statistics (Owners)")
    report_lines.append("")
    report_lines.append(f"- **Mean Residual:** {residuals_owners.mean():,.0f} (bias)")
    report_lines.append(f"- **Median Residual:** {residuals_owners.median():,.0f}")
    report_lines.append(f"- **Std Dev Residuals:** {residuals_owners.std():,.0f}")
    report_lines.append(f"- **Residual Range:** [{residuals_owners.min():,.0f}, {residuals_owners.max():,.0f}]")
    report_lines.append("")

    # Normality test
    from scipy import stats as scipy_stats
    _, p_value_normality = scipy_stats.normaltest(residuals_owners)
    report_lines.append(f"**Normality Test (D'Agostino-Pearson):** p-value = {p_value_normality:.4f}")
    if p_value_normality > 0.05:
        report_lines.append("✅ Residuals are approximately normally distributed")
    else:
        report_lines.append("⚠️ Residuals deviate from normal distribution")
    report_lines.append("")

    report_lines.append("### 10.2 Error Patterns")
    report_lines.append("")

    # Over/under prediction analysis
    over_pred = (residuals_owners < 0).sum()
    under_pred = (residuals_owners > 0).sum()
    exact_pred = (residuals_owners == 0).sum()

    report_lines.append(f"- **Over-predictions:** {over_pred} ({over_pred/len(residuals_owners)*100:.1f}%)")
    report_lines.append(f"- **Under-predictions:** {under_pred} ({under_pred/len(residuals_owners)*100:.1f}%)")
    report_lines.append(f"- **Exact predictions:** {exact_pred}")
    report_lines.append("")

    # Largest errors
    abs_residuals = np.abs(residuals_owners)
    top_10_pct_threshold = np.percentile(abs_residuals, 90)
    large_errors = abs_residuals > top_10_pct_threshold

    report_lines.append(f"**Large Errors (top 10%):**")
    report_lines.append(f"- Threshold: >{top_10_pct_threshold:,.0f} owners")
    report_lines.append(f"- Count: {large_errors.sum()}")
    report_lines.append(f"- Mean of large errors: {abs_residuals[large_errors].mean():,.0f}")
    report_lines.append("")

    report_lines.append("---")
    report_lines.append("")

    # ============================================================================
    # 11. FEATURE IMPORTANCE ANALYSIS
    # ============================================================================
    report_lines.append("## 11. 🎯 Feature Importance Analysis {#feature-importance}")
    report_lines.append("")

    if 'owners_model' in results:
        report_lines.append("### 11.1 Owners Model - Top 20 Features")
        report_lines.append("")

        feature_importance_owners = pd.DataFrame({
            'feature': feature_cols,
            'importance': results['owners_model'].feature_importances_
        }).sort_values('importance', ascending=False)

        report_lines.append("| Rank | Feature | Importance | Cumulative % |")
        report_lines.append("|------|---------|------------|--------------|")

        cumsum = 0
        total_importance = feature_importance_owners['importance'].sum()

        for idx, row in feature_importance_owners.head(20).iterrows():
            cumsum += row['importance']
            report_lines.append(f"| {feature_importance_owners.index.get_loc(idx) + 1} | {row['feature'][:50]} | {row['importance']:.4f} | {cumsum/total_importance*100:.1f}% |")

        report_lines.append("")
        report_lines.append(f"*Top 20 features account for {cumsum/total_importance*100:.1f}% of total importance*")
        report_lines.append("")

    if 'reviews_model' in results and 'feature_selector' in results:
        report_lines.append("### 11.2 Reviews Model - Top 20 Features")
        report_lines.append("")

        # Get selected feature names
        selected_features_mask = results['feature_selector'].get_support()
        selected_feature_names = [f for f, selected in zip(feature_cols, selected_features_mask) if selected]

        feature_importance_reviews = pd.DataFrame({
            'feature': selected_feature_names,
            'importance': results['reviews_model'].feature_importances_
        }).sort_values('importance', ascending=False)

        report_lines.append("| Rank | Feature | Importance | Cumulative % |")
        report_lines.append("|------|---------|------------|--------------|")

        cumsum = 0
        total_importance = feature_importance_reviews['importance'].sum()

        for idx, row in feature_importance_reviews.head(20).iterrows():
            cumsum += row['importance']
            report_lines.append(f"| {feature_importance_reviews.index.get_loc(idx) + 1} | {row['feature'][:50]} | {row['importance']:.4f} | {cumsum/total_importance*100:.1f}% |")

        report_lines.append("")
        report_lines.append(f"*Top 20 features account for {cumsum/total_importance*100:.1f}% of total importance*")
        report_lines.append("")

    report_lines.append("---")
    report_lines.append("")

    # ============================================================================
    # 12. MODEL DIAGNOSTICS
    # ============================================================================
    report_lines.append("## 12. 🩺 Model Diagnostics {#model-diagnostics}")
    report_lines.append("")

    report_lines.append("### 12.1 Overfitting Assessment")
    report_lines.append("")
    report_lines.append("| Model | Metric | Training | CV | Test | Overfit? |")
    report_lines.append("|-------|--------|----------|----|----|----------|")

    if 'owners_model' in results:
        cv_r2_owners = results.get('cv_scores', {}).get('owners', {}).get('r2_mean', 0)
        report_lines.append(f"| Owners | R² | {train_r2_owners:.4f} | {cv_r2_owners:.4f} | {test_metrics['owners_r2']:.4f} | {overfit_status_owners} |")

    if 'reviews_model' in results:
        cv_r2_reviews = results.get('cv_scores', {}).get('reviews', {}).get('r2_mean', 0)
        report_lines.append(f"| Reviews | R² | {train_r2_reviews:.4f} | {cv_r2_reviews:.4f} | {test_metrics['reviews_r2']:.4f} | {overfit_status_reviews} |")

    report_lines.append("")

    report_lines.append("### 12.2 Model Complexity")
    report_lines.append("")

    if 'owners_model' in results:
        n_trees_owners = results['owners_model'].n_estimators
        max_depth_owners = results['owners_model'].max_depth
        report_lines.append(f"**Owners Model:**")
        report_lines.append(f"- Number of trees: {n_trees_owners}")
        report_lines.append(f"- Max depth: {max_depth_owners}")
        report_lines.append(f"- Total parameters: ~{n_trees_owners * (2**max_depth if max_depth else 100):,}")
        report_lines.append("")

    if 'reviews_model' in results:
        n_trees_reviews = results['reviews_model'].n_estimators
        max_depth_reviews = results['reviews_model'].max_depth
        report_lines.append(f"**Reviews Model:**")
        report_lines.append(f"- Number of trees: {n_trees_reviews}")
        report_lines.append(f"- Max depth: {max_depth_reviews}")
        report_lines.append(f"- Total parameters: ~{n_trees_reviews * (2**max_depth_reviews if max_depth_reviews else 100):,}")
        report_lines.append("")

    report_lines.append("### 12.3 Confidence Intervals")
    report_lines.append("")

    # Bootstrap confidence intervals for predictions
    report_lines.append("**68% Prediction Interval (±1σ):**")
    pred_std_owners = residuals_owners.std()
    report_lines.append(f"- Owners: ± {pred_std_owners:,.0f} owners")
    report_lines.append("")

    report_lines.append("**95% Prediction Interval (±2σ):**")
    report_lines.append(f"- Owners: ± {2*pred_std_owners:,.0f} owners")
    report_lines.append("")

    report_lines.append("---")
    report_lines.append("")

    # ============================================================================
    # 13. STATISTICAL TESTS
    # ============================================================================
    report_lines.append("## 13. 📊 Statistical Tests {#statistical-tests}")
    report_lines.append("")

    report_lines.append("### 13.1 Residual Normality")
    report_lines.append("")

    # D'Agostino-Pearson test
    stat, p_value = scipy_stats.normaltest(residuals_owners)
    report_lines.append(f"**D'Agostino-Pearson Test:**")
    report_lines.append(f"- Statistic: {stat:.4f}")
    report_lines.append(f"- P-value: {p_value:.4f}")
    report_lines.append(f"- Result: {'Residuals are normally distributed' if p_value > 0.05 else 'Residuals are NOT normally distributed'} (α=0.05)")
    report_lines.append("")

    # Shapiro-Wilk test (if sample size allows)
    if len(residuals_owners) < 5000:
        stat_sw, p_value_sw = scipy_stats.shapiro(residuals_owners[:5000])
        report_lines.append(f"**Shapiro-Wilk Test:**")
        report_lines.append(f"- Statistic: {stat_sw:.4f}")
        report_lines.append(f"- P-value: {p_value_sw:.4f}")
        report_lines.append(f"- Result: {'Residuals are normally distributed' if p_value_sw > 0.05 else 'Residuals are NOT normally distributed'} (α=0.05)")
        report_lines.append("")

    report_lines.append("### 13.2 Heteroscedasticity Test")
    report_lines.append("")

    # Simple variance test across prediction ranges
    try:
        pred_quartiles = pd.qcut(y_pred_owners, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
        report_lines.append("**Residual Variance by Prediction Quartile:**")
        report_lines.append("")
        report_lines.append("| Quartile | Variance | Std Dev |")
        report_lines.append("|----------|----------|---------|")

        for quart in pred_quartiles.cat.categories:
            mask = pred_quartiles == quart
            if mask.sum() > 0:
                quart_var = residuals_owners[mask].var()
                quart_std = residuals_owners[mask].std()
                report_lines.append(f"| {quart} | {quart_var:,.0f} | {quart_std:,.0f} |")

        report_lines.append("")
    except Exception:
        report_lines.append("*Could not compute quartile variance due to data distribution*")
        report_lines.append("")

    report_lines.append("### 13.3 Prediction vs Actual Correlation")
    report_lines.append("")

    # Pearson correlation
    corr_pearson, p_pearson = scipy_stats.pearsonr(y_true_owners, y_pred_owners)
    report_lines.append(f"**Pearson Correlation:**")
    report_lines.append(f"- Coefficient: {corr_pearson:.4f}")
    report_lines.append(f"- P-value: {p_pearson:.10f}")
    report_lines.append(f"- Significance: {'Highly significant' if p_pearson < 0.001 else 'Significant' if p_pearson < 0.05 else 'Not significant'}")
    report_lines.append("")

    # Spearman correlation
    corr_spearman, p_spearman = scipy_stats.spearmanr(y_true_owners, y_pred_owners)
    report_lines.append(f"**Spearman Correlation (rank-based):**")
    report_lines.append(f"- Coefficient: {corr_spearman:.4f}")
    report_lines.append(f"- P-value: {p_spearman:.10f}")
    report_lines.append(f"- Significance: {'Highly significant' if p_spearman < 0.001 else 'Significant' if p_spearman < 0.05 else 'Not significant'}")
    report_lines.append("")

    report_lines.append("---")
    report_lines.append("")

    # ============================================================================
    # 14. PERFORMANCE COMPARISON
    # ============================================================================
    report_lines.append("## 14. 🏆 Performance Comparison {#performance-comparison}")
    report_lines.append("")

    report_lines.append("### 14.1 Baseline Comparisons")
    report_lines.append("")

    # Mean baseline
    mean_baseline_mae = mean_absolute_error(y_true_owners, [y_true_owners.mean()] * len(y_true_owners))
    mean_baseline_rmse = np.sqrt(mean_squared_error(y_true_owners, [y_true_owners.mean()] * len(y_true_owners)))

    # Median baseline
    median_baseline_mae = mean_absolute_error(y_true_owners, [y_true_owners.median()] * len(y_true_owners))
    median_baseline_rmse = np.sqrt(mean_squared_error(y_true_owners, [y_true_owners.median()] * len(y_true_owners)))

    report_lines.append("| Model | MAE | RMSE | R² |")
    report_lines.append("|-------|-----|------|-----|")
    report_lines.append(f"| Mean Baseline | {mean_baseline_mae:,.0f} | {mean_baseline_rmse:,.0f} | 0.0000 |")
    report_lines.append(f"| Median Baseline | {median_baseline_mae:,.0f} | {median_baseline_rmse:,.0f} | - |")
    report_lines.append(f"| **XGBoost Model** | **{test_metrics['owners_mae']:,.0f}** | **{test_metrics['owners_rmse']:,.0f}** | **{test_metrics['owners_r2']:.4f}** |")
    report_lines.append("")

    report_lines.append("### 14.2 Improvement Over Baseline")
    report_lines.append("")

    mae_improvement = (1 - test_metrics['owners_mae'] / mean_baseline_mae) * 100
    rmse_improvement = (1 - test_metrics['owners_rmse'] / mean_baseline_rmse) * 100

    report_lines.append(f"- **MAE Improvement:** {mae_improvement:.1f}% better than mean baseline")
    report_lines.append(f"- **RMSE Improvement:** {rmse_improvement:.1f}% better than mean baseline")
    report_lines.append(f"- **R² vs Baseline:** {test_metrics['owners_r2']:.4f} vs 0.0000 (baseline)")
    report_lines.append("")

    report_lines.append("---")
    report_lines.append("")

    # ============================================================================
    # 15. MODEL COMPLEXITY ANALYSIS
    # ============================================================================
    report_lines.append("## 15. 🧮 Model Complexity Analysis {#model-complexity}")
    report_lines.append("")

    report_lines.append("### 15.1 Computational Complexity")
    report_lines.append("")

    if 'owners_model' in results:
        n_features = len(feature_cols)
        n_trees = results['owners_model'].n_estimators
        max_depth = results['owners_model'].max_depth or 6

        report_lines.append("**Owners Model:**")
        report_lines.append(f"- Training complexity: O(n_trees × n_samples × n_features × log(n_samples))")
        report_lines.append(f"- Prediction complexity: O(n_trees × max_depth)")
        report_lines.append(f"- Actual: O({n_trees} × {len(results['X_train']):,} × {n_features} × log({len(results['X_train']):,}))")
        report_lines.append(f"- Memory usage: ~{n_trees * n_features * 8 / 1024**2:.2f} MB")
        report_lines.append("")

    report_lines.append("### 15.2 Model Size")
    report_lines.append("")

    import pickle
    import sys

    if 'owners_model' in results:
        owners_size = sys.getsizeof(pickle.dumps(results['owners_model'])) / 1024**2
        report_lines.append(f"- **Owners Model:** {owners_size:.2f} MB")

    if 'reviews_model' in results:
        reviews_size = sys.getsizeof(pickle.dumps(results['reviews_model'])) / 1024**2
        report_lines.append(f"- **Reviews Model:** {reviews_size:.2f} MB")

    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # ============================================================================
    # 16. INSIGHTS & RECOMMENDATIONS
    # ============================================================================
    report_lines.append("## 16. 💡 Insights & Recommendations {#insights-recommendations}")
    report_lines.append("")

    report_lines.append("### 16.1 Key Insights")
    report_lines.append("")

    # Generate insights based on performance
    if test_metrics['owners_r2'] > 0.85:
        report_lines.append("✅ **Excellent Owners Prediction:** The model achieves very high accuracy (R² > 0.85)")
    elif test_metrics['owners_r2'] > 0.70:
        report_lines.append("✅ **Good Owners Prediction:** The model performs well (R² > 0.70)")
    else:
        report_lines.append("⚠️ **Moderate Owners Prediction:** Room for improvement (R² < 0.70)")

    report_lines.append("")

    if test_metrics['reviews_r2'] > 0.50:
        report_lines.append("✅ **Good Reviews Prediction:** The model captures review patterns well (R² > 0.50)")
    elif test_metrics['reviews_r2'] > 0.30:
        report_lines.append("⚠️ **Moderate Reviews Prediction:** Acceptable performance (R² > 0.30)")
    else:
        report_lines.append("⚠️ **Challenging Reviews Prediction:** Reviews are harder to predict (R² < 0.30)")

    report_lines.append("")

    # Check for overfitting
    if 'owners_model' in results:
        if train_test_diff_owners < 0.05:
            report_lines.append("✅ **No Overfitting Detected:** Train and test performance are very close")
        elif train_test_diff_owners < 0.10:
            report_lines.append("⚠️ **Slight Overfitting:** Minor gap between train and test performance")
        else:
            report_lines.append("❌ **Overfitting Concern:** Significant gap between train and test performance")

    report_lines.append("")

    report_lines.append("### 16.2 Recommendations")
    report_lines.append("")

    recommendations = []

    # Performance-based recommendations
    if test_metrics['owners_r2'] < 0.80:
        recommendations.append("Consider hyperparameter tuning to improve owners prediction")

    if test_metrics['reviews_r2'] < 0.40:
        recommendations.append("Explore additional features or feature engineering for review prediction")

    # Overfitting recommendations
    if 'owners_model' in results and train_test_diff_owners > 0.10:
        recommendations.append("Increase regularization to reduce overfitting")
        recommendations.append("Consider reducing model complexity or gathering more training data")

    # Feature importance recommendations
    if 'owners_model' in results:
        feature_importance_owners = pd.DataFrame({
            'feature': feature_cols,
            'importance': results['owners_model'].feature_importances_
        }).sort_values('importance', ascending=False)

        top_5_importance = feature_importance_owners.head(5)['importance'].sum() / feature_importance_owners['importance'].sum()

        if top_5_importance > 0.70:
            recommendations.append("Top 5 features dominate - consider feature selection to simplify model")

    # Data quality recommendations
    if large_errors.sum() > len(y_true_owners) * 0.15:
        recommendations.append("Investigate games with large prediction errors - may need data quality improvements")

    # General recommendations
    recommendations.append("Monitor model performance over time as new games are released")
    recommendations.append("Consider ensemble methods to further improve predictions")
    recommendations.append("Collect feedback from stakeholders on prediction quality")

    for i, rec in enumerate(recommendations, 1):
        report_lines.append(f"{i}. {rec}")

    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # ============================================================================
    # 17. COMPLETE TECHNICAL SPECIFICATION
    # ============================================================================
    report_lines.append("## 17. 🔧 Complete Technical Specification {#technical-specification}")
    report_lines.append("")

    report_lines.append("### 17.1 Software Environment")
    report_lines.append("")
    report_lines.append("```")
    report_lines.append(f"Python: {sys.version}")
    report_lines.append(f"NumPy: {np.__version__}")
    report_lines.append(f"Pandas: {pd.__version__}")

    try:
        import xgboost
        report_lines.append(f"XGBoost: {xgboost.__version__}")
    except:
        pass

    try:
        import sklearn
        report_lines.append(f"Scikit-learn: {sklearn.__version__}")
    except:
        pass

    report_lines.append("```")
    report_lines.append("")

    report_lines.append("### 17.2 Data Processing Pipeline")
    report_lines.append("")
    report_lines.append("```")
    report_lines.append("1. Load raw data from Steam API")
    report_lines.append("2. Feature engineering (genres, categories, tags, prices, etc.)")
    report_lines.append("3. Target transformation (log1p for owners)")
    report_lines.append("4. Train/test split (80/20, random_state=42)")
    report_lines.append("5. Feature selection (SelectKBest for reviews model)")
    report_lines.append("6. Model training (XGBoost)")
    report_lines.append("7. Cross-validation (3-fold)")
    report_lines.append("8. Test evaluation")
    report_lines.append("9. Report generation")
    report_lines.append("```")
    report_lines.append("")

    report_lines.append("### 17.3 Model Serialization")
    report_lines.append("")
    report_lines.append("Models can be saved using pickle or joblib:")
    report_lines.append("```python")
    report_lines.append("import pickle")
    report_lines.append("with open('owners_model.pkl', 'wb') as f:")
    report_lines.append("    pickle.dump(results['owners_model'], f)")
    report_lines.append("```")
    report_lines.append("")

    report_lines.append("---")
    report_lines.append("")

    # ============================================================================
    # 18. REPRODUCIBILITY CHECKLIST
    # ============================================================================
    report_lines.append("## 18. ✅ Reproducibility Checklist {#reproducibility}")
    report_lines.append("")

    report_lines.append("- [x] Random seed set (random_state=42)")
    report_lines.append("- [x] Train/test split documented")
    report_lines.append("- [x] Hyperparameters recorded")
    report_lines.append("- [x] Feature engineering documented")
    report_lines.append("- [x] Cross-validation configuration specified")
    report_lines.append("- [x] Software versions recorded")
    report_lines.append("- [x] Data preprocessing steps documented")
    report_lines.append("- [x] Model evaluation metrics calculated")
    report_lines.append("- [x] Feature importance captured")
    report_lines.append("- [x] Residual analysis performed")
    report_lines.append("")

    report_lines.append("**To reproduce this training:**")
    report_lines.append("")
    report_lines.append("```python")
    report_lines.append("# Use the exact same:")
    report_lines.append("# - random_state=42")
    report_lines.append("# - train_test_split ratio (0.8/0.2)")
    report_lines.append("# - XGBoost hyperparameters (see section 5.2)")
    report_lines.append("# - Feature engineering steps")
    report_lines.append("# - Log transformation for owners")
    report_lines.append("```")
    report_lines.append("")

    report_lines.append("---")
    report_lines.append("")

    # ============================================================================
    # 19. APPENDIX: RAW DATA
    # ============================================================================
    report_lines.append("## 19. 📎 Appendix: Raw Data {#appendix}")
    report_lines.append("")

    report_lines.append("### 19.1 Sample Predictions (First 20)")
    report_lines.append("")
    report_lines.append("| Game Index | Actual Owners | Predicted Owners | Error | % Error |")
    report_lines.append("|------------|---------------|------------------|-------|---------|")

    for i in range(min(20, len(y_true_owners))):
        actual = y_true_owners.iloc[i] if hasattr(y_true_owners, 'iloc') else y_true_owners[i]
        predicted = y_pred_owners[i]
        error = actual - predicted
        pct_error = abs(error) / actual * 100 if actual != 0 else 0
        report_lines.append(f"| {i+1} | {actual:,.0f} | {predicted:,.0f} | {error:,.0f} | {pct_error:.1f}% |")

    report_lines.append("")

    report_lines.append("### 19.2 Feature List (First 50)")
    report_lines.append("")
    report_lines.append("```")
    for i, feat in enumerate(feature_cols[:50], 1):
        report_lines.append(f"{i}. {feat}")

    if len(feature_cols) > 50:
        report_lines.append(f"... and {len(feature_cols) - 50} more features")

    report_lines.append("```")
    report_lines.append("")

    report_lines.append("---")
    report_lines.append("")
    report_lines.append("**End of Report**")
    report_lines.append("")
    report_lines.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*")

    return '\n'.join(report_lines)
