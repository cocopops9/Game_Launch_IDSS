# Game Launch IDSS - Training Report

**Generated:** 2025-11-11 18:29:29

## Dataset Overview
- **Total Games:** 27,075
- **Training Set:** 21,660 games (80%)
- **Test Set:** 5,415 games (20%)
- **Features Used:** 43

### Data Ranges
- **Owners:** 10,000 - 150,000,000 (median: 10,000)
- **Review Ratio:** 0.000 - 1.000 (mean: 0.714)

---

## Model 1: Owners Prediction (GradientBoostingRegressor)

### Purpose
Predict game ownership (user reach)

### Architecture
- **Estimators:** 150
- **Learning Rate:** 0.08
- **Max Depth:** 5
- **Regularization:** subsample=0.8, min_samples_split=15

### Performance

**Cross-Validation (5-fold):**
- Mean R²: 0.882 ± 0.006

**Training Set:**
- R²: 0.900
- MAE: 47,753 owners

**Test Set (Unseen Data):**
- R²: 0.888
- MAE: 62,074 owners
- RMSE: 1,039,135 owners
- SMAPE: 24.5%

**Interpretation:**
Test R² of 0.888 means the model explains 88.8% of variance in log(owners). SMAPE of 24.5% indicates typical prediction error.

### Model Complexity
- **Features Used:** 47
- **Total Trees:** 150
- **Max Tree Depth:** 5
- **Min Samples per Split:** 15
- **Min Samples per Leaf:** 8
- **Features Considered per Split:** sqrt

**Analysis:** This configuration creates 150 decision trees, each with max depth 5. The model considers √n features at each split, providing good balance between model diversity and computational efficiency.

### Residual Analysis

**Training Set Residuals:**
- Mean: 20,847 (should be ~0)
- Std Dev: 890,859
- Range: [-21,316,405, 119,701,962]
- Median: -106

**Test Set Residuals:**
- Mean: 33,149 (should be ~0)
- Std Dev: 1,038,702
- Range: [-4,816,900, 71,112,855]
- Median: -90

**Analysis:** Mean residuals close to 0 indicate unbiased predictions. Test residual std dev of 1,038,702 shows typical prediction error magnitude.

### Prediction Distribution

**Training Set:**
- Predictions range: 8,174 - 56,316,405
- Mean predicted: 113,354 (actual: 134,201)
- Std dev predicted: 807,485 (actual: 1,346,958)

**Test Set:**
- Predictions range: 8,324 - 19,816,900
- Mean predicted: 100,498 (actual: 133,647)
- Std dev predicted: 510,324 (actual: 1,249,886)

**Analysis:** Prediction distribution should match actual distribution. Significant differences indicate model bias or inability to capture full variance.

### Top 10 Features by Importance (Model-Based)
1. **total_ratings**: 0.3106
2. **median_playtime**: 0.2615
3. **engagement_score**: 0.1597
4. **game_age_days**: 0.0659
5. **average_playtime**: 0.0614
6. **release_year**: 0.0336
7. **price**: 0.0139
8. **tag_Free_To_Play**: 0.0136
9. **tag_Indie**: 0.0133
10. **tag_Multiplayer**: 0.0116


### Top 10 Features by Target Correlation
1. **total_ratings**: 0.736
2. **engagement_score**: 0.250
3. **average_playtime**: 0.174
4. **game_age_days**: 0.149
5. **release_year**: -0.142
6. **is_free**: 0.075
7. **required_age**: 0.074
8. **linux**: 0.051
9. **mac**: 0.042
10. **tag_Adventure**: -0.040


**Analysis:** Feature importance (model-based) shows which features the model uses most for predictions. Target correlation shows linear relationship with outcome. Both metrics are important for understanding model behavior.

---

## Model 2: Review Ratio Prediction (LightGBM)

### Purpose
Predict review quality (user satisfaction)

### Architecture
- **Estimators:** 150
- **Learning Rate:** 0.05
- **Max Depth:** 6
- **Num Leaves:** 31
- **Regularization:** L1=0.15, L2=0.15

### Performance

**Cross-Validation (5-fold):**
- Mean R²: 0.168 ± 0.006

**Training Set:**
- R²: 0.249
- MAE: 0.155

**Test Set (Unseen Data):**
- R²: 0.159
- MAE: 0.163
- RMSE: 0.214

**Interpretation:**
Test R² of 0.159 means the model explains 15.9% of variance in review ratio. MAE of 0.163 is the average prediction error.

### Model Complexity
- **Features Used:** 47
- **Total Trees:** 150
- **Max Tree Depth:** 6
- **Num Leaves per Tree:** 31
- **Min Samples per Leaf:** 15

**Analysis:** LightGBM uses 150 leaf-wise trees with up to 31 leaves each. Leaf-wise growth is more efficient than depth-wise growth for complex patterns.

### Residual Analysis

**Training Set Residuals:**
- Mean: 0.0002 (should be ~0)
- Std Dev: 0.2025
- Range: [-0.8427, 0.7464]
- Median: 0.0269

**Test Set Residuals:**
- Mean: 0.0011 (should be ~0)
- Std Dev: 0.2139
- Range: [-0.8599, 0.7283]
- Median: 0.0313

**Analysis:** Mean residuals close to 0 indicate unbiased predictions. Review ratio residuals are on 0-1 scale.

### Prediction Distribution

**Training Set:**
- Predictions range: 0.199 - 0.959
- Mean predicted: 0.714 (actual: 0.714)
- Std dev predicted: 0.092 (actual: 0.234)

**Test Set:**
- Predictions range: 0.272 - 0.958
- Mean predicted: 0.715 (actual: 0.716)
- Std dev predicted: 0.090 (actual: 0.233)

**Analysis:** Prediction distribution alignment with actual values indicates model's ability to capture the full range of outcomes.

### Top 10 Features by Importance (Model-Based)
1. **total_ratings**: 687.0000
2. **price**: 491.0000
3. **game_age_days**: 316.0000
4. **achievements**: 264.0000
5. **engagement_score**: 250.0000
6. **release_month**: 250.0000
7. **release_year**: 232.0000
8. **num_genres**: 212.0000
9. **num_categories**: 210.0000
10. **total_tags**: 136.0000


### Top 10 Features by Target Correlation
1. **has_achievements**: 0.172
2. **engagement_score**: 0.164
3. **mac**: 0.122
4. **linux**: 0.116
5. **tag_Simulation**: -0.104
6. **price**: 0.076
7. **tag_Strategy**: -0.050
8. **tag_Action**: -0.036
9. **game_age_days**: 0.027
10. **total_ratings**: 0.026


**Analysis:** Comparing model-based importance with direct correlation helps identify non-linear relationships that the model captures.

---

## Feature Impact Analysis

### Top 5 Features That Improve Owners
- **tag_Open_World**: +3400.0% improvement
- **total_ratings**: +1400.0% improvement
- **engagement_score**: +1400.0% improvement
- **tag_Survival**: +1400.0% improvement
- **tag_Fps**: +1400.0% improvement


### Top 5 Features That Improve Review Ratio
- **total_ratings**: +14.2% improvement
- **engagement_score**: +14.2% improvement
- **achievements**: +14.1% improvement
- **num_categories**: +13.4% improvement
- **tag_Puzzle**: +13.2% improvement


---

## Model Insights

### What the Models Learned
- Owners model learned to predict game reach using platform support, genres, pricing, and game features
- Review model learned to predict user satisfaction based on game quality indicators and feature combinations
- Both models use 43 features from actual Steam data to make predictions


### Why This Approach
- Log transformation for owners handles the wide 10K-200M range and makes predictions more stable
- Gradient Boosting for owners provides better handling of non-linear relationships
- LightGBM for reviews is faster and handles sparse features (tags) efficiently
- Cross-validation ensures models generalize well to unseen games
- 80/20 train/test split validates real-world performance


### Model Behavior
- Owners model: CV R² = 0.882, Test R² = 0.888 (overfitting check: minimal)
- Review model: CV R² = 0.168, Test R² = 0.159 (overfitting check: minimal)
- Models focus on IMPROVEMENT ANALYSIS rather than exact prediction accuracy
- Feature impact analysis shows what changes improve outcomes, not just correlation


### Recommendations for Improvement
- ⚠️ Review model has low R². Reviews may be inherently unpredictable from available features.


---

## Conclusion

These models analyze actual Steam data to provide:
1. **Predictive insights** - estimate potential owners and review quality
2. **Improvement analysis** - show what features improve outcomes
3. **Data-driven recommendations** - suggest optimal game configurations

The models prioritize **understanding what improves game success** over perfect prediction accuracy.
