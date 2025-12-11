# 🎮 Game Launch IDSS - Intelligent Decision Support System

A comprehensive Streamlit application that provides data-driven insights for game launch decisions using machine learning models trained on 27,075 Steam games.

## 🎯 Overview

The Game Launch IDSS helps game developers make informed decisions about their game's features, pricing, and positioning by analyzing historical Steam market data. The system provides:

- **Market Positioning**: Percentile-based ranking of your game configuration (XGBoost Rank)
- **Improvement Scenarios**: Data-driven recommendations for feature additions
- **Success Factors**: Key strengths and readiness assessment
- **Risk Assessment**: Potential challenges and market conditions
- **Owners Prediction**: Expected owner count for model performance analysis (XGBoost)
- **Review Prediction**: Expected review ratio based on game features (XGBoost)

## 📊 Dataset

- **27,075 Steam games** with comprehensive feature data
- **~87 engineered features** including:
  - Price and monetization (price, free-to-play, price tiers)
  - Platforms (Windows, Mac, Linux)
  - Genres (Action, RPG, Strategy, etc.)
  - Categories (Single-player, Multiplayer, Co-op, etc.)
  - Steam features (Trading Cards, Workshop, Cloud, Achievements)
  - Developer/publisher history
  - Release timing and game age
  - Popular tags and themes

## 🚀 Quick Start

### Installation

```bash
# 1. Install dependencies
pip install streamlit pandas numpy scikit-learn xgboost scipy

# 2. Place steam.csv in the project folder

# 3. Run the application
streamlit run app.py
```

### First Launch

The system will automatically train models on first run (takes 2-3 minutes). Models are saved and reused for subsequent launches.

## 🧠 Core Features

### 1. Market Positioning

The system ranks your game configuration against all 27,075 Steam games using **XGBoost Rank** machine learning model:

- **XGBoost (rank target)** - Direct ranking optimization trained on normalized ranks
- Uses ~87 pre-launch features (no post-launch data like ratings or playtime)
- Optimized specifically for accurate relative positioning

**Performance:**
- Spearman Rank Correlation: **0.72**
- This measures how accurately the system ranks games relative to each other
- A correlation > 0.60 is considered reliable for decision support

### 2. Improvement Scenarios

Get actionable recommendations for increasing your game's market position:

```
SUGGESTED IMPROVEMENTS:
  Add Steam Trading Cards: +2.5pp → 54th percentile (High confidence)
  Target multiplayer audience: +1.8pp → 53th percentile (High confidence)
  Add Steam Workshop: +1.2pp → 52th percentile (High confidence)
```

**Confidence Levels** are based on historical impact analysis:
- **High**: Validated by strong historical data (median lift > 100%)
- **Medium**: Moderate historical evidence (median lift 30-100%)
- **Low**: Weak or inconsistent historical evidence

### 3. Success Factors

Identifies your game's strengths:
- **Readiness Score**: Overall assessment (0-100)
- **Key Strengths**: Features that position your game favorably
- **Impact Level**: Magnitude of each factor's contribution

### 4. Risk Assessment

Highlights potential challenges:
- Market saturation in your genre/category
- Price positioning concerns
- Platform limitations
- Missing critical features

### 5. Owners Prediction

Predicts expected absolute owner count using XGBoost regression (log-scale):
- **R² Score**: ~0.36 (moderate prediction capability)
- **Mean Absolute Error**: Varies by owner tier
- **Use**: Model performance visualization and benchmarking

**Note:** Used for data analysis and model performance metrics. The ranking model (percentile) is more reliable for decision-making.

### 6. Review Prediction

Predicts expected positive review ratio using XGBoost regression:
- **R² Score**: 0.109 (limited prediction capability)
- **Mean Absolute Error**: 0.197 on 0-1 scale

**Note:** Review quality depends on game execution, which cannot be measured pre-launch. The model has very limited predictive power. Use as rough directional guidance only.

## 📁 Project Structure

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit application entry point |
| `models.py` | ML model training (owners prediction + review prediction) |
| `ranking_integration.py` | XGBoost Rank model for percentile positioning |
| `intelligence_engine.py` | Business intelligence analysis engine |
| `new_game.py` | Interactive game configuration analyzer |
| `data_analysis.py` | Model performance metrics and data exploration |
| `steam.csv` | Steam games dataset (27,075 games) |

## 🎮 How to Use

### Analyze a New Game Configuration

1. **Navigate to "New Game Analysis"** tab
2. **Configure your game:**
   - Set price ($0-60+)
   - Select platforms (Windows/Mac/Linux)
   - Choose genres (Action, RPG, Strategy, etc.)
   - Select categories (Single-player, Multiplayer, etc.)
   - Add Steam features (Trading Cards, Workshop, etc.)
   - Input developer/publisher stats

3. **Review the analysis:**
   - Market percentile ranking
   - Tier classification (elite/strong/average/weak/struggling)
   - Success factors and readiness score
   - Improvement recommendations
   - Risk assessment

4. **Iterate and compare:**
   - Try different feature combinations
   - See how changes affect percentile ranking
   - Identify highest-impact improvements

### Explore Market Data

1. **Navigate to "Data Analysis"** tab
2. **Review model performance:**
   - Ranking accuracy (Spearman correlation: 0.72)
   - Percentile distance metric
   - Review prediction performance

3. **Analyze market distributions:**
   - Owners by genre
   - Owners by category
   - Market share analysis
   - Success patterns

## 🔬 Technical Details

### How Ranking Works

The system uses percentile-based positioning rather than absolute owner predictions:

1. **Feature Engineering**: ~87 pre-launch features extracted from game configuration
2. **Rank Prediction**: XGBoost model trained on normalized ranks (0-1) using rankdata
3. **Percentile Conversion**: Rank scores converted to 1-99 percentile scale using reference distribution
4. **Validation**: Each improvement scenario validated against historical data

**Key Insight**: The model is trained to predict **relative rankings** (not absolute owners), which is more stable and reliable for market positioning. Uses only features available before launch (no post-launch metrics).

### Model Validation

- **Cross-validation**: 5-fold stratified cross-validation
- **Metric**: Spearman rank correlation (measures ranking accuracy)
- **Reliability threshold**: 0.60 for production use
- **Current performance**: 0.72 (reliable for decision support)

### Improvement Scenarios

Recommendations are generated by:

1. Testing 15+ feature modifications on your game configuration
2. Predicting new percentile for each modification
3. Ranking improvements by impact (percentile points gained)
4. Assigning confidence based on historical feature impact analysis

High-confidence improvements are validated by analyzing thousands of games with/without each feature.

## 📈 Example Output

```
=== MARKET POSITIONING ===
Overall Percentile: 51.2th
Tier: average
Estimated Owners: 10,000 - 20,000

=== SUCCESS FACTORS ===
Readiness Score: 68/100
✅ Multi-platform support (+Strong)
✅ Competitive pricing (+Medium)
✅ Popular genre (+Medium)

=== IMPROVEMENT SCENARIOS ===
  Add Steam Trading Cards: +2.5pp → 54th (High confidence)
  Target multiplayer audience: +1.8pp → 53th (High confidence)
  Add Steam Workshop: +1.2pp → 52th (High confidence)

=== RISK ASSESSMENT ===
⚠️ High competition in Action genre
⚠️ No achievements system
ℹ️ Single-player only market smaller than multiplayer
```

## 🎯 Use Cases

- **Pre-launch Planning**: Optimize feature set before development
- **Feature Prioritization**: Identify highest-impact additions
- **Competitive Analysis**: See how your game stacks up
- **Pricing Strategy**: Test different price points
- **Platform Decisions**: Evaluate multi-platform value
- **Market Research**: Understand success patterns in Steam market

## ⚠️ Limitations

- **Historical Data Only**: Models trained on existing games, cannot predict market shifts
- **Correlation Not Causation**: Features associated with success may not cause it
- **Execution Matters**: Great features poorly implemented will still fail
- **Marketing Impact**: Models don't account for marketing budget or virality
- **Review Quality**: Cannot predict actual game quality, only expected ratio based on features

## 🛠️ Development

### Adding New Features

1. Update feature engineering in `models.py`
2. Retrain models (delete saved model files)
3. Update input forms in `new_game.py`

### Customizing Analysis

- Modify confidence thresholds in `intelligence_engine.py`
- Adjust tier boundaries in `ranking_integration.py`
- Add new improvement scenarios in `intelligence_engine.py`

## 📊 Model Performance Summary

| Model Component | Algorithm | Metric | Value |
|----------------|-----------|--------|-------|
| **Market Positioning** | XGBoost Rank | Spearman Correlation | 0.72 |
| Market Positioning | XGBoost Rank | Percentile Precision | ±8-10 pts (75% within) |
| **Owners Prediction** | XGBoost (log) | R² Score | ~0.36 |
| Owners Prediction | XGBoost (log) | Use Case | Data analysis |
| **Review Prediction** | XGBoost | R² Score | 0.109 |
| Review Prediction | XGBoost | MAE | 0.197 |

## 📝 License

This project is for educational and research purposes. Steam data is publicly available but subject to Valve's terms of service.

## 🤝 Contributing

This is a research project. For questions or suggestions, please open an issue.

---

**Built with:** Python • Streamlit • scikit-learn • XGBoost • pandas • numpy
