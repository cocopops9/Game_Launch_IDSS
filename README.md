# 🎮 Game Launch IDSS - Updated with Intelligent Ranking System

## ⚠️ Honest Assessment

### Two Spearman Correlations (Both Are Real)

| Validation Method | Spearman | Purpose | Meaning |
|-------------------|----------|---------|---------|
| **Random split** | **0.7241** | Comparing "what-if" configs | Can rank games in same dataset |
| **Temporal split** | **0.4183** | Predicting NEW games | Realistic for future prediction |

### Why Two Numbers?

1. **Random split (0.72)**: Train on 80% random games, test on 20% random games
   - High because model "sees" similar games during training
   - Valid for: "If I add multiplayer, will my game rank higher?"
   
2. **Temporal split (0.42)**: Train on older games, test on newer games
   - Lower because predicting truly unseen future games is harder
   - This is the REALISTIC metric for new game success prediction

### ⚠️ The 0.60 Threshold Is NOT Met for Realistic Prediction

The temporal Spearman (0.42) does NOT meet the 0.60 reliability threshold. Use recommendations as **directional guidance**, not guarantees.

## How to Run

```bash
# 1. Install dependencies
pip install streamlit pandas numpy scikit-learn xgboost scipy

# 2. Place steam.csv in the folder

# 3. Run
cd game_idss_updated
streamlit run app.py
```

## How the Ranking Model Works

### 1. Feature Engineering (73 features)
- Price: price, price_log, is_free, price_tier
- Platforms: windows, mac, linux, platform_count
- Achievements: count, has_achievements, achievements_log
- Time: game_age, release_year, release_month
- Genres: 12 binary features
- Categories: 14 binary features (Steam features)
- Tags: 26 binary features

### 2. Ensemble Model (4 models)
```
XGBoost (rank target)       → Spearman ~0.72
XGBoost (log-owners)        → Spearman ~0.71
GradientBoosting (rank)     → Spearman ~0.71
RandomForest (log-owners)   → Spearman ~0.70
```
Weighted average based on individual performance.

### 3. Why Random Split Shows Higher Correlation

The 0.72 comes from random split because:
- Test games are similar to training games (same time period, similar market)
- Model learns patterns that apply to the same distribution

The 0.42 temporal split is more realistic because:
- Future games may have different patterns
- Market conditions change over time
- New genres and features emerge

## Example Output

```
Market Position: 51.2th percentile
Tier: average

RANKING QUALITY:
  Random Spearman:   0.7241 (config comparison)
  Temporal Spearman: 0.4183 (realistic prediction)
  Reliable (>=0.60): False

IMPROVEMENTS:
  Target multiplayer audience: +2.5pp → 54th (High confidence)
  Add Steam Trading Cards: +1.5pp → 53th (High confidence)
  Add Steam Workshop: +1.2pp → 52th (High confidence)
```

## Confidence Levels

Based on historical data analysis:

| Feature | Historical Lift | Confidence |
|---------|-----------------|------------|
| Steam Trading Cards | +250% median | High ✅ |
| Multiplayer tag | +650% median | High ✅ |
| Steam Workshop | +180% median | High ✅ |
| Steam Cloud | +50% median | Medium |
| Multi-platform | Variable | Low |

## Files

| File | Purpose |
|------|---------|
| `ranking_integration.py` | Core ranking model with honest metrics |
| `intelligence_engine.py` | Business intelligence using the ranker |
| `app.py` | Streamlit main application |
| `new_game.py` | Game analysis interface |

## Bottom Line

- **Use the 0.72 Spearman** for comparing different configurations of YOUR game
- **Use the 0.42 Spearman** as the realistic expectation for new game prediction
- **Treat recommendations as directional guidance**, not guarantees
- **High confidence** improvements (Trading Cards, Multiplayer, Workshop) are validated by historical data
