# 🎮 Game Launch Decision Support System (IDSS)

An intelligent web-based system to help game producers make data-driven decisions for game launches using machine learning predictions and knowledge-based recommendations.

## 📋 Features

### 1. **New Game Prediction**
- Input game details (price, platforms, genres, release timing)
- Get ML-powered predictions for:
  - Number of expected owners (with confidence ranges)
  - Positive review ratio
- Receive data-driven recommendations based on actual Steam data analysis
- Adjust parameters and see real-time impact on predictions
- Save multiple configurations for comparison

### 2. **My Games**
- **Persistent Storage**: All games and configurations are saved to disk and survive page refreshes
- **Three Organized Views**:
  - **Games List**: Table of all your games with IDs, configuration counts, and performance stats
  - **All Configurations**: Ranked table of configurations across all games with CSV export
  - **Statistics**: Visual analytics with charts showing price vs owners and review distributions
- **Per-Game Configuration Management**: Select individual games to view all their configurations
- Compare different parameter sets and their predicted outcomes
- Track timestamp and configuration history for each game
- Sort configurations by predicted performance metrics

### 3. **Data Analysis Dashboard**
- **Correlation Analysis**: Interactive heatmaps showing relationships between features from actual Steam data
- **Feature Importance**: Understand which factors most influence success for both ownership and review predictions
- **Trend Analysis**: Visualize price-performance relationships and seasonal patterns
- **Platform Combination Analysis**: Bar charts showing game counts and performance by platform combinations (Windows Only, Mac Only, Linux Only, Windows+Mac, Windows+Linux, Mac+Linux, Windows+Mac+Linux)
- **Model Performance**: Track prediction accuracy metrics (R², MAE, RMSE)

## 🚀 Quick Start

### Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Prepare your Steam data:
   - Place `steam.csv` in the project directory (required)
   - Optionally place `steamspy_tag_data.csv` for additional tag data

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser to `http://localhost:8501`

### Dependencies (Latest Versions)
All dependencies are pinned to their latest stable versions in `requirements.txt`:
- **Streamlit 1.51.0** - Web framework
- **Pandas 2.3.3** - Data processing
- **NumPy 2.3.4** - Numerical computing
- **Plotly 6.4.0** - Interactive visualizations
- **Scikit-learn 1.7.2** - Machine learning models
- **LightGBM 4.6.0** - Gradient boosting
- **Seaborn 0.13.2** - Statistical visualizations
- **Matplotlib 3.10.7** - Plotting library

## 📊 Using Your Steam Data

The system requires actual Steam data to function properly.

### Required Files

1. **steam.csv** - Main Steam game data (required)
   - Must contain columns: `name`, `price`, `release_date`, `platforms`, `owners`, `positive_reviews`, `negative_reviews`, `genres`

2. **steamspy_tag_data.csv** - Optional tag/genre information for enhanced analysis

### Data Format

#### steam.csv Required Columns
- `name`: Game name
- `price`: Price in USD (numeric)
- `release_date`: Release date (various formats supported)
- `platforms`: Platform availability as semicolon-separated string (e.g., "windows;mac;linux")
- `owners`: Ownership range as string (e.g., "10000-20000")
- `positive_reviews`: Number of positive reviews (numeric)
- `negative_reviews`: Number of negative reviews (numeric)
- `genres`: Game genres
- `required_age`: Age rating (optional)

#### steamspy_tag_data.csv (Optional)
- `appid`: Steam app ID
- `steamspy_tags`: Tags associated with the game

## 🎯 Key Functionalities

### Prediction Models
- **Ownership Prediction**: GradientBoosting model trained on actual Steam data
- **Review Ratio Prediction**: LightGBM model for predicting positive review percentage

### Data-Driven Recommendations
The system analyzes historical Steam data to provide insights on:
- Price optimization based on market trends
- Platform support impact on player reach
- Genre/feature additions that correlate with better reviews
- Release timing seasonality patterns
- Key success factors identified through feature importance analysis

### Parameter Adjustment
- Adjust pricing from $0.99 to $59.99 (or free)
- Select platform combinations (Windows, Mac, Linux)
- Choose from available genre tags
- Set release month
- Configure age ratings

## 🔧 Customization

### Adding New Features

To add new predictive features:

1. Update the data preprocessing in `data_loader.py`:
```python
def preprocess_steam_data(self, df):
    # Add your feature processing
    df['new_feature'] = ...
```

2. Update the input form in `app.py`:
```python
# In new_game_page() function
new_feature = st.slider("New Feature", min_val, max_val)
```

### Modifying Recommendations

Edit the `generate_data_driven_recommendations()` function in `app.py` to add custom business rules based on your data analysis.

## 📈 Model Performance

The system displays:
- **R² Score**: Model fit quality (higher is better)
- **MAE**: Mean Absolute Error in predictions
- **RMSE**: Root Mean Square Error
- **Feature Importance**: Which features most influence predictions
- **Prediction vs Actual**: Calibration plots comparing model predictions to real outcomes

## 💡 Usage Tips

1. **Start with complete data**: Ensure your Steam CSV is properly formatted with all required columns
2. **Compare configurations**: Save multiple versions to find optimal settings
3. **Monitor correlations**: Use the Data Analysis tab to understand feature relationships
4. **Check seasonal patterns**: Release timing significantly affects initial sales
5. **Platform strategy**: Windows support is critical (correlates strongly with ownership)

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **ML Models**: scikit-learn (GradientBoosting), LightGBM
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Seaborn, Matplotlib
- **Deployment Ready**: Docker-compatible

## 📊 Data Flow

1. **Input**: User enters game parameters through the UI
2. **Processing**: Features are extracted and normalized
3. **Prediction**: ML models predict owners and review ratio based on trained models
4. **Analysis**: System analyzes predictions against historical Steam data patterns
5. **Recommendations**: Knowledge engine generates actionable insights from data
6. **Persistence**: Configurations automatically saved to `saved_games.json` for permanent storage
7. **Comparison**: Access and compare configurations across sessions

## 🔍 Troubleshooting

### Common Issues

1. **File not found error**: Ensure `steam.csv` is in the same directory as `app.py`
2. **Import errors**: Run `pip install -r requirements.txt` to install all dependencies
3. **Data validation warnings**: Check that CSV columns match expected format
4. **Memory issues**: For very large datasets (100k+ games), consider sampling data
5. **Games not persisting**: Check write permissions for `saved_games.json` in the project directory
6. **Version conflicts**: Use Python 3.11+ and install dependencies from requirements.txt with exact versions

## 🚀 Deployment

### Local Deployment
```bash
streamlit run app.py --server.port 8501
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

### Cloud Deployment
The app is compatible with:
- Heroku (with Procfile)
- AWS EC2/ECS
- Google Cloud Run
- Azure App Service

## 📋 System Requirements

- **Python 3.11+** (tested with 3.11.14)
- **Operating System**: Linux, macOS, or Windows
- **RAM**: Minimum 4GB (8GB+ recommended for large datasets)
- **Storage**: 500MB+ for dependencies and data

### Python Dependencies (see requirements.txt)
- streamlit==1.51.0
- pandas==2.3.3
- numpy==2.3.4
- plotly==6.4.0
- scikit-learn==1.7.2
- lightgbm==4.6.0
- seaborn==0.13.2
- matplotlib==3.10.7

## 📝 Example Usage

1. Upload `steam.csv` to the project directory
2. Launch the application
3. Go to "New Game" tab
4. Enter your game details:
   - Price: $14.99
   - Platforms: Windows, Mac
   - Genres: Indie, Singleplayer, Adventure
   - Release Month: November
5. Click "Predict" to see predictions and recommendations
6. Save the configuration for later comparison

## 🤝 Contributing

To customize or enhance:
1. Review the modular structure (separate data_loader and app)
2. Add new features to the preprocessing pipeline
3. Extend recommendation logic with custom business rules
4. Test thoroughly with your Steam dataset

## 📄 License

MIT License - Feel free to use and modify for your projects

## 🔧 Support

For issues or questions:
- Check the troubleshooting section
- Review data_loader.py for data format validation
- Examine model performance metrics in the Data Analysis tab
- Verify your steam.csv matches the required format

---

**Note**: This system is designed for decision support and should be used alongside market research and domain expertise for best results. Predictions are based on historical Steam data patterns and may vary based on marketing, community engagement, and other external factors.
