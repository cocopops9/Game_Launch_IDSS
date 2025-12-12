"""
Generate Engineered Features CSV
================================
This script generates a CSV file containing all engineered features
used in the Game Launch IDSS model.

Output: data/engineered_features.csv
Columns: appid, name, [all engineered features]

Usage:
    python generate_features_csv.py
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


def parse_owners_range(owners_str):
    """Parse SteamSpy owners range to numeric value (midpoint)"""
    if pd.isna(owners_str):
        return 10000

    try:
        owners_str = str(owners_str).strip()

        if '..' in owners_str:
            parts = owners_str.split('..')
            lower = int(parts[0].replace(',', '').replace(' ', '').strip())
            upper = int(parts[1].replace(',', '').replace(' ', '').strip())
            return (lower + upper) / 2
        elif ' - ' in owners_str:
            parts = owners_str.split(' - ')
            lower = int(parts[0].replace(',', '').strip())
            upper = int(parts[1].replace(',', '').strip())
            return (lower + upper) / 2
        elif '-' in owners_str and not owners_str.startswith('-'):
            parts = owners_str.split('-')
            if len(parts) == 2:
                lower = int(parts[0].replace(',', '').strip())
                upper = int(parts[1].replace(',', '').strip())
                return (lower + upper) / 2

        return int(str(owners_str).replace(',', ''))
    except Exception as e:
        return 10000


def generate_engineered_features(df):
    """
    Generate all engineered features from the raw Steam data.
    Returns DataFrame with appid, name, and all engineered features.
    """
    print("Starting feature engineering...")

    feature_cols = []

    # Keep track of original appid and name
    result_df = pd.DataFrame()
    result_df['appid'] = df['appid'].copy()
    result_df['name'] = df['name'].copy()

    # ========================================================================
    # 1. PARSE OWNERS TO NUMERIC
    # ========================================================================
    print("  [1/12] Parsing owners data...")
    df['owners_numeric'] = df['owners'].apply(parse_owners_range)
    result_df['owners_numeric'] = df['owners_numeric']
    feature_cols.append('owners_numeric')

    # ========================================================================
    # 2. PRICE FEATURES
    # ========================================================================
    print("  [2/12] Creating price features...")
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
        result_df['price'] = df['price']
        result_df['price_log'] = np.log1p(df['price'])
        result_df['price_squared'] = df['price'] ** 2
        result_df['is_free'] = (df['price'] == 0).astype(int)
        result_df['price_tier'] = pd.cut(df['price'],
                                         bins=[-0.01, 0, 5, 10, 20, 40, 60, float('inf')],
                                         labels=[0, 1, 2, 3, 4, 5, 6]).astype(int)
        feature_cols.extend(['price', 'price_log', 'price_squared', 'is_free', 'price_tier'])

    # ========================================================================
    # 3. TIME/AGE FEATURES
    # ========================================================================
    print("  [3/12] Creating time/age features...")
    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        reference_date = pd.Timestamp('2017-05-01')

        result_df['game_age_days'] = (reference_date - df['release_date']).dt.days
        result_df['game_age_days'] = result_df['game_age_days'].fillna(365).clip(0, 10000)
        result_df['game_age_years'] = result_df['game_age_days'] / 365
        result_df['game_age_log'] = np.log1p(result_df['game_age_days'])
        result_df['release_year'] = df['release_date'].dt.year.fillna(2015).astype(int)
        result_df['release_month'] = df['release_date'].dt.month.fillna(6).astype(int)
        result_df['release_quarter'] = df['release_date'].dt.quarter.fillna(2).astype(int)
        result_df['release_day_of_week'] = df['release_date'].dt.dayofweek.fillna(3).astype(int)
        result_df['is_holiday_release'] = df['release_date'].dt.month.isin([11, 12, 6, 7]).fillna(False).astype(int)
        result_df['is_weekend_release'] = df['release_date'].dt.dayofweek.isin([4, 5]).fillna(False).astype(int)

        feature_cols.extend(['game_age_days', 'game_age_years', 'game_age_log',
                            'release_year', 'release_month', 'release_quarter',
                            'release_day_of_week', 'is_holiday_release', 'is_weekend_release'])

    # ========================================================================
    # 4. REQUIRED AGE FEATURES
    # ========================================================================
    print("  [4/12] Creating required age features...")
    if 'required_age' in df.columns:
        df['required_age'] = pd.to_numeric(df['required_age'], errors='coerce').fillna(0)
        result_df['required_age'] = df['required_age']
        result_df['is_mature'] = (df['required_age'] >= 18).astype(int)
        result_df['is_teen'] = ((df['required_age'] >= 13) & (df['required_age'] < 18)).astype(int)
        feature_cols.extend(['required_age', 'is_mature', 'is_teen'])

    # ========================================================================
    # 5. ACHIEVEMENT FEATURES
    # ========================================================================
    print("  [5/12] Creating achievement features...")
    if 'achievements' in df.columns:
        df['achievements'] = pd.to_numeric(df['achievements'], errors='coerce').fillna(0)
        result_df['achievements'] = df['achievements']
        result_df['has_achievements'] = (df['achievements'] > 0).astype(int)
        result_df['achievements_log'] = np.log1p(df['achievements'])
        result_df['achievements_tier'] = pd.cut(df['achievements'],
                                               bins=[-0.01, 0, 1, 10, 50, 100, float('inf')],
                                               labels=[0, 1, 2, 3, 4, 5]).astype(int)
        feature_cols.extend(['achievements', 'has_achievements', 'achievements_log', 'achievements_tier'])

    # ========================================================================
    # 6. PLATFORM FEATURES
    # ========================================================================
    print("  [6/12] Creating platform features...")
    if 'platforms' in df.columns:
        result_df['windows'] = df['platforms'].str.contains('windows', case=False, na=False).astype(int)
        result_df['mac'] = df['platforms'].str.contains('mac', case=False, na=False).astype(int)
        result_df['linux'] = df['platforms'].str.contains('linux', case=False, na=False).astype(int)
        result_df['platform_count'] = result_df['windows'] + result_df['mac'] + result_df['linux']
        result_df['is_cross_platform'] = (result_df['platform_count'] >= 2).astype(int)
        result_df['is_all_platforms'] = (result_df['platform_count'] == 3).astype(int)
        result_df['is_windows_only'] = ((result_df['windows'] == 1) & (result_df['platform_count'] == 1)).astype(int)
        feature_cols.extend(['windows', 'mac', 'linux', 'platform_count',
                            'is_cross_platform', 'is_all_platforms', 'is_windows_only'])

    # Language feature
    if 'english' in df.columns:
        result_df['english'] = pd.to_numeric(df['english'], errors='coerce').fillna(1).astype(int)
        feature_cols.append('english')

    # ========================================================================
    # 7. CATEGORY FEATURES
    # ========================================================================
    print("  [7/12] Creating category features...")
    if 'categories' in df.columns:
        result_df['num_categories'] = df['categories'].fillna('').str.split(';').str.len()
        feature_cols.append('num_categories')

        important_categories = [
            'Single-player', 'Multi-player', 'Online Multi-Player',
            'Steam Achievements', 'Steam Trading Cards', 'Steam Cloud',
            'Full controller support', 'Partial Controller Support',
            'Steam Leaderboards', 'Co-op', 'Online Co-op',
            'Shared/Split Screen', 'VR Support', 'Steam Workshop',
            'In-App Purchases', 'Includes level editor', 'Commentary available',
            'Local Multi-Player', 'Cross-Platform Multiplayer', 'MMO'
        ]

        for category in important_categories:
            col_name = f'cat_{category.lower().replace(" ", "_").replace("-", "_").replace("/", "_")}'
            result_df[col_name] = df['categories'].str.contains(category, case=False, na=False).astype(int)
            feature_cols.append(col_name)

    # ========================================================================
    # 8. GENRE FEATURES
    # ========================================================================
    print("  [8/12] Creating genre features...")
    if 'genres' in df.columns:
        result_df['num_genres'] = df['genres'].fillna('').str.split(';').str.len()
        feature_cols.append('num_genres')

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
            result_df[col_name] = df['genres'].str.contains(genre, case=False, na=False).astype(int)
            feature_cols.append(col_name)

    # ========================================================================
    # 9. TAG FEATURES (Top tags from steamspy_tags)
    # ========================================================================
    print("  [9/12] Creating tag features...")
    if 'steamspy_tags' in df.columns:
        result_df['num_tags'] = df['steamspy_tags'].fillna('').str.split(';').str.len()
        feature_cols.append('num_tags')

        # Top 60 most common tags
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

        for tag in top_tags:
            col_name = f'tag_{tag.lower().replace(" ", "_").replace("&", "and").replace("-", "_")}'
            result_df[col_name] = df['steamspy_tags'].str.contains(tag, case=False, na=False).astype(int)
            feature_cols.append(col_name)

    # ========================================================================
    # 10. DEVELOPER/PUBLISHER FEATURES
    # ========================================================================
    print("  [10/12] Creating developer/publisher features...")
    if 'developer' in df.columns:
        dev_counts = df['developer'].value_counts()
        result_df['developer_game_count'] = df['developer'].map(dev_counts).fillna(1)
        result_df['is_prolific_developer'] = (result_df['developer_game_count'] >= 5).astype(int)
        result_df['is_new_developer'] = (result_df['developer_game_count'] == 1).astype(int)
        feature_cols.extend(['developer_game_count', 'is_prolific_developer', 'is_new_developer'])

        # Top 20 developers as binary features
        top_developers = dev_counts.head(20).index.tolist()
        for i, dev in enumerate(top_developers):
            if pd.notna(dev):
                col_name = f'dev_top_{i+1}'
                result_df[col_name] = (df['developer'] == dev).astype(int)
                feature_cols.append(col_name)

    if 'publisher' in df.columns:
        pub_counts = df['publisher'].value_counts()
        result_df['publisher_game_count'] = df['publisher'].map(pub_counts).fillna(1)
        result_df['is_major_publisher'] = (result_df['publisher_game_count'] >= 10).astype(int)
        feature_cols.extend(['publisher_game_count', 'is_major_publisher'])

        if 'developer' in df.columns:
            result_df['is_self_published'] = (df['developer'] == df['publisher']).astype(int)
            feature_cols.append('is_self_published')

    # ========================================================================
    # 11. MARKET SATURATION FEATURES
    # ========================================================================
    print("  [11/12] Creating market saturation features...")
    if 'release_date' in df.columns:
        df_temp = df.copy()
        df_temp['release_date'] = pd.to_datetime(df_temp['release_date'], errors='coerce')
        df_temp['year_month'] = df_temp['release_date'].dt.to_period('M')
        df_temp['year_quarter'] = df_temp['release_date'].dt.to_period('Q')

        games_per_month = df_temp.groupby('year_month').size()
        games_per_quarter = df_temp.groupby('year_quarter').size()

        result_df['market_saturation_month'] = df_temp['year_month'].map(games_per_month).fillna(100)
        result_df['market_saturation_quarter'] = df_temp['year_quarter'].map(games_per_quarter).fillna(300)
        result_df['market_saturation_log'] = np.log1p(result_df['market_saturation_month'])
        feature_cols.extend(['market_saturation_month', 'market_saturation_quarter', 'market_saturation_log'])

    # ========================================================================
    # 12. INTERACTION FEATURES
    # ========================================================================
    print("  [12/12] Creating interaction features...")

    # Indie × Multiplatform
    if 'genre_indie' in result_df.columns and 'is_cross_platform' in result_df.columns:
        result_df['indie_multiplatform'] = result_df['genre_indie'] * result_df['is_cross_platform']
        feature_cols.append('indie_multiplatform')

    # Price × Action genre
    if 'price' in result_df.columns and 'genre_action' in result_df.columns:
        result_df['price_x_action'] = result_df['price'] * result_df['genre_action']
        feature_cols.append('price_x_action')

    # Platform count × Price
    if 'platform_count' in result_df.columns and 'price' in result_df.columns:
        result_df['platforms_x_price'] = result_df['platform_count'] * result_df['price']
        feature_cols.append('platforms_x_price')

    # F2P × Multiplayer
    if 'is_free' in result_df.columns and 'cat_multi_player' in result_df.columns:
        result_df['f2p_multiplayer'] = result_df['is_free'] * result_df['cat_multi_player']
        feature_cols.append('f2p_multiplayer')

    # Achievements × Premium price
    if 'has_achievements' in result_df.columns and 'price' in result_df.columns:
        result_df['achievements_x_price'] = result_df['has_achievements'] * result_df['price']
        feature_cols.append('achievements_x_price')

    # Holiday release × Casual genre
    if 'is_holiday_release' in result_df.columns and 'genre_casual' in result_df.columns:
        result_df['holiday_casual'] = result_df['is_holiday_release'] * result_df['genre_casual']
        feature_cols.append('holiday_casual')

    # Clean up - ensure all features are numeric
    print("\nCleaning up features...")
    for col in feature_cols:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0)

    # Remove duplicates from feature_cols
    feature_cols = list(dict.fromkeys(feature_cols))

    print(f"\nTotal engineered features: {len(feature_cols)}")

    return result_df, feature_cols


def main():
    """Main function to generate the engineered features CSV"""

    print("=" * 60)
    print("GENERATING ENGINEERED FEATURES CSV")
    print("=" * 60)

    # Get the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')

    # Input and output paths - steam.csv can be in root or data directory
    input_path = os.path.join(script_dir, 'steam.csv')
    if not os.path.exists(input_path):
        input_path = os.path.join(data_dir, 'steam.csv')

    output_path = os.path.join(data_dir, 'engineered_features.csv')

    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found.")
        print("Please ensure steam.csv is in the project root or data folder.")
        return None

    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Load raw data
    print(f"\nLoading data from: {input_path}")
    df = pd.read_csv(input_path, quotechar='"', escapechar='\\', on_bad_lines='warn')
    print(f"Loaded {len(df)} games")

    # Display original columns
    print(f"\nOriginal columns ({len(df.columns)}): {list(df.columns)}")

    # Generate engineered features
    result_df, feature_cols = generate_engineered_features(df)

    # Reorder columns: appid, name, then all features
    final_columns = ['appid', 'name'] + feature_cols
    result_df = result_df[final_columns]

    # Save to CSV
    print(f"\nSaving engineered features to: {output_path}")
    result_df.to_csv(output_path, index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total games: {len(result_df)}")
    print(f"Total columns: {len(result_df.columns)}")
    print(f"  - Identifier columns: 2 (appid, name)")
    print(f"  - Engineered features: {len(feature_cols)}")
    print(f"\nOutput file: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    # Show first few columns as preview
    print("\nFirst 10 feature columns:")
    for i, col in enumerate(feature_cols[:10], 1):
        print(f"  {i}. {col}")
    print(f"  ... and {len(feature_cols) - 10} more features")

    # Show sample data
    print("\nSample data (first 3 rows, first 8 columns):")
    print(result_df.iloc[:3, :8].to_string())

    return result_df


if __name__ == "__main__":
    result = main()
