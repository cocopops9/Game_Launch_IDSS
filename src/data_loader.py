"""
Data Loader Utility for Steam Dataset
Handles loading and preprocessing of Steam CSV data - REQUIRED FILES
"""

import pandas as pd
import csv
import numpy as np
from typing import Dict, Tuple, List
import json
from datetime import datetime
import os

class SteamDataLoader:
    """Load and preprocess Steam game data from required CSV files"""
    
    def __init__(self, steam_csv_path: str, tags_csv_path: str = None):
        self.steam_csv_path = steam_csv_path
        self.tags_csv_path = tags_csv_path
        self.data = None
        self.tag_data = None
        
        # Check if files exist
        if not os.path.exists(steam_csv_path):
            raise FileNotFoundError(f"Required file not found: {steam_csv_path}")
        
    def load_steam_data(self) -> pd.DataFrame:
        """Load main Steam dataset - REQUIRED"""
        try:
            df = pd.read_csv(
            self.steam_csv_path,
            quotechar='"',
            escapechar='\\',
            on_bad_lines='warn'
        )
            print(f"Loaded {len(df)} games from {self.steam_csv_path}")
            
            # Check for minimum required columns
            required_cols = ['price']  # At minimum we need price
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"Warning: Missing columns {missing_cols}, will use defaults")
            
            self.data = self.preprocess_steam_data(df)
            return self.data
            
        except Exception as e:
            raise Exception(f"Error loading Steam data from {self.steam_csv_path}: {e}")
    
    def load_tag_data(self) -> pd.DataFrame:
        """Load Steam tag/genre data if available"""
        try:
            if self.tags_csv_path and os.path.exists(self.tags_csv_path):
                df = pd.read_csv(self.tags_csv_path)
                print(f"Loaded tag data from {self.tags_csv_path}")
                self.tag_data = df
                return df
            else:
                print("Tag data file not provided or not found")
                return None
        except Exception as e:
            print(f"Error loading tag data: {e}")
            return None
    
    def preprocess_steam_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the Steam dataset with enhanced feature engineering"""
        
        df = df.copy()
        
        # Handle release date
        if 'release_date' in df.columns:
            try:
                # Try multiple date formats
                for date_format in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y']:
                    try:
                        df['release_date'] = pd.to_datetime(df['release_date'], format=date_format)
                        break
                    except:
                        continue
                
                if pd.api.types.is_datetime64_any_dtype(df['release_date']):
                    df['release_month'] = df['release_date'].dt.month
                    df['release_year'] = df['release_date'].dt.year
                else:
                    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
                    df['release_month'] = df['release_date'].dt.month
            except:
                df['release_month'] = np.random.randint(1, 13, len(df))
        else:
            df['release_month'] = np.random.randint(1, 13, len(df))
        
        # Handle price - critical feature
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
        else:
            print("Warning: No price column found, using default values")
            df['price'] = 19.99
        
        # Create is_free flag
        df['is_free'] = (df['price'] == 0).astype(int)
        
        # Handle platforms - parse semicolon-separated values
        if 'platforms' in df.columns:
            try:
                # Parse semicolon-separated platform strings (e.g., "windows;mac;linux")
                df['windows'] = df['platforms'].str.contains('windows', case=False, na=False).astype(int)
                df['mac'] = df['platforms'].str.contains('mac', case=False, na=False).astype(int)
                df['linux'] = df['platforms'].str.contains('linux', case=False, na=False).astype(int)
            except Exception as e:
                print(f"Warning: Error parsing platforms: {e}")
                # Default to Windows if parsing fails
                df['windows'] = 1
                df['mac'] = 0
                df['linux'] = 0
        else:
            # Check for individual platform columns
            df['windows'] = df.get('windows', 1)
            df['mac'] = df.get('mac', 0)
            df['linux'] = df.get('linux', 0)
        
        # Handle owners - critical for prediction
        if 'steamspy_owners' in df.columns:
            df['owners'] = self.parse_owners_range(df['steamspy_owners'])
        elif 'owners' in df.columns:
            # Check if owners contains range strings (e.g., "10000-20000")
            if df['owners'].dtype == 'object':
                # Owners is a string column with ranges
                df['owners'] = self.parse_owners_range(df['owners'])
            else:
                # Owners is already numeric
                df['owners'] = pd.to_numeric(df['owners'], errors='coerce').fillna(1000)
        else:
            # Estimate based on other features if available
            print("Warning: No owners data found, estimating based on price")
            base_owners = 10000
            price_effect = (60 - df['price'].fillna(20)) * 100
            df['owners'] = np.maximum(100, base_owners + price_effect + np.random.normal(0, 2000, len(df)))
        
        
        # Validate owners data
        print(f"  📊 Owners data validation:")
        print(f"    - Type: {df['owners'].dtype}")
        print(f"    - Non-null count: {df['owners'].notna().sum()} / {len(df)}")
        print(f"    - Range: [{df['owners'].min():,.0f}, {df['owners'].max():,.0f}]")
        print(f"    - Mean: {df['owners'].mean():,.0f}")
        print(f"    - Median: {df['owners'].median():,.0f}")
        print(f"    - NaN count: {df['owners'].isna().sum()}")
        print(f"    - Unique values: {df['owners'].nunique()}")
        
        # Calculate review ratio - REQUIRE REAL DATA
        if 'positive_reviews' in df.columns and 'negative_reviews' in df.columns:
            df['positive_reviews'] = pd.to_numeric(df['positive_reviews'], errors='coerce').fillna(0)
            df['negative_reviews'] = pd.to_numeric(df['negative_reviews'], errors='coerce').fillna(0)
            total_reviews = df['positive_reviews'] + df['negative_reviews']
            df['review_ratio'] = np.where(
                total_reviews > 0,
                df['positive_reviews'] / total_reviews,
                0.7  # Default ratio if no reviews for that specific game
            )
            df['positive_ratings'] = df['positive_reviews']
            df['negative_ratings'] = df['negative_reviews']
        elif 'positive' in df.columns and 'negative' in df.columns:
            # Alternative column names
            df['positive_reviews'] = pd.to_numeric(df['positive'], errors='coerce').fillna(0)
            df['negative_reviews'] = pd.to_numeric(df['negative'], errors='coerce').fillna(0)
            total_reviews = df['positive_reviews'] + df['negative_reviews']
            df['review_ratio'] = np.where(
                total_reviews > 0,
                df['positive_reviews'] / total_reviews,
                0.7
            )
            df['positive_ratings'] = df['positive_reviews']
            df['negative_ratings'] = df['negative_reviews']
        elif 'positive_ratings' in df.columns and 'negative_ratings' in df.columns:
            # Already have ratings columns
            df['positive_ratings'] = pd.to_numeric(df['positive_ratings'], errors='coerce').fillna(0)
            df['negative_ratings'] = pd.to_numeric(df['negative_ratings'], errors='coerce').fillna(0)
            total_ratings = df['positive_ratings'] + df['negative_ratings']
            df['review_ratio'] = np.where(
                total_ratings > 0,
                df['positive_ratings'] / total_ratings,
                0.7
            )
            df['positive_reviews'] = df['positive_ratings']
            df['negative_reviews'] = df['negative_ratings']
        else:
            # NO FAKE DATA - Raise error if review data not found
            available_cols = ', '.join(df.columns[:20])
            raise ValueError(
                f"ERROR: Review/rating data not found in steam.csv!\n"
                f"Required columns (one of):\n"
                f"  - 'positive_reviews' and 'negative_reviews'\n"
                f"  - 'positive' and 'negative'\n"
                f"  - 'positive_ratings' and 'negative_ratings'\n\n"
                f"Available columns in your CSV: {available_cols}...\n\n"
                f"Please check your steam.csv file has the correct column names."
            )
        
        # Handle required age
        if 'required_age' in df.columns:
            df['required_age'] = pd.to_numeric(df['required_age'], errors='coerce').fillna(0)
        else:
            df['required_age'] = 0
        
        # Process genres and tags
        if 'genres' in df.columns:
            df = self.process_genres(df)
        
        if 'steamspy_tags' in df.columns:
            df = self.process_tags(df)
        elif 'tags' in df.columns:
            df['steamspy_tags'] = df['tags']
            df = self.process_tags(df)
        
        # If we have the tags CSV, merge it
        if self.tag_data is not None:
            df = self.merge_tag_data(df)
        
        # Ensure we have at least some tag columns
        default_tags = ['Action', 'Adventure', 'Strategy', 'RPG', 'Simulation', 
                       'Indie', 'Multiplayer', 'Singleplayer', 'VR', 'Early_Access',
                       'Casual', 'Sports', 'Racing', 'Puzzle', 'Horror']
        
        for tag in default_tags:
            col_name = f'tag_{tag}'
            if col_name not in df.columns:
                # Try to infer from existing data
                if 'genres' in df.columns or 'steamspy_tags' in df.columns:
                    # Already processed
                    if col_name not in df.columns:
                        df[col_name] = 0
                else:
                    df[col_name] = 0
        
        # Clean up data
        df = df.fillna(0)
        
        # Ensure numeric types for key columns
        numeric_cols = ['price', 'owners', 'review_ratio', 'windows', 'mac', 'linux', 
                       'required_age', 'release_month', 'is_free']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    
    def parse_owners_range(self, owners_str_series):
        """Parse SteamSpy owners range to numeric value (midpoint) and bounds"""
        owners_numeric = []
        owners_lower = []
        owners_upper = []
        
        for owner_str in owners_str_series:
            if pd.isna(owner_str):
                owners_numeric.append(10000)
                owners_lower.append(5000)
                owners_upper.append(20000)
                continue
                
            try:
                owner_str = str(owner_str).strip()
                
                # Handle different formats
                if '..' in owner_str:
                    # Format: "10,000 .. 20,000"
                    parts = owner_str.split('..')
                    lower = int(parts[0].replace(',', '').replace(' ', '').strip())
                    upper = int(parts[1].replace(',', '').replace(' ', '').strip())
                    owners_numeric.append((lower + upper) / 2)
                    owners_lower.append(lower)
                    owners_upper.append(upper)
                elif ' - ' in owner_str:
                    # Format: "10000 - 20000" or "10,000 - 20,000"
                    parts = owner_str.split(' - ')
                    lower = int(parts[0].replace(',', '').strip())
                    upper = int(parts[1].replace(',', '').strip())
                    owners_numeric.append((lower + upper) / 2)
                    owners_lower.append(lower)
                    owners_upper.append(upper)
                elif '-' in owner_str and not owner_str.startswith('-'):
                    # Format: "10000-20000" (most common in steam.csv)
                    parts = owner_str.split('-')
                    if len(parts) == 2:
                        lower = int(parts[0].replace(',', '').strip())
                        upper = int(parts[1].replace(',', '').strip())
                        owners_numeric.append((lower + upper) / 2)
                        owners_lower.append(lower)
                        owners_upper.append(upper)
                    else:
                        # Negative number or complex format
                        owners_numeric.append(10000)
                        owners_lower.append(5000)
                        owners_upper.append(20000)
                else:
                    # Try to parse as single number
                    val = int(str(owner_str).replace(',', ''))
                    owners_numeric.append(val)
                    owners_lower.append(int(val * 0.7))  # ±30% range
                    owners_upper.append(int(val * 1.3))
            except Exception as e:
                print(f"Warning: Could not parse owner string '{owner_str}': {e}")
                owners_numeric.append(10000)
                owners_lower.append(5000)
                owners_upper.append(20000)
        
        return pd.Series(owners_numeric, index=owners_str_series.index)
    
    def process_genres(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process genre information into binary columns"""
        genre_mapping = {
            'Action': ['action', 'shooter', 'fighting', 'combat'],
            'Adventure': ['adventure', 'exploration', 'story'],
            'Strategy': ['strategy', 'rts', 'turn-based', 'tactical'],
            'RPG': ['rpg', 'role-playing', 'role playing'],
            'Simulation': ['simulation', 'simulator', 'sim'],
            'Indie': ['indie', 'independent'],
            'Multiplayer': ['multiplayer', 'multi-player', 'online', 'co-op', 'coop', 'pvp'],
            'Singleplayer': ['singleplayer', 'single-player', 'single player'],
            'VR': ['vr', 'virtual reality'],
            'Early_Access': ['early access', 'early-access'],
            'Casual': ['casual'],
            'Sports': ['sports', 'sport', 'racing', 'football', 'soccer'],
            'Racing': ['racing', 'race', 'driving'],
            'Puzzle': ['puzzle', 'logic'],
            'Horror': ['horror', 'scary', 'terror']
        }
        
        for main_genre, keywords in genre_mapping.items():
            col_name = f'tag_{main_genre}'
            df[col_name] = 0
            
            if 'genres' in df.columns:
                for keyword in keywords:
                    mask = df['genres'].str.contains(keyword, case=False, na=False)
                    df.loc[mask, col_name] = 1
        
        return df
    
    def process_tags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process Steam tags into binary columns"""
        if 'steamspy_tags' not in df.columns:
            return df
        
        # Important tags to track
        important_tags = [
            'action', 'adventure', 'strategy', 'rpg', 'simulation',
            'indie', 'multiplayer', 'singleplayer', 'single-player',
            'vr', 'early access', 'free to play', 'puzzle', 'casual',
            'sports', 'racing', 'horror', 'survival', 'open world',
            'fps', 'platformer', 'roguelike', 'sandbox', 'tactical'
        ]
        
        for tag in important_tags:
            # Clean tag name for column
            tag_col = f'tag_{tag.replace(" ", "_").replace("-", "_").title()}'
            
            # Check if tag exists in steamspy_tags
            mask = df['steamspy_tags'].str.contains(tag, case=False, na=False)
            df[tag_col] = mask.astype(int)
        
        return df
    
    def merge_tag_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge additional tag data if available"""
        if self.tag_data is None:
            return df
        
        try:
            # Merge on app_id if available
            if 'appid' in self.tag_data.columns and 'steam_appid' in df.columns:
                df = df.merge(self.tag_data, 
                            left_on='steam_appid', 
                            right_on='appid', 
                            how='left',
                            suffixes=('', '_tag'))
                
                # Process any additional tag columns
                if 'steamspy_tags' in self.tag_data.columns:
                    df['steamspy_tags'] = df['steamspy_tags_tag'].fillna(df['steamspy_tags'])
                    df = self.process_tags(df)
        except Exception as e:
            print(f"Could not merge tag data: {e}")
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns for model training"""
        # Define columns to exclude from features
        exclude_cols = [
            'name', 'steam_appid', 'appid', 'owners', 'review_ratio',
            'positive_reviews', 'negative_reviews', 'positive', 'negative',
            'release_date', 'steamspy_owners', 'genres', 'steamspy_tags',
            'tags', 'platforms', 'description', 'short_description',
            'developer', 'publisher', 'categories'
        ]
        
        if self.data is not None:
            # Get all numeric columns that are not in exclude list
            feature_cols = []
            for col in self.data.columns:
                if col not in exclude_cols:
                    # Check if column is numeric or binary
                    if pd.api.types.is_numeric_dtype(self.data[col]):
                        feature_cols.append(col)
            return feature_cols
        else:
            # Return default feature columns
            return ['price', 'release_month', 'windows', 'mac', 'linux', 
                   'is_free', 'required_age'] + \
                   [f'tag_{tag}' for tag in ['Action', 'Adventure', 'Strategy', 
                    'RPG', 'Simulation', 'Indie', 'Multiplayer', 'Singleplayer', 
                    'VR', 'Early_Access', 'Casual', 'Sports', 'Racing', 'Puzzle', 'Horror']]
    
    def split_features_targets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Split data into features and target variables"""
        feature_cols = self.get_feature_columns()
        
        # Ensure all required columns exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        X = df[feature_cols]
        y_owners = df['owners'] if 'owners' in df.columns else pd.Series([10000] * len(df))
        y_reviews = df['review_ratio'] if 'review_ratio' in df.columns else pd.Series([0.7] * len(df))
        
        return X, y_owners, y_reviews
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics of the loaded data"""
        if self.data is None:
            return {}
        
        summary = {
            'total_games': len(self.data),
            'price_range': (self.data['price'].min(), self.data['price'].max()),
            'avg_owners': self.data['owners'].mean() if 'owners' in self.data.columns else 0,
            'avg_review_ratio': self.data['review_ratio'].mean() if 'review_ratio' in self.data.columns else 0,
            'platform_distribution': {
                'Windows': self.data['windows'].sum() if 'windows' in self.data.columns else 0,
                'Mac': self.data['mac'].sum() if 'mac' in self.data.columns else 0,
                'Linux': self.data['linux'].sum() if 'linux' in self.data.columns else 0
            },
            'feature_count': len(self.get_feature_columns())
        }
        
        return summary

# Example usage
if __name__ == "__main__":
    # Check for Steam CSV files
    steam_path = "steam.csv"
    tags_path = "steamspy_tag_data.csv"
    
    if not os.path.exists(steam_path):
        print(f"ERROR: Required file '{steam_path}' not found!")
        print("Please ensure steam.csv is in the current directory.")
        exit(1)
    
    try:
        # Initialize loader with Steam data
        loader = SteamDataLoader(
            steam_csv_path=steam_path,
            tags_csv_path=tags_path if os.path.exists(tags_path) else None
        )
        
        # Load and preprocess data
        steam_data = loader.load_steam_data()
        
        # Load tag data if available
        if os.path.exists(tags_path):
            tag_data = loader.load_tag_data()
        
        # Get summary
        summary = loader.get_data_summary()
        
        print(f"\n✅ Successfully loaded Steam data!")
        print(f"📊 Data Summary:")
        print(f"  - Total games: {summary['total_games']}")
        print(f"  - Price range: ${summary['price_range'][0]:.2f} - ${summary['price_range'][1]:.2f}")
        print(f"  - Average owners: {summary['avg_owners']:,.0f}")
        print(f"  - Average review ratio: {summary['avg_review_ratio']:.1%}")
        print(f"  - Platform distribution: {summary['platform_distribution']}")
        print(f"  - Features available: {summary['feature_count']}")
        
        # Show sample of data
        print(f"\n📋 Sample data (first 5 games):")
        print(steam_data[['name', 'price', 'owners', 'review_ratio']].head() if 'name' in steam_data.columns else steam_data.head())
        
        # Get feature columns
        feature_cols = loader.get_feature_columns()
        print(f"\n🎯 Feature columns for model training ({len(feature_cols)}):")
        print(f"  {', '.join(feature_cols[:10])}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nPlease ensure:")
        print("1. steam.csv exists and is properly formatted")
        print("2. The file contains required columns (price, platforms, etc.)")
        print("3. The file is not corrupted")
