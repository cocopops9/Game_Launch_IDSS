#!/usr/bin/env python3
"""
Diagnostic Script for Game Launch IDSS
Verifies that v1.0.6 fixes are working correctly
"""

import pandas as pd
import numpy as np
from data_loader import SteamDataLoader

def print_section(title):
    """Print section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def check_raw_data():
    """Check the raw CSV format"""
    print_section("1. RAW CSV DATA CHECK")
    
    try:
        df = pd.read_csv('steam.csv', nrows=10)
        
        print("\n📋 First 10 rows of 'owners' column:")
        print(df['owners'].to_string())
        
        print(f"\n📊 Column dtype: {df['owners'].dtype}")
        
        if df['owners'].dtype == 'object':
            print("✅ Owners column contains strings (expected for ranges)")
            
            # Check format
            sample = str(df['owners'].iloc[0])
            if '-' in sample:
                print(f"✅ Range format detected: '{sample}'")
            else:
                print(f"⚠️  Unexpected format: '{sample}'")
        else:
            print("⚠️  Owners column is numeric (might be pre-processed)")
            
    except Exception as e:
        print(f"❌ Error reading raw CSV: {e}")

def check_parsed_data():
    """Check data after loading through data_loader"""
    print_section("2. PARSED DATA CHECK")
    
    try:
        loader = SteamDataLoader('steam.csv')
        df = loader.load_steam_data()
        
        print(f"\n📊 Total games loaded: {len(df):,}")
        print(f"📊 Owners column dtype: {df['owners'].dtype}")
        print(f"📊 Owners non-null: {df['owners'].notna().sum():,} / {len(df):,}")
        print(f"📊 Owners NaN count: {df['owners'].isna().sum():,}")
        
        print(f"\n📈 Owners Statistics:")
        print(f"  - Minimum: {df['owners'].min():,.0f}")
        print(f"  - Maximum: {df['owners'].max():,.0f}")
        print(f"  - Mean: {df['owners'].mean():,.0f}")
        print(f"  - Median: {df['owners'].median():,.0f}")
        print(f"  - Std Dev: {df['owners'].std():,.0f}")
        
        print(f"\n🎯 Unique Values Check:")
        unique_count = df['owners'].nunique()
        print(f"  - Unique owners values: {unique_count:,}")
        
        if unique_count == 1:
            print("  ❌ ERROR: All owners have the same value!")
            print(f"     Value: {df['owners'].iloc[0]}")
            print("     → Parsing failed, all defaulted to same number")
        elif unique_count < 10:
            print(f"  ⚠️  WARNING: Only {unique_count} unique values (expected thousands)")
        else:
            print(f"  ✅ Good: {unique_count:,} unique values")
        
        print(f"\n📊 Sample of parsed values:")
        print(df[['name', 'owners']].head(10).to_string(index=False))
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_correlations(df):
    """Check if correlations are working"""
    print_section("3. CORRELATION CHECK")
    
    if df is None:
        print("❌ Cannot check correlations - data loading failed")
        return
    
    try:
        # Check if we have necessary columns
        required = ['owners', 'price', 'windows', 'mac', 'linux']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            print(f"⚠️  Missing columns: {missing}")
            return
        
        # Calculate correlation
        test_cols = ['price', 'windows', 'mac', 'linux', 'owners']
        corr_df = df[test_cols].copy()
        
        # Clean data
        for col in corr_df.columns:
            corr_df[col] = pd.to_numeric(corr_df[col], errors='coerce').fillna(0)
        
        correlations = corr_df.corr()
        
        print("\n📈 Correlations with Owners:")
        for col in ['price', 'windows', 'mac', 'linux']:
            corr_value = correlations.loc[col, 'owners']
            if pd.isna(corr_value):
                print(f"  - {col}: ❌ NaN (ERROR)")
            else:
                status = "✅" if abs(corr_value) > 0.01 else "⚠️"
                print(f"  - {col}: {status} {corr_value:.3f}")
        
        # Check for NaN in correlation matrix
        nan_count = correlations.isna().sum().sum()
        if nan_count > 0:
            print(f"\n❌ ERROR: {nan_count} NaN values in correlation matrix")
        else:
            print(f"\n✅ No NaN values in correlation matrix")
            
    except Exception as e:
        print(f"❌ Error calculating correlations: {e}")
        import traceback
        traceback.print_exc()

def check_distribution(df):
    """Check distribution of owners"""
    print_section("4. DISTRIBUTION CHECK")
    
    if df is None:
        print("❌ Cannot check distribution - data loading failed")
        return
    
    try:
        # Create bins
        bins = [0, 1000, 10000, 100000, 1000000, 10000000, 100000000, float('inf')]
        labels = ['<1K', '1K-10K', '10K-100K', '100K-1M', '1M-10M', '10M-100M', '>100M']
        
        df['owners_bin'] = pd.cut(df['owners'], bins=bins, labels=labels)
        
        print("\n📊 Owners Distribution:")
        dist = df['owners_bin'].value_counts().sort_index()
        
        for label in labels:
            count = dist.get(label, 0)
            pct = (count / len(df)) * 100
            bar = '█' * int(pct / 2)
            print(f"  {label:>10}: {count:>6,} ({pct:>5.1f}%) {bar}")
        
        if df['owners_bin'].iloc[0] == df['owners_bin'].iloc[-1]:
            print("\n⚠️  WARNING: All values in same bin")
        else:
            print("\n✅ Values distributed across multiple bins")
            
    except Exception as e:
        print(f"❌ Error checking distribution: {e}")

def check_platforms(df):
    """Check platform parsing"""
    print_section("5. PLATFORM CHECK")
    
    if df is None:
        print("❌ Cannot check platforms - data loading failed")
        return
    
    try:
        for platform in ['windows', 'mac', 'linux']:
            if platform in df.columns:
                count = df[platform].sum()
                pct = (count / len(df)) * 100
                
                if pct == 100:
                    status = "⚠️  WARNING"
                elif pct == 0:
                    status = "⚠️  WARNING"
                else:
                    status = "✅"
                    
                print(f"  {platform.capitalize():>7}: {count:>6,} / {len(df):,} ({pct:>5.1f}%) {status}")
            else:
                print(f"  {platform.capitalize():>7}: ❌ Column not found")
        
        # Check for multi-platform games
        if all(col in df.columns for col in ['windows', 'mac', 'linux']):
            multi = ((df['windows'] == 1) & (df['mac'] == 1) & (df['linux'] == 1)).sum()
            print(f"\n  Multi-platform (all 3): {multi:,} games ({multi/len(df)*100:.1f}%)")
            
    except Exception as e:
        print(f"❌ Error checking platforms: {e}")

def check_features(df):
    """Check available features"""
    print_section("6. FEATURE CHECK")
    
    if df is None:
        print("❌ Cannot check features - data loading failed")
        return
    
    try:
        loader = SteamDataLoader('steam.csv')
        feature_cols = loader.get_feature_columns()
        
        print(f"\n📊 Total features: {len(feature_cols)}")
        print(f"\n🎯 Feature categories:")
        
        basic = [f for f in feature_cols if f in ['price', 'release_month', 'required_age', 'is_free']]
        platforms = [f for f in feature_cols if f in ['windows', 'mac', 'linux']]
        tags = [f for f in feature_cols if f.startswith('tag_')]
        other = [f for f in feature_cols if f not in basic + platforms + tags]
        
        print(f"  - Basic features: {len(basic)}")
        print(f"  - Platform features: {len(platforms)}")
        print(f"  - Tag features: {len(tags)}")
        print(f"  - Other features: {len(other)}")
        
        if len(feature_cols) == 0:
            print("\n❌ ERROR: No features detected!")
        elif len(feature_cols) < 10:
            print(f"\n⚠️  WARNING: Only {len(feature_cols)} features (expected 20-30)")
        else:
            print(f"\n✅ Good: {len(feature_cols)} features available")
            
        # Show top tags
        if tags:
            print(f"\n  Sample tag features:")
            for tag in tags[:5]:
                tag_name = tag.replace('tag_', '').replace('_', ' ').title()
                count = df[tag].sum() if tag in df.columns else 0
                print(f"    - {tag_name}: {count:,} games")
                
    except Exception as e:
        print(f"❌ Error checking features: {e}")

def main():
    """Run all diagnostic checks"""
    print("\n" + "🔍" * 30)
    print("  GAME LAUNCH IDSS - DIAGNOSTIC REPORT")
    print("  Version 1.0.6 Verification")
    print("🔍" * 30)
    
    # Check 1: Raw data
    check_raw_data()
    
    # Check 2: Parsed data
    df = check_parsed_data()
    
    # Check 3: Correlations
    check_correlations(df)
    
    # Check 4: Distribution
    check_distribution(df)
    
    # Check 5: Platforms
    check_platforms(df)
    
    # Check 6: Features
    check_features(df)
    
    # Final summary
    print_section("SUMMARY")
    
    if df is not None:
        unique = df['owners'].nunique()
        nan_owners = df['owners'].isna().sum()
        
        issues = []
        
        if unique == 1:
            issues.append("❌ All owners have same value (parsing failed)")
        elif unique < 100:
            issues.append(f"⚠️  Low unique values ({unique})")
            
        if nan_owners > 0:
            issues.append(f"⚠️  {nan_owners} NaN values in owners")
            
        if df['owners'].mean() < 100:
            issues.append("❌ Mean owners unreasonably low")
            
        if not issues:
            print("\n🎉 ALL CHECKS PASSED!")
            print("\n✅ Data is loading correctly")
            print("✅ Owners are being parsed from ranges")
            print("✅ Values are realistic and varied")
            print("✅ Ready to train models")
        else:
            print("\n⚠️  ISSUES DETECTED:")
            for issue in issues:
                print(f"  {issue}")
            print("\n📝 Please review the detailed output above")
            print("💡 Refer to FIX_GUIDE_v1.0.6.md for troubleshooting")
    else:
        print("\n❌ DATA LOADING FAILED")
        print("📝 Check error messages above")
        print("💡 Ensure steam.csv exists and is properly formatted")
    
    print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    main()
