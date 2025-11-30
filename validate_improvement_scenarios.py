"""
Validate improvement scenarios using historical evidence
Instead of relying on model predictions, use actual data comparisons
"""

import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st
from typing import Dict, List, Tuple


def validate_scenario_with_evidence(df: pd.DataFrame,
                                     feature: str,
                                     has_feature_condition,
                                     lacks_feature_condition,
                                     control_for: List[str] = None) -> Dict:
    """
    Validate if a feature actually increases owners using historical data

    Args:
        df: Full dataset
        feature: Name of feature being tested
        has_feature_condition: Boolean mask for games WITH the feature
        lacks_feature_condition: Boolean mask for games WITHOUT the feature
        control_for: List of features to control for (e.g., genre, price tier)

    Returns:
        Dict with evidence metrics
    """

    # Get games with and without the feature
    with_feature = df[has_feature_condition]
    without_feature = df[lacks_feature_condition]

    if len(with_feature) < 30 or len(without_feature) < 30:
        return {
            'valid': False,
            'reason': f'Insufficient data (with={len(with_feature)}, without={len(without_feature)})'
        }

    # Calculate effect size
    owners_with = with_feature['owners']
    owners_without = without_feature['owners']

    # Median lift (more robust than mean)
    median_with = owners_with.median()
    median_without = owners_without.median()
    median_lift = (median_with - median_without) / median_without

    # Mean ratio
    mean_with = owners_with.mean()
    mean_without = owners_without.mean()
    mean_ratio = mean_with / mean_without

    # Statistical significance test (Mann-Whitney U test, non-parametric)
    statistic, p_value = stats.mannwhitneyu(owners_with, owners_without, alternative='greater')

    # Effect size (Cohen's d for log-transformed data)
    log_with = np.log1p(owners_with)
    log_without = np.log1p(owners_without)
    cohen_d = (log_with.mean() - log_without.mean()) / np.sqrt((log_with.std()**2 + log_without.std()**2) / 2)

    # Determine confidence level
    if p_value < 0.001 and abs(cohen_d) > 0.5:
        confidence = "Very High"
        confidence_score = 95
    elif p_value < 0.01 and abs(cohen_d) > 0.3:
        confidence = "High"
        confidence_score = 85
    elif p_value < 0.05 and abs(cohen_d) > 0.2:
        confidence = "Moderate"
        confidence_score = 70
    elif p_value < 0.10:
        confidence = "Low"
        confidence_score = 55
    else:
        confidence = "Very Low"
        confidence_score = 30

    # Only consider valid if statistically significant
    is_valid = p_value < 0.05 and median_lift > 0

    return {
        'valid': is_valid,
        'feature': feature,
        'sample_with': len(with_feature),
        'sample_without': len(without_feature),
        'median_with': median_with,
        'median_without': median_without,
        'median_lift_pct': median_lift * 100,
        'mean_ratio': mean_ratio,
        'cohen_d': cohen_d,
        'p_value': p_value,
        'confidence': confidence,
        'confidence_score': confidence_score,
        'is_significant': p_value < 0.05
    }


def validate_all_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test all common improvement scenarios with historical evidence
    """

    results = []

    # Scenario 1: Multiplayer
    if 'tag_multiplayer' in df.columns or 'tag_Multiplayer' in df.columns:
        mp_col = 'tag_multiplayer' if 'tag_multiplayer' in df.columns else 'tag_Multiplayer'

        result = validate_scenario_with_evidence(
            df,
            feature="Multiplayer Support",
            has_feature_condition=df[mp_col] == 1,
            lacks_feature_condition=df[mp_col] == 0
        )
        if result['valid']:
            results.append(result)

    # Scenario 2: Multi-platform (all 3 platforms)
    if all(col in df.columns for col in ['windows', 'mac', 'linux']):
        all_platforms = (df['windows'] == 1) & (df['mac'] == 1) & (df['linux'] == 1)
        single_platform = (df['windows'] == 1) & (df['mac'] == 0) & (df['linux'] == 0)

        result = validate_scenario_with_evidence(
            df,
            feature="All Platforms (Win+Mac+Linux)",
            has_feature_condition=all_platforms,
            lacks_feature_condition=single_platform
        )
        if result['valid']:
            results.append(result)

    # Scenario 3: Steam Trading Cards
    if 'cat_steam_trading_cards' in df.columns:
        result = validate_scenario_with_evidence(
            df,
            feature="Steam Trading Cards",
            has_feature_condition=df['cat_steam_trading_cards'] == 1,
            lacks_feature_condition=df['cat_steam_trading_cards'] == 0
        )
        if result['valid']:
            results.append(result)

    # Scenario 4: Achievements (20+)
    if 'achievements' in df.columns:
        has_achievements = df['achievements'] >= 20
        no_achievements = df['achievements'] == 0

        result = validate_scenario_with_evidence(
            df,
            feature="Achievements (20+)",
            has_feature_condition=has_achievements,
            lacks_feature_condition=no_achievements
        )
        if result['valid']:
            results.append(result)

    # Scenario 5: Steam Cloud
    if 'cat_steam_cloud' in df.columns:
        result = validate_scenario_with_evidence(
            df,
            feature="Steam Cloud",
            has_feature_condition=df['cat_steam_cloud'] == 1,
            lacks_feature_condition=df['cat_steam_cloud'] == 0
        )
        if result['valid']:
            results.append(result)

    # Scenario 6: Controller Support
    if 'cat_full_controller_support' in df.columns:
        result = validate_scenario_with_evidence(
            df,
            feature="Full Controller Support",
            has_feature_condition=df['cat_full_controller_support'] == 1,
            lacks_feature_condition=df['cat_full_controller_support'] == 0
        )
        if result['valid']:
            results.append(result)

    # Scenario 7: Free-to-Play
    if 'price' in df.columns:
        is_free = df['price'] == 0
        is_paid = df['price'] > 0

        result = validate_scenario_with_evidence(
            df,
            feature="Free-to-Play Model",
            has_feature_condition=is_free,
            lacks_feature_condition=is_paid
        )
        if result['valid']:
            results.append(result)

    # Convert to DataFrame
    if results:
        df_results = pd.DataFrame(results)
        # Sort by confidence score
        df_results = df_results.sort_values('confidence_score', ascending=False)
        return df_results
    else:
        return pd.DataFrame()


def display_validated_scenarios():
    """Streamlit page to display validated improvement scenarios"""

    st.title("✅ Evidence-Based Improvement Scenarios")

    st.markdown("""
    These recommendations are based on **historical evidence** from 27,000+ Steam games,
    not model predictions. Each scenario shows the **actual difference** between games
    with and without the feature.
    """)

    if 'df' not in st.session_state:
        st.error("Dataset not loaded. Analyze a game first.")
        return

    df = st.session_state.df

    with st.spinner("Validating scenarios with historical data..."):
        results = validate_all_scenarios(df)

    if len(results) == 0:
        st.warning("No scenarios passed statistical validation.")
        return

    st.success(f"✅ Found {len(results)} scenarios backed by strong evidence")

    st.markdown("---")

    # Display each validated scenario
    for idx, row in results.iterrows():

        # Color code by confidence
        if row['confidence_score'] >= 85:
            box_color = "#d4edda"  # Green
            icon = "🟢"
        elif row['confidence_score'] >= 70:
            box_color = "#d1ecf1"  # Blue
            icon = "🔵"
        else:
            box_color = "#fff3cd"  # Yellow
            icon = "🟡"

        st.markdown(f"""
        <div style="background-color: {box_color}; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
            <h3>{icon} {row['feature']}</h3>
        </div>
        """, unsafe_allow_html=True)

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Median Lift",
                f"+{row['median_lift_pct']:.0f}%",
                help=f"Games with this feature have {row['median_lift_pct']:.0f}% more owners (median)"
            )

        with col2:
            st.metric(
                "Mean Ratio",
                f"{row['mean_ratio']:.2f}x",
                help=f"Games with this feature have {row['mean_ratio']:.2f}x more owners on average"
            )

        with col3:
            st.metric(
                "Confidence",
                row['confidence'],
                help=f"Statistical confidence: {row['confidence_score']}%"
            )

        with col4:
            st.metric(
                "Sample Size",
                f"{row['sample_with']:,}",
                delta=f"vs {row['sample_without']:,} without",
                delta_color="off"
            )

        # Evidence details
        with st.expander("📊 Statistical Evidence"):
            st.markdown(f"""
            **Effect Size (Cohen's d):** {row['cohen_d']:.3f}
            - < 0.2: Small effect
            - 0.2-0.5: Medium effect
            - > 0.5: Large effect

            **p-value:** {row['p_value']:.6f}
            - < 0.001: Very strong evidence
            - < 0.01: Strong evidence
            - < 0.05: Moderate evidence

            **Sample Sizes:**
            - Games WITH {row['feature']}: {row['sample_with']:,}
            - Games WITHOUT {row['feature']}: {row['sample_without']:,}

            **Median Owners:**
            - With feature: {row['median_with']:,.0f}
            - Without feature: {row['median_without']:,.0f}
            - **Difference: +{row['median_lift_pct']:.1f}%**
            """)

        st.markdown("---")

    # Summary recommendation
    st.markdown("## 💡 Summary")

    high_confidence = results[results['confidence_score'] >= 85]

    if len(high_confidence) > 0:
        st.success(f"""
        **{len(high_confidence)} High-Confidence Recommendations:**

        {', '.join(high_confidence['feature'].tolist())}

        These features have **strong historical evidence** of increasing game reach.
        """)

    # Download results
    st.markdown("### 💾 Export Results")
    csv = results.to_csv(index=False)
    st.download_button(
        "Download Evidence Report",
        csv,
        "validated_scenarios.csv",
        "text/csv"
    )


if __name__ == "__main__":
    display_validated_scenarios()
