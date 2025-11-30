"""
Stratified Ranking Analysis - Check if model ranks well WITHIN categories
but poorly ACROSS categories

This is critical: If Spearman is high within homogeneous groups but low overall,
the percentile system is more reliable than it appears!
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


def analyze_stratified_ranking():
    """
    Analyze ranking accuracy within specific game categories

    Hypothesis: Model may rank well within similar games (indie vs indie)
    but struggle across different types (indie vs AAA)
    """

    if 'models' not in st.session_state or 'df' not in st.session_state:
        st.error("Train models and load data first!")
        return

    models = st.session_state.models
    df = st.session_state.df

    # Get test set predictions
    y_actual = models['y_test_actual']
    y_pred = models['owners_pred']
    X_test = models['X_test']

    # Get overall Spearman
    overall_spearman, _ = spearmanr(y_actual, y_pred)

    st.title("🎯 Stratified Ranking Analysis")

    st.info(f"""
    **Overall Spearman Correlation: {overall_spearman:.4f}**

    This analysis checks if the model ranks well WITHIN specific game categories,
    even if cross-category ranking is poor.
    """)

    results = []

    # ============================================================================
    # 1. GENRE-BASED RANKING (Non-tunable by user)
    # ============================================================================
    st.markdown("## 1. Ranking Accuracy by Genre")
    st.markdown("*These are fixed characteristics the user cannot change*")

    genre_cols = [c for c in X_test.columns if c.startswith('genre_')]

    for genre_col in genre_cols[:15]:  # Top 15 genres
        genre_name = genre_col.replace('genre_', '').replace('_', ' ').title()

        # Get games in this genre (in test set)
        genre_mask = X_test[genre_col] == 1

        if genre_mask.sum() >= 30:  # Need at least 30 games for reliable stats
            y_actual_genre = y_actual[genre_mask]
            y_pred_genre = y_pred[genre_mask.values]

            spearman_genre, p_value = spearmanr(y_actual_genre, y_pred_genre)

            results.append({
                'Category': 'Genre',
                'Subcategory': genre_name,
                'Games': genre_mask.sum(),
                'Spearman': spearman_genre,
                'p-value': p_value,
                'vs Overall': spearman_genre - overall_spearman
            })

    # ============================================================================
    # 2. PRICE TIER RANKING (Non-tunable once set)
    # ============================================================================
    st.markdown("## 2. Ranking Accuracy by Price Tier")

    # Create price tiers
    if 'price' in X_test.columns:
        price_series = X_test['price']

        price_tiers = {
            'Free (F2P)': (price_series == 0),
            'Budget ($0.01-$5)': (price_series > 0) & (price_series <= 5),
            'Low ($5-$10)': (price_series > 5) & (price_series <= 10),
            'Mid ($10-$20)': (price_series > 10) & (price_series <= 20),
            'Premium ($20-$30)': (price_series > 20) & (price_series <= 30),
            'AAA ($30+)': (price_series > 30)
        }

        for tier_name, tier_mask in price_tiers.items():
            if tier_mask.sum() >= 30:
                y_actual_tier = y_actual[tier_mask]
                y_pred_tier = y_pred[tier_mask.values]

                spearman_tier, p_value = spearmanr(y_actual_tier, y_pred_tier)

                results.append({
                    'Category': 'Price Tier',
                    'Subcategory': tier_name,
                    'Games': tier_mask.sum(),
                    'Spearman': spearman_tier,
                    'p-value': p_value,
                    'vs Overall': spearman_tier - overall_spearman
                })

    # ============================================================================
    # 3. DEVELOPER REPUTATION (Non-tunable)
    # ============================================================================
    st.markdown("## 3. Ranking Accuracy by Developer Tier")

    dev_cols = [c for c in X_test.columns if c.startswith('dev_top_')]

    if dev_cols:
        # Check if game is from top developer
        has_top_dev = X_test[dev_cols].sum(axis=1) > 0

        categories = {
            'Top Developer': has_top_dev,
            'Unknown Developer': ~has_top_dev
        }

        for cat_name, cat_mask in categories.items():
            if cat_mask.sum() >= 30:
                y_actual_cat = y_actual[cat_mask]
                y_pred_cat = y_pred[cat_mask.values]

                spearman_cat, p_value = spearmanr(y_actual_cat, y_pred_cat)

                results.append({
                    'Category': 'Developer Reputation',
                    'Subcategory': cat_name,
                    'Games': cat_mask.sum(),
                    'Spearman': spearman_cat,
                    'p-value': p_value,
                    'vs Overall': spearman_cat - overall_spearman
                })

    # ============================================================================
    # 4. RELEASE YEAR (Non-tunable once released)
    # ============================================================================
    st.markdown("## 4. Ranking Accuracy by Release Era")

    if 'release_year' in X_test.columns:
        year_series = X_test['release_year']

        year_ranges = {
            'Very Old (pre-2015)': year_series < 2015,
            'Old (2015-2017)': (year_series >= 2015) & (year_series < 2018),
            'Recent (2018-2020)': (year_series >= 2018) & (year_series < 2021),
            'New (2021+)': year_series >= 2021
        }

        for range_name, range_mask in year_ranges.items():
            if range_mask.sum() >= 30:
                y_actual_range = y_actual[range_mask]
                y_pred_range = y_pred[range_mask.values]

                spearman_range, p_value = spearmanr(y_actual_range, y_pred_range)

                results.append({
                    'Category': 'Release Era',
                    'Subcategory': range_name,
                    'Games': range_mask.sum(),
                    'Spearman': spearman_range,
                    'p-value': p_value,
                    'vs Overall': spearman_range - overall_spearman
                })

    # ============================================================================
    # RESULTS SUMMARY
    # ============================================================================

    df_results = pd.DataFrame(results)

    st.markdown("## 📊 Results Summary")

    # Overall statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Overall Spearman", f"{overall_spearman:.4f}")

    with col2:
        better_than_overall = (df_results['Spearman'] > overall_spearman).sum()
        st.metric("Categories Better Than Overall", f"{better_than_overall}/{len(df_results)}")

    with col3:
        max_spearman = df_results['Spearman'].max()
        st.metric("Best Category Spearman", f"{max_spearman:.4f}")

    # Detailed results table
    st.markdown("### Detailed Results")

    # Color code by performance
    def color_spearman(val):
        if val >= 0.70:
            return 'background-color: #d4edda'  # Green
        elif val >= 0.60:
            return 'background-color: #d1ecf1'  # Light blue
        elif val >= 0.50:
            return 'background-color: #fff3cd'  # Yellow
        else:
            return 'background-color: #f8d7da'  # Red

    styled_df = df_results.style.applymap(
        color_spearman,
        subset=['Spearman']
    ).format({
        'Spearman': '{:.4f}',
        'p-value': '{:.4f}',
        'vs Overall': '{:+.4f}'
    })

    st.dataframe(styled_df, use_container_width=True)

    # ============================================================================
    # VISUALIZATION
    # ============================================================================

    st.markdown("### Spearman Correlation by Category")

    # Sort by Spearman descending
    df_results_sorted = df_results.sort_values('Spearman', ascending=True)

    fig = go.Figure()

    # Add overall line
    fig.add_hline(
        y=overall_spearman,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Overall: {overall_spearman:.4f}",
        annotation_position="right"
    )

    # Add bars
    fig.add_trace(go.Bar(
        x=df_results_sorted['Spearman'],
        y=df_results_sorted['Subcategory'],
        orientation='h',
        marker=dict(
            color=df_results_sorted['Spearman'],
            colorscale='RdYlGn',
            cmin=0,
            cmax=1,
            colorbar=dict(title="Spearman")
        ),
        text=df_results_sorted['Spearman'].round(4),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Spearman: %{x:.4f}<br>Games: %{customdata}<extra></extra>',
        customdata=df_results_sorted['Games']
    ))

    fig.update_layout(
        title="Ranking Accuracy by Game Category",
        xaxis_title="Spearman Correlation",
        yaxis_title="Category",
        height=max(400, len(df_results_sorted) * 25),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # ============================================================================
    # KEY FINDINGS
    # ============================================================================

    st.markdown("## 🔍 Key Findings")

    # Find best and worst categories
    best_cat = df_results.loc[df_results['Spearman'].idxmax()]
    worst_cat = df_results.loc[df_results['Spearman'].idxmin()]

    st.success(f"""
    **Best Ranking Accuracy:**
    - Category: {best_cat['Category']} - {best_cat['Subcategory']}
    - Spearman: {best_cat['Spearman']:.4f} ({best_cat['Games']} games)
    - Improvement over overall: {best_cat['vs Overall']:+.4f}
    """)

    st.error(f"""
    **Worst Ranking Accuracy:**
    - Category: {worst_cat['Category']} - {worst_cat['Subcategory']}
    - Spearman: {worst_cat['Spearman']:.4f} ({worst_cat['Games']} games)
    - Difference from overall: {worst_cat['vs Overall']:+.4f}
    """)

    # ============================================================================
    # INTERPRETATION & RECOMMENDATIONS
    # ============================================================================

    st.markdown("## 💡 Interpretation")

    # Check if model is good within categories but bad overall
    avg_category_spearman = df_results['Spearman'].mean()
    categories_better = (df_results['Spearman'] > overall_spearman).sum()
    pct_better = (categories_better / len(df_results)) * 100

    if avg_category_spearman > overall_spearman + 0.10:
        st.success(f"""
        ✅ **Great News: Within-Category Ranking is Much Better!**

        - Overall Spearman: {overall_spearman:.4f}
        - Average within categories: {avg_category_spearman:.4f}
        - {categories_better}/{len(df_results)} categories ({pct_better:.0f}%) perform better than overall

        **What this means:**
        - The model ranks well WITHIN similar games (e.g., Indie vs Indie)
        - Poor overall score is due to cross-category comparisons (Indie vs AAA)
        - **For your IDSS, this is actually GOOD** because users naturally compare within their market segment

        **Recommendation:**
        - Show percentiles WITHIN the user's category (e.g., "75th among Indie games")
        - De-emphasize overall market percentile
        - This makes your system MORE reliable than the 0.43 overall Spearman suggests!
        """)
    elif avg_category_spearman > overall_spearman:
        st.info(f"""
        ℹ️ **Slight Improvement Within Categories**

        - Overall Spearman: {overall_spearman:.4f}
        - Average within categories: {avg_category_spearman:.4f}
        - Improvement: {avg_category_spearman - overall_spearman:.4f}

        The model performs marginally better within categories, but not dramatically.
        """)
    else:
        st.warning(f"""
        ⚠️ **No Improvement Within Categories**

        - Overall Spearman: {overall_spearman:.4f}
        - Average within categories: {avg_category_spearman:.4f}

        The model struggles uniformly across all categories. Consider:
        1. Adding more predictive features
        2. Using ranking-specific loss functions
        3. Collecting better quality data
        """)

    # Check for specific high-performing categories
    high_performing = df_results[df_results['Spearman'] >= 0.60]

    if len(high_performing) > 0:
        st.markdown("### 🎯 High-Performing Categories (Spearman ≥ 0.60)")

        st.markdown("These categories have **good ranking accuracy** and can be trusted:")

        for _, row in high_performing.iterrows():
            st.markdown(f"- **{row['Subcategory']}** ({row['Category']}): Spearman = {row['Spearman']:.4f}")

        st.info(f"""
        **Actionable Insight:**

        For games in these categories, you can confidently show percentile rankings.
        Consider offering a **"confidence indicator"** in your UI:

        - High confidence (🟢): Categories with Spearman > 0.60
        - Medium confidence (🟡): Categories with Spearman 0.40-0.60
        - Low confidence (🔴): Categories with Spearman < 0.40
        """)


if __name__ == "__main__":
    analyze_stratified_ranking()
