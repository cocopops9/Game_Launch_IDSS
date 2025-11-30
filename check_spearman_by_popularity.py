"""
Check if model ranks better within popularity clusters
(comparing low vs low, mid vs mid, high vs high)
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import streamlit as st


def check_spearman_by_popularity():
    """Check if ranking is better within popularity tiers"""

    st.title("📊 Spearman by Popularity Tier")

    if 'models' not in st.session_state:
        st.error("Models not trained")
        return

    models = st.session_state.models
    y_actual = models.get('y_owners_actual_test')
    X_test = models.get('X_test')
    owners_model = models.get('owners_model')

    if y_actual is None or X_test is None or owners_model is None:
        st.error("Test data not available")
        return

    # Generate predictions
    owners_pred_log = owners_model.predict(X_test)
    y_pred = np.expm1(owners_pred_log)

    # Overall Spearman
    overall_spearman, _ = spearmanr(y_actual, y_pred)
    st.info(f"**Overall Spearman: {overall_spearman:.4f}**")

    st.markdown("---")
    st.markdown("## By Actual Owner Count (Ground Truth)")

    # Create popularity tiers based on ACTUAL owners
    tiers_actual = {
        'Very Low (10K-20K)': (y_actual >= 10000) & (y_actual < 20000),
        'Low (20K-50K)': (y_actual >= 20000) & (y_actual < 50000),
        'Medium (50K-200K)': (y_actual >= 50000) & (y_actual < 200000),
        'High (200K-1M)': (y_actual >= 200000) & (y_actual < 1000000),
        'Very High (1M+)': (y_actual >= 1000000)
    }

    results = []

    for tier_name, mask in tiers_actual.items():
        if mask.sum() >= 30:
            y_true_tier = y_actual[mask]
            y_pred_tier = y_pred[mask.values]

            spearman_tier, _ = spearmanr(y_true_tier, y_pred_tier)

            # Percentile error
            actual_pct = pd.Series(y_true_tier.values).rank(pct=True) * 100
            pred_pct = pd.Series(y_pred_tier).rank(pct=True) * 100
            mae_pct = np.mean(np.abs(actual_pct - pred_pct))

            results.append({
                'Tier': tier_name,
                'Games': mask.sum(),
                'Spearman': spearman_tier,
                'MAE Percentile': mae_pct,
                'vs Overall': spearman_tier - overall_spearman,
                'Avg Actual': y_true_tier.mean(),
                'Avg Predicted': y_pred_tier.mean()
            })

    df_results = pd.DataFrame(results)

    # Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall", f"{overall_spearman:.4f}")
    with col2:
        avg = df_results['Spearman'].mean()
        st.metric("Avg by Tier", f"{avg:.4f}")
    with col3:
        improvement = avg - overall_spearman
        st.metric("Improvement", f"{improvement:+.4f}")

    st.dataframe(df_results.style.format({
        'Spearman': '{:.4f}',
        'MAE Percentile': '{:.2f}',
        'vs Overall': '{:+.4f}',
        'Avg Actual': '{:,.0f}',
        'Avg Predicted': '{:,.0f}'
    }), use_container_width=True)

    # Interpretation
    if df_results['Spearman'].mean() > overall_spearman + 0.10:
        st.success("✅ Model ranks better within popularity tiers!")
    else:
        st.warning("⚠️ No significant improvement within popularity tiers")


if __name__ == "__main__":
    check_spearman_by_popularity()
