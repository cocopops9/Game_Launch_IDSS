"""
Diagnostic script to understand why Spearman correlation is low
Run this after models are trained to analyze ranking issues
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import streamlit as st

def diagnose_ranking_issues():
    """Analyze why ranking accuracy is low"""

    if 'models' not in st.session_state:
        st.error("Train models first!")
        return

    models = st.session_state.models

    y_actual = models['y_test_actual']
    y_pred = models['owners_pred']

    st.title("🔬 Ranking Accuracy Diagnostic")

    # Overall metrics
    spearman, _ = spearmanr(y_actual, y_pred)

    st.metric("Spearman Correlation", f"{spearman:.4f}")

    # 1. Check for systematic bias in different owner ranges
    st.markdown("## 1. Prediction Bias by Owner Range")

    # Split into quartiles
    quartiles = pd.qcut(y_actual, q=4, labels=['Q1 (0-25%)', 'Q2 (25-50%)', 'Q3 (50-75%)', 'Q4 (75-100%)'])

    results = []
    for q in quartiles.unique():
        mask = quartiles == q
        actual_q = y_actual[mask]
        pred_q = y_pred[mask]

        spearman_q, _ = spearmanr(actual_q, pred_q)

        results.append({
            'Quartile': q,
            'Games': mask.sum(),
            'Spearman': spearman_q,
            'Avg Actual': actual_q.mean(),
            'Avg Predicted': pred_q.mean(),
            'Bias': pred_q.mean() - actual_q.mean()
        })

    df_results = pd.DataFrame(results)
    st.dataframe(df_results)

    # 2. Worst ranking errors
    st.markdown("## 2. Worst Ranking Errors")

    actual_ranks = y_actual.rank(pct=True) * 100
    pred_ranks = pd.Series(y_pred).rank(pct=True) * 100

    rank_errors = np.abs(actual_ranks.values - pred_ranks.values)

    st.metric("Average Percentile Error", f"{rank_errors.mean():.2f} points")
    st.metric("Median Percentile Error", f"{np.median(rank_errors):.2f} points")
    st.metric("90th Percentile Error", f"{np.percentile(rank_errors, 90):.2f} points")

    # Show worst cases
    worst_indices = np.argsort(rank_errors)[-10:]

    st.markdown("### Top 10 Worst Ranking Errors")
    worst_cases = pd.DataFrame({
        'Actual Owners': y_actual.iloc[worst_indices].values,
        'Predicted Owners': y_pred[worst_indices],
        'Actual Percentile': actual_ranks.iloc[worst_indices].values,
        'Predicted Percentile': pred_ranks.iloc[worst_indices].values,
        'Error (percentile points)': rank_errors[worst_indices]
    })
    st.dataframe(worst_cases.sort_values('Error (percentile points)', ascending=False))

    # 3. Scatter plot of ranks
    st.markdown("## 3. Actual vs Predicted Ranks (Percentiles)")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(actual_ranks, pred_ranks, alpha=0.3, s=10)
    ax.plot([0, 100], [0, 100], 'r--', label='Perfect ranking')
    ax.set_xlabel('Actual Percentile')
    ax.set_ylabel('Predicted Percentile')
    ax.set_title(f'Ranking Agreement (Spearman = {spearman:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # 4. Error distribution
    st.markdown("## 4. Percentile Error Distribution")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(rank_errors, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(rank_errors.mean(), color='r', linestyle='--', label=f'Mean = {rank_errors.mean():.2f}')
    ax.axvline(np.median(rank_errors), color='g', linestyle='--', label=f'Median = {np.median(rank_errors):.2f}')
    ax.set_xlabel('Percentile Error (absolute)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Ranking Errors')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # 5. Recommendations
    st.markdown("## 5. Recommendations")

    if spearman < 0.5:
        st.error("""
        **🚨 Low Spearman Correlation (<0.5)**

        The model's ranking accuracy is poor. Consider:
        1. **Feature Engineering**: Add more predictive features
        2. **Model Selection**: Try different algorithms (LightGBM, CatBoost)
        3. **Target Transformation**: Experiment with different transformations
        4. **Rank-based Loss**: Train with ranking-specific loss functions
        5. **Reduce Reliance**: Acknowledge percentiles are approximate, not precise
        """)
    elif spearman < 0.6:
        st.warning("""
        **⚠️ Moderate Spearman Correlation (0.5-0.6)**

        Ranking is fair but not great. Suggestions:
        1. Tune hyperparameters focusing on ranking metrics
        2. Add domain-specific features (marketing spend, community size)
        3. Use percentiles as **directional guidance**, not precise values
        """)
    else:
        st.success("✅ Good ranking accuracy!")

    # 6. Adjusted system messaging
    st.markdown("## 6. How to Communicate This to Users")

    st.info(f"""
    **Current Spearman: {spearman:.4f}**

    **Honest Messaging:**
    - "The system provides **approximate** market positioning"
    - "Percentiles should be interpreted with a margin of ±{rank_errors.mean():.0f} points"
    - "Use this for **directional** insights, not precise predictions"
    - "Focus on improvement scenarios rather than exact percentile values"

    **Don't claim:**
    - "Accurate percentile positioning" (only if Spearman > 0.70)
    - "Reliable ranking" (only if Spearman > 0.60)
    """)


if __name__ == "__main__":
    diagnose_ranking_issues()
