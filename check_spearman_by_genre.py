"""
Check Spearman Correlation by Genre
Quick diagnostic to see if ranking is better within genres
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import streamlit as st


def check_spearman_by_genre():
    """Compute Spearman correlation within each genre"""

    st.title("🎮 Spearman Correlation by Genre")

    if 'models' not in st.session_state:
        st.error("❌ Models not trained yet. Go to 'Analyze New Game' tab first.")
        return

    models = st.session_state.models

    # Get test set data - using correct key names from models wrapper
    y_actual = models.get('y_owners_actual_test')
    X_test = models.get('X_test')

    if y_actual is None or X_test is None:
        st.error("❌ Test data not available. Please retrain models.")
        return

    # Need to regenerate predictions since they're not stored in models dict
    owners_model = models.get('owners_model')
    if owners_model is None:
        st.error("❌ Owners model not available.")
        return

    # Generate predictions
    import numpy as np
    owners_pred_log = owners_model.predict(X_test)
    y_pred = np.expm1(owners_pred_log)  # Back-transform from log

    # Overall Spearman
    overall_spearman, _ = spearmanr(y_actual, y_pred)

    st.info(f"**Overall Spearman Correlation: {overall_spearman:.4f}**")

    # Find genre columns
    genre_cols = [c for c in X_test.columns if c.startswith('genre_')]

    if not genre_cols:
        st.error("No genre columns found in test data")
        return

    st.markdown(f"Found {len(genre_cols)} genres in the dataset")
    st.markdown("---")

    # Analyze each genre
    results = []

    for genre_col in genre_cols:
        genre_name = genre_col.replace('genre_', '').replace('_', ' ').title()

        # Get games in this genre
        genre_mask = X_test[genre_col] == 1
        n_games = genre_mask.sum()

        if n_games >= 30:  # Need at least 30 games for reliable stats
            y_actual_genre = y_actual[genre_mask]
            y_pred_genre = y_pred[genre_mask.values]

            spearman_genre, p_value = spearmanr(y_actual_genre, y_pred_genre)

            # Calculate percentile error within this genre
            actual_pct_genre = pd.Series(y_actual_genre.values).rank(pct=True) * 100
            pred_pct_genre = pd.Series(y_pred_genre).rank(pct=True) * 100
            mae_pct_genre = np.mean(np.abs(actual_pct_genre - pred_pct_genre))

            results.append({
                'Genre': genre_name,
                'Games': n_games,
                'Spearman': spearman_genre,
                'MAE Percentile': mae_pct_genre,
                'vs Overall': spearman_genre - overall_spearman,
                'p-value': p_value
            })

    # Convert to DataFrame and sort
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('Spearman', ascending=False)

    # Summary stats
    st.markdown("## 📊 Summary Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Overall Spearman", f"{overall_spearman:.4f}")

    with col2:
        avg_genre = df_results['Spearman'].mean()
        st.metric("Average Genre Spearman", f"{avg_genre:.4f}")

    with col3:
        better_count = (df_results['Spearman'] > overall_spearman).sum()
        st.metric("Genres Better Than Overall", f"{better_count}/{len(df_results)}")

    with col4:
        best_spearman = df_results['Spearman'].max()
        st.metric("Best Genre Spearman", f"{best_spearman:.4f}")

    # Key insight
    improvement = avg_genre - overall_spearman

    if improvement >= 0.15:
        st.success(f"""
        ✅ **Excellent News!**

        Average genre-specific Spearman ({avg_genre:.4f}) is **{improvement:.4f} higher** than overall ({overall_spearman:.4f}).

        This means the model ranks well WITHIN genres but struggles ACROSS genres.
        **Your percentile system is more reliable than the overall score suggests!**
        """)
    elif improvement >= 0.08:
        st.info(f"""
        ℹ️ **Moderate Improvement**

        Average genre-specific Spearman ({avg_genre:.4f}) is {improvement:.4f} higher than overall ({overall_spearman:.4f}).

        The model performs somewhat better within genres.
        """)
    else:
        st.warning(f"""
        ⚠️ **Limited Improvement**

        Average genre-specific Spearman ({avg_genre:.4f}) is only {improvement:.4f} higher than overall ({overall_spearman:.4f}).

        The model struggles uniformly across genres.
        """)

    st.markdown("---")

    # Results table
    st.markdown("## 📋 Detailed Results by Genre")

    # Color-code the table
    def highlight_spearman(row):
        colors = []
        spearman = row['Spearman']

        for col in row.index:
            if col == 'Spearman':
                if spearman >= 0.70:
                    colors.append('background-color: #d4edda')  # Green
                elif spearman >= 0.60:
                    colors.append('background-color: #d1ecf1')  # Blue
                elif spearman >= 0.50:
                    colors.append('background-color: #fff3cd')  # Yellow
                elif spearman >= overall_spearman:
                    colors.append('background-color: #f8f9fa')  # Light gray
                else:
                    colors.append('background-color: #f8d7da')  # Red
            else:
                colors.append('')

        return colors

    styled_df = df_results.style.apply(highlight_spearman, axis=1).format({
        'Spearman': '{:.4f}',
        'MAE Percentile': '{:.2f}',
        'vs Overall': '{:+.4f}',
        'p-value': '{:.4f}'
    })

    st.dataframe(styled_df, use_container_width=True, height=600)

    # Top performers
    st.markdown("## 🏆 Best Performing Genres (Spearman ≥ 0.60)")

    top_genres = df_results[df_results['Spearman'] >= 0.60]

    if len(top_genres) > 0:
        for idx, row in top_genres.iterrows():
            st.success(f"""
            **{row['Genre']}**
            - Spearman: {row['Spearman']:.4f} ({"Excellent" if row['Spearman'] >= 0.70 else "Good"})
            - Games: {row['Games']:,}
            - MAE Percentile: {row['MAE Percentile']:.2f} points
            - Improvement over overall: {row['vs Overall']:+.4f}
            """)

        st.info(f"""
        **💡 Recommendation:**

        For games in these {len(top_genres)} genres, you can show percentile rankings with **high confidence**.

        Consider adding genre-specific percentiles in your UI:
        - "Your game ranks at the **72nd percentile among {top_genres.iloc[0]['Genre']} games**"
        - Include a confidence indicator: 🟢 High Confidence (Spearman > 0.60)
        """)
    else:
        st.warning("No genres have Spearman ≥ 0.60")

    # Poor performers
    st.markdown("## ⚠️ Worst Performing Genres (Bottom 5)")

    worst_genres = df_results.tail(5)

    for idx, row in worst_genres.iterrows():
        st.error(f"""
        **{row['Genre']}**
        - Spearman: {row['Spearman']:.4f}
        - Games: {row['Games']:,}
        - MAE Percentile: {row['MAE Percentile']:.2f} points
        """)

    # Download results
    st.markdown("---")
    st.markdown("### 💾 Download Results")

    csv = df_results.to_csv(index=False)
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name="spearman_by_genre.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    check_spearman_by_genre()
