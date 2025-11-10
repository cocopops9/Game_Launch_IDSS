"""
Data Analysis page - visualizations and insights
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from models import load_steam_data, train_models


def data_analysis_page():
    """Data Analysis page - visualizations and insights"""
    st.markdown('<h2 class="sub-header">📈 Data Analysis & Insights</h2>', unsafe_allow_html=True)
    
    # Load data and models
    df, loader = load_steam_data()
    
    if not st.session_state.models_trained:
        with st.spinner("Training models and analyzing data..."):
            st.session_state.models = train_models(df, loader)
            st.session_state.models_trained = True
    
    models = st.session_state.models
    data_analysis = st.session_state.data_analysis
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Improvement Analysis", "Feature Importance",
                                             "Correlation Analysis", "Trend Analysis", "Model Performance"])
    
    with tab1:
        st.subheader("📈 What Improves Game Success?")
        st.info("💡 This analysis shows how each feature impacts owners and review ratios, based on all games in the dataset.")

        if 'feature_impacts' in models:
            feature_impacts = models['feature_impacts']

            # Overview metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Features Analyzed", len(feature_impacts))
            with col2:
                positive_owners = sum(1 for v in feature_impacts.values() if v.get('owners_improvement_pct', 0) > 0)
                st.metric("Features Improving Owners", positive_owners)
            with col3:
                positive_reviews = sum(1 for v in feature_impacts.values() if v.get('reviews_improvement_pct', 0) > 0)
                st.metric("Features Improving Reviews", positive_reviews)

            st.markdown("---")

            # Two columns for owners and reviews
            col_own, col_rev = st.columns(2)

            with col_own:
                st.markdown("### 🎮 Features That Improve Owners")

                # Sort by owners improvement
                sorted_by_owners = sorted(
                    feature_impacts.items(),
                    key=lambda x: x[1].get('owners_improvement_pct', 0),
                    reverse=True
                )

                # Show top 15
                top_15_owners = [(k, v) for k, v in sorted_by_owners if v.get('owners_improvement_pct', 0) > 0][:15]

                if top_15_owners:
                    # Create chart
                    chart_data = pd.DataFrame([
                        {
                            'Feature': feat.replace('tag_', '').replace('_', ' ').title(),
                            'Improvement %': impact['owners_improvement_pct'],
                            'Type': impact['type']
                        }
                        for feat, impact in top_15_owners
                    ])

                    fig = px.bar(
                        chart_data,
                        x='Improvement %',
                        y='Feature',
                        orientation='h',
                        title='Top 15 Features for Increasing Owners',
                        color='Improvement %',
                        color_continuous_scale='Blues',
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Detailed breakdown
                    with st.expander("📊 View Detailed Breakdown"):
                        for feat, impact in top_15_owners:
                            feat_name = feat.replace('tag_', '').replace('_', ' ').title()
                            pct = impact['owners_improvement_pct']

                            if impact['type'] == 'binary':
                                with_val = impact['owners_with']
                                without_val = impact['owners_without']
                                sample_with = impact['sample_with']
                                sample_without = impact['sample_without']
                                st.markdown(f"""
                                **{feat_name}**: +{pct}%
                                - With feature: {with_val:,} owners ({sample_with} games)
                                - Without feature: {without_val:,} owners ({sample_without} games)
                                """)
                            else:
                                top_val = impact['owners_top25']
                                bottom_val = impact['owners_bottom25']
                                q75 = impact.get('q75_value', 0)
                                q25 = impact.get('q25_value', 0)
                                st.markdown(f"""
                                **{feat_name}**: +{pct}%
                                - Top 25% (≥{q75}): {top_val:,} owners
                                - Bottom 25% (≤{q25}): {bottom_val:,} owners
                                """)

            with col_rev:
                st.markdown("### ⭐ Features That Improve Review Ratio")

                # Sort by review improvement
                sorted_by_reviews = sorted(
                    feature_impacts.items(),
                    key=lambda x: x[1].get('reviews_improvement_pct', 0),
                    reverse=True
                )

                # Show top 15
                top_15_reviews = [(k, v) for k, v in sorted_by_reviews if v.get('reviews_improvement_pct', 0) > 0][:15]

                if top_15_reviews:
                    # Create chart
                    chart_data = pd.DataFrame([
                        {
                            'Feature': feat.replace('tag_', '').replace('_', ' ').title(),
                            'Improvement %': impact['reviews_improvement_pct'],
                            'Type': impact['type']
                        }
                        for feat, impact in top_15_reviews
                    ])

                    fig = px.bar(
                        chart_data,
                        x='Improvement %',
                        y='Feature',
                        orientation='h',
                        title='Top 15 Features for Improving Review Ratio',
                        color='Improvement %',
                        color_continuous_scale='Greens',
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Detailed breakdown
                    with st.expander("📊 View Detailed Breakdown"):
                        for feat, impact in top_15_reviews:
                            feat_name = feat.replace('tag_', '').replace('_', ' ').title()
                            pct = impact['reviews_improvement_pct']

                            if impact['type'] == 'binary':
                                with_val = impact['reviews_with']
                                without_val = impact['reviews_without']
                                sample_with = impact['sample_with']
                                sample_without = impact['sample_without']
                                st.markdown(f"""
                                **{feat_name}**: +{pct}%
                                - With feature: {with_val:.1%} ratio ({sample_with} games)
                                - Without feature: {without_val:.1%} ratio ({sample_without} games)
                                """)
                            else:
                                top_val = impact['reviews_top25']
                                bottom_val = impact['reviews_bottom25']
                                q75 = impact.get('q75_value', 0)
                                q25 = impact.get('q25_value', 0)
                                st.markdown(f"""
                                **{feat_name}**: +{pct}%
                                - Top 25% (≥{q75}): {top_val:.1%} ratio
                                - Bottom 25% (≤{q25}): {bottom_val:.1%} ratio
                                """)
        else:
            st.warning("Feature impact data not available. Please retrain the models.")

    with tab2:
        st.subheader("🎯 Feature Importance")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**For Predicting Owners:**")
            if 'feature_importance_owners' in data_analysis:
                fig_owners = px.bar(
                    data_analysis['feature_importance_owners'].head(10),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Top 10 Features for Owner Prediction",
                    color='importance',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_owners, use_container_width=True)

        with col2:
            st.write("**For Predicting Review Ratio:**")
            if 'feature_importance_reviews' in data_analysis:
                fig_reviews = px.bar(
                    data_analysis['feature_importance_reviews'].head(10),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Top 10 Features for Review Prediction",
                    color='importance',
                    color_continuous_scale='Plasma'
                )
                st.plotly_chart(fig_reviews, use_container_width=True)

    with tab3:
        st.subheader("🔗 Correlation Matrix")

        if 'correlations' in data_analysis:
            corr_matrix = data_analysis['correlations']

            # Select important features
            important_features = ['price', 'owners', 'review_ratio', 'windows', 'mac', 'linux']
            available_features = [f for f in important_features if f in corr_matrix.columns]

            # Add some tag features
            tag_features = [col for col in corr_matrix.columns if col.startswith('tag_')][:5]
            available_features.extend(tag_features)

            # Create subset correlation matrix
            subset_corr = corr_matrix.loc[available_features, available_features]

            # Create heatmap
            fig = px.imshow(
                subset_corr,
                labels=dict(x="Features", y="Features", color="Correlation"),
                x=available_features,
                y=available_features,
                color_continuous_scale="RdBu",
                aspect="auto",
                title="Feature Correlation Heatmap (Actual Steam Data)"
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

            # Key insights
            st.markdown("**📊 Key Data Insights:**")

            price_owners_corr = corr_matrix.loc['price', 'owners'] if 'price' in corr_matrix.index else 0
            windows_owners_corr = corr_matrix.loc['windows', 'owners'] if 'windows' in corr_matrix.index else 0

            insights = [
                f"• **Price Impact**: Correlation with owners is {price_owners_corr:.3f}",
                f"• **Windows Platform**: Correlation with owners is {windows_owners_corr:.3f}",
                f"• **Dataset Size**: Analysis based on {len(df)} games",
            ]

            for insight in insights:
                st.markdown(insight)

    with tab4:
        st.subheader("📊 Trend Analysis from Steam Data")
        
        # Price distribution analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Price vs Owners
            fig_price_owners = px.scatter(
                df.sample(min(500, len(df))),
                x='price',
                y='owners',
                title="Price vs Owners (Sample of 500 games)",
                opacity=0.6
            )
            fig_price_owners.update_layout(
                xaxis_title="Price ($)",
                yaxis_title="Number of Owners",
                yaxis_type="log"
            )
            st.plotly_chart(fig_price_owners, use_container_width=True)
        
        with col2:
            # Release month distribution
            monthly_stats = df.groupby('release_month').agg({
                'owners': 'median',
                'review_ratio': 'mean',
                'name': 'count'
            }).reset_index()
            monthly_stats.columns = ['Month', 'Median Owners', 'Avg Review Ratio', 'Game Count']
            
            fig_monthly = px.bar(
                monthly_stats,
                x='Month',
                y='Median Owners',
                title="Median Owners by Release Month",
                color='Avg Review Ratio',
                color_continuous_scale='RdYlGn'
            )
            fig_monthly.update_xaxes(tickmode='array', 
                                    tickvals=list(range(1, 13)),
                                    ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            st.plotly_chart(fig_monthly, use_container_width=True)
        
        # Platform analysis
        st.subheader("Platform Distribution Analysis")

        # Create platform combination categories
        platform_combinations = []

        # Define all possible platform combinations
        df_platforms_temp = df.copy()
        df_platforms_temp['windows'] = df_platforms_temp.get('windows', 0).fillna(0)
        df_platforms_temp['mac'] = df_platforms_temp.get('mac', 0).fillna(0)
        df_platforms_temp['linux'] = df_platforms_temp.get('linux', 0).fillna(0)

        # Windows only
        mask = (df_platforms_temp['windows'] == 1) & (df_platforms_temp['mac'] == 0) & (df_platforms_temp['linux'] == 0)
        count = mask.sum()
        if count > 0:
            platform_combinations.append({
                'Platform': 'Windows Only',
                'Games': count,
                'Avg Owners': df_platforms_temp[mask]['owners'].mean(),
                'Avg Review': df_platforms_temp[mask]['review_ratio'].mean() if 'review_ratio' in df_platforms_temp.columns else 0
            })

        # Mac only
        mask = (df_platforms_temp['windows'] == 0) & (df_platforms_temp['mac'] == 1) & (df_platforms_temp['linux'] == 0)
        count = mask.sum()
        if count > 0:
            platform_combinations.append({
                'Platform': 'Mac Only',
                'Games': count,
                'Avg Owners': df_platforms_temp[mask]['owners'].mean(),
                'Avg Review': df_platforms_temp[mask]['review_ratio'].mean() if 'review_ratio' in df_platforms_temp.columns else 0
            })

        # Linux only
        mask = (df_platforms_temp['windows'] == 0) & (df_platforms_temp['mac'] == 0) & (df_platforms_temp['linux'] == 1)
        count = mask.sum()
        if count > 0:
            platform_combinations.append({
                'Platform': 'Linux Only',
                'Games': count,
                'Avg Owners': df_platforms_temp[mask]['owners'].mean(),
                'Avg Review': df_platforms_temp[mask]['review_ratio'].mean() if 'review_ratio' in df_platforms_temp.columns else 0
            })

        # Windows + Mac
        mask = (df_platforms_temp['windows'] == 1) & (df_platforms_temp['mac'] == 1) & (df_platforms_temp['linux'] == 0)
        count = mask.sum()
        if count > 0:
            platform_combinations.append({
                'Platform': 'Windows + Mac',
                'Games': count,
                'Avg Owners': df_platforms_temp[mask]['owners'].mean(),
                'Avg Review': df_platforms_temp[mask]['review_ratio'].mean() if 'review_ratio' in df_platforms_temp.columns else 0
            })

        # Windows + Linux
        mask = (df_platforms_temp['windows'] == 1) & (df_platforms_temp['mac'] == 0) & (df_platforms_temp['linux'] == 1)
        count = mask.sum()
        if count > 0:
            platform_combinations.append({
                'Platform': 'Windows + Linux',
                'Games': count,
                'Avg Owners': df_platforms_temp[mask]['owners'].mean(),
                'Avg Review': df_platforms_temp[mask]['review_ratio'].mean() if 'review_ratio' in df_platforms_temp.columns else 0
            })

        # Mac + Linux
        mask = (df_platforms_temp['windows'] == 0) & (df_platforms_temp['mac'] == 1) & (df_platforms_temp['linux'] == 1)
        count = mask.sum()
        if count > 0:
            platform_combinations.append({
                'Platform': 'Mac + Linux',
                'Games': count,
                'Avg Owners': df_platforms_temp[mask]['owners'].mean(),
                'Avg Review': df_platforms_temp[mask]['review_ratio'].mean() if 'review_ratio' in df_platforms_temp.columns else 0
            })

        # Windows + Mac + Linux (All platforms)
        mask = (df_platforms_temp['windows'] == 1) & (df_platforms_temp['mac'] == 1) & (df_platforms_temp['linux'] == 1)
        count = mask.sum()
        if count > 0:
            platform_combinations.append({
                'Platform': 'All Platforms',
                'Games': count,
                'Avg Owners': df_platforms_temp[mask]['owners'].mean(),
                'Avg Review': df_platforms_temp[mask]['review_ratio'].mean() if 'review_ratio' in df_platforms_temp.columns else 0
            })

        if platform_combinations:
            df_platforms = pd.DataFrame(platform_combinations)

            col1, col2 = st.columns(2)

            with col1:
                # Bar chart instead of pie chart for platform combinations
                fig_platform_games = px.bar(
                    df_platforms,
                    x='Platform',
                    y='Games',
                    title="Games by Platform Combination",
                    text='Games',
                    color='Games',
                    color_continuous_scale='Blues'
                )
                fig_platform_games.update_traces(texttemplate='%{text}', textposition='outside')
                fig_platform_games.update_layout(
                    xaxis_title="Platform Combination",
                    yaxis_title="Number of Games",
                    showlegend=False,
                    xaxis={'categoryorder':'total descending'}
                )
                st.plotly_chart(fig_platform_games, use_container_width=True)
            
            with col2:
                fig_platform_performance = px.bar(
                    df_platforms,
                    x='Platform',
                    y='Avg Owners',
                    title="Average Owners by Platform Combination",
                    color='Avg Review',
                    color_continuous_scale='Viridis',
                    labels={'Avg Owners': 'Average Owners', 'Avg Review': 'Avg Review Ratio'}
                )
                fig_platform_performance.update_layout(
                    xaxis_title="Platform Combination",
                    yaxis_title="Average Number of Owners",
                    xaxis={'categoryorder':'total descending'}
                )
                st.plotly_chart(fig_platform_performance, use_container_width=True)

    with tab5:
        st.subheader("🎯 Model Performance Metrics")

        # Calculate metrics
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

        # Make predictions on test set
        X_test_df = pd.DataFrame(models['X_test'], columns=models['feature_cols'])
        # Handle NaN values
        X_test_df = X_test_df.fillna(0)

        owners_pred_log = models['owners_model'].predict(X_test_df)
        reviews_pred = models['review_model'].predict(X_test_df)

        # Convert predictions back from log space
        if models.get('uses_log_transform', False):
            owners_pred = np.expm1(owners_pred_log)  # Inverse of log1p
            y_owners_actual = models['y_owners_actual_test']  # Use original scale for metrics
            y_owners_log = models['y_owners_test']  # Log scale for R²
        else:
            owners_pred = owners_pred_log
            y_owners_actual = models['y_owners_test']
            y_owners_log = y_owners_actual

        y_reviews_actual = models['y_reviews_test']

        # Owners metrics (calculated on original scale)
        mae_owners = mean_absolute_error(y_owners_actual, owners_pred)
        rmse_owners = np.sqrt(mean_squared_error(y_owners_actual, owners_pred))

        # R² calculated on log scale for better interpretation
        r2_owners_log = r2_score(y_owners_log, owners_pred_log)

        # Normalized metrics for owners
        mean_owners = np.mean(y_owners_actual)
        median_owners = np.median(y_owners_actual)

        # SMAPE: Symmetric Mean Absolute Percentage Error (better for wide ranges)
        smape_owners = np.mean(2 * np.abs(owners_pred - y_owners_actual) / (np.abs(owners_pred) + np.abs(y_owners_actual) + 1)) * 100

        # Median APE (more robust to outliers than MAPE)
        median_ape_owners = np.median(np.abs((y_owners_actual - owners_pred) / np.maximum(y_owners_actual, 1))) * 100

        # Percentage within ranges (practical metrics)
        within_20pct = np.mean(np.abs(owners_pred - y_owners_actual) / y_owners_actual < 0.20) * 100
        within_50pct = np.mean(np.abs(owners_pred - y_owners_actual) / y_owners_actual < 0.50) * 100

        mae_pct_owners = (mae_owners / mean_owners) * 100
        rmse_pct_owners = (rmse_owners / mean_owners) * 100

        # Review ratio metrics
        r2_reviews = r2_score(y_reviews_actual, reviews_pred)
        mae_reviews = mean_absolute_error(y_reviews_actual, reviews_pred)
        rmse_reviews = np.sqrt(mean_squared_error(y_reviews_actual, reviews_pred))

        # Normalized metrics for reviews
        mape_reviews = np.mean(np.abs((y_reviews_actual - reviews_pred) / np.maximum(y_reviews_actual, 0.01))) * 100

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Owners Prediction Model (GradientBoosting with Log Transform)**")
            st.metric("R² Score (log)", f"{r2_owners_log:.3f}", help="R² on log-transformed scale (better for exponential data)")
            st.metric("SMAPE", f"{smape_owners:.1f}%", help="Symmetric MAPE - handles wide ranges better than MAPE")
            st.metric("Median APE", f"{median_ape_owners:.1f}%", help="Median Absolute Percentage Error - robust to outliers")
            st.metric("Within ±20%", f"{within_20pct:.1f}%", help="Percentage of predictions within 20% of actual")
            st.metric("Within ±50%", f"{within_50pct:.1f}%", help="Percentage of predictions within 50% of actual")

            # Show raw metrics in expander
            with st.expander("📊 Additional Metrics"):
                st.write(f"**MAE:** {mae_owners:,.0f} owners ({mae_pct_owners:.1f}% of mean)")
                st.write(f"**RMSE:** {rmse_owners:,.0f} owners ({rmse_pct_owners:.1f}% of mean)")
                st.write(f"**Mean Actual:** {mean_owners:,.0f} owners")
                st.write(f"**Median Actual:** {median_owners:,.0f} owners")
                st.info("💡 Log transformation enabled! Predicting log(owners) handles the 10K-200M range much better.")

        with col2:
            st.write("**Review Ratio Model (LightGBM)**")
            st.metric("R² Score", f"{r2_reviews:.3f}", help="Coefficient of determination (0=worst, 1=perfect)")
            st.metric("MAPE", f"{mape_reviews:.1f}%", help="Mean Absolute Percentage Error - average prediction error as %")
            st.metric("MAE", f"{mae_reviews:.3f}", help="Mean Absolute Error (review ratio is 0-1 scale)")
            st.metric("RMSE", f"{rmse_reviews:.3f}", help="Root Mean Squared Error (review ratio is 0-1 scale)")

            # Show interpretation
            with st.expander("📊 Interpretation"):
                st.write(f"**Average Actual Review Ratio:** {np.mean(y_reviews_actual):.3f}")
                st.write(f"**Std Dev:** {np.std(y_reviews_actual):.3f}")
                if r2_reviews < 0:
                    st.warning("⚠️ Negative R² indicates the model performs worse than simply predicting the mean.")
                    st.write("This suggests review ratio is hard to predict from the available features.")
        
        # Scatter plots
        st.subheader("Prediction vs Actual Values")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_owners_scatter = px.scatter(
                x=models['y_owners_test'],
                y=owners_pred,
                title="Owners: Predicted vs Actual",
                labels={'x': 'Actual Owners', 'y': 'Predicted Owners'},
                opacity=0.6
            )
            # Add perfect prediction line
            max_val = max(models['y_owners_test'].max(), owners_pred.max())
            fig_owners_scatter.add_trace(go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            st.plotly_chart(fig_owners_scatter, use_container_width=True)
        
        with col2:
            fig_reviews_scatter = px.scatter(
                x=models['y_reviews_test'],
                y=reviews_pred,
                title="Review Ratio: Predicted vs Actual",
                labels={'x': 'Actual Review Ratio', 'y': 'Predicted Review Ratio'},
                opacity=0.6,
                color_discrete_sequence=['green']
            )
            # Add perfect prediction line
            fig_reviews_scatter.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            st.plotly_chart(fig_reviews_scatter, use_container_width=True)

# Main app
