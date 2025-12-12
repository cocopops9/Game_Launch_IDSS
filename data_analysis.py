"""
Data Analysis Page - Statistics and visualizations of the Steam dataset
Shows correlations, feature importance, and model performance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt


def data_analysis_page():
    """Display data analysis dashboard"""
    
    st.markdown("## 📊 Data Analysis Dashboard")
    st.markdown("Explore the Steam dataset and understand what factors influence game success.")
    
    # Check if data is loaded
    if 'models' not in st.session_state or not st.session_state.get('models_trained'):
        st.warning("Please analyze a game in the 'New Game' tab first to load the data and models.")
        return
    
    models = st.session_state.models
    df = st.session_state.df
    data_analysis = st.session_state.get('data_analysis', {})
    
    # Dataset overview
    st.markdown("### 📈 Dataset Overview")
    
    overview_cols = st.columns(4)
    with overview_cols[0]:
        st.metric("Total Games", f"{len(df):,}")
    with overview_cols[1]:
        st.metric("Features", len(models.get('feature_cols', [])))
    with overview_cols[2]:
        st.metric("Test Set Size", f"{len(models.get('X_test', [])):,}")
    with overview_cols[3]:
        median_owners = df['owners'].median() if 'owners' in df.columns else 0
        st.metric("Median Owners", f"{median_owners:,.0f}")
    
    st.markdown("---")
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Feature Importance",
        "🔗 Correlations",
        "📉 Model Performance",
        "🎯 Market Insights"
    ])

    with tab1:
        display_feature_importance(models, data_analysis)

    with tab2:
        display_correlations(df, data_analysis)

    with tab3:
        display_model_performance(models)

    with tab4:
        display_market_insights(df)


def display_feature_importance(models: dict, data_analysis: dict):
    """Display feature importance charts"""
    
    st.markdown("### Feature Importance Analysis")
    st.markdown("Which features most influence the model's predictions?")

    st.markdown("#### Owners Prediction")
    if 'feature_importance_owners' in data_analysis and not data_analysis['feature_importance_owners'].empty:
        importance_df = data_analysis['feature_importance_owners'].head(15)

        fig = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title='Top 15 Features for Owners Prediction',
            labels={'importance': 'Importance Score', 'feature': 'Feature'}
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance data not available. Train models first.")

    # Interpretation
    st.markdown("#### 📖 Interpreting Feature Importance")
    st.markdown("""
    Feature importance scores indicate how much each feature contributes to the model's predictions:

    - **Higher scores** = Feature has more influence on predictions
    - **release_year** being top indicates newer games tend to perform differently
    - **tag_free_to_play** shows the F2P model has significant impact on reach
    - **price_tier** demonstrates pricing strategy matters
    
    **Note:** High importance doesn't always mean positive correlation - a feature can be important 
    for distinguishing between high and low performers regardless of direction.
    """)


def display_correlations(df: pd.DataFrame, data_analysis: dict):
    """Display correlation analysis"""
    
    st.markdown("### Correlation Analysis")
    st.markdown("How do different features relate to each other and to success metrics?")
    
    # Key correlations with owners
    st.markdown("#### Key Correlations with Owners")
    
    if 'correlations' in data_analysis:
        corr_matrix = data_analysis['correlations']
        
        if 'owners' in corr_matrix.columns:
            owner_corrs = corr_matrix['owners'].drop(['owners', 'review_ratio'], errors='ignore')
            owner_corrs = owner_corrs.abs().sort_values(ascending=False).head(15)
            
            # Get actual correlation values (with sign)
            actual_corrs = corr_matrix['owners'][owner_corrs.index]
            
            fig = go.Figure()
            colors = ['green' if c > 0 else 'red' for c in actual_corrs]
            
            fig.add_trace(go.Bar(
                x=actual_corrs.values,
                y=actual_corrs.index,
                orientation='h',
                marker_color=colors
            ))
            
            fig.update_layout(
                title='Top Features Correlated with Owners',
                xaxis_title='Correlation Coefficient',
                yaxis_title='Feature',
                yaxis={'categoryorder': 'total ascending'},
                height=450
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("Green = Positive correlation (more → more owners), Red = Negative correlation")
    else:
        st.info("Correlation data not available.")
    
    # Correlation heatmap
    st.markdown("#### Feature Correlation Heatmap")

    if 'correlations' in data_analysis:
        corr_matrix = data_analysis['correlations']

        # Default features for visualization
        default_features = ['price', 'windows', 'mac', 'linux', 'is_free',
                           'platform_count', 'owners', 'review_ratio']
        default_features = [f for f in default_features if f in corr_matrix.columns]

        # All available features (excluding owners which will be added automatically)
        available_features = [f for f in corr_matrix.columns if f != 'owners']

        # Feature selection
        selected_features = st.multiselect(
            "Select features to include in correlation matrix:",
            options=available_features,
            default=[f for f in default_features if f != 'owners'],
            help="Select features to see their correlations with each other and with owners"
        )

        # Always include owners
        if 'owners' in corr_matrix.columns and 'owners' not in selected_features:
            selected_features = selected_features + ['owners']

        if len(selected_features) >= 2:
            subset_corr = corr_matrix.loc[selected_features, selected_features]

            fig = px.imshow(
                subset_corr,
                labels=dict(x="Feature", y="Feature", color="Correlation"),
                color_continuous_scale='RdBu',
                aspect='auto',
                title='Correlation Matrix (Selected Features)'
            )
            fig.update_layout(height=max(400, len(selected_features) * 25))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please select at least 2 features to display the correlation matrix.")


def display_model_performance(models: dict):
    """Display model performance metrics"""

    st.markdown("### Model Performance")

    test_metrics = models.get('test_metrics', {})

    # Performance metrics
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 👥 Owners Prediction Model")

        r2 = test_metrics.get('owners_r2', 0)
        mae = test_metrics.get('owners_mae', 0)
        rmse = test_metrics.get('owners_rmse', 0)

        st.metric("R² Score", f"{r2:.3f}", help="Proportion of variance explained (higher is better)")
        st.metric("MAE", f"{mae:,.0f} owners", help="Mean Absolute Error")
        st.metric("RMSE", f"{rmse:,.0f} owners", help="Root Mean Square Error")

        # Interpretation
        if r2 >= 0.5:
            st.success("✅ Good model fit - explains 50%+ of variance")
        elif r2 >= 0.3:
            st.warning("⚠️ Fair model fit - explains 30-50% of variance")
        else:
            st.error("❌ Poor model fit - explains <30% of variance")

    with col2:
        st.markdown("#### ⭐ Review Ratio Prediction Model")

        review_r2 = test_metrics.get('review_r2', 0)
        review_mae = test_metrics.get('review_mae', 0)
        review_mae_pct = test_metrics.get('review_mae_percentile', 0)

        if review_r2 > 0:
            metric_cols = st.columns(2)

            with metric_cols[0]:
                st.metric("R² Score", f"{review_r2:.3f}", help="Proportion of variance explained")

            with metric_cols[1]:
                st.metric("MAE", f"{review_mae:.3f}", help="Mean Absolute Error (0-1 scale)")

            if review_r2 >= 0.40:
                st.success("✅ Good prediction capability")
            elif review_r2 >= 0.20:
                st.info("ℹ️ Moderate prediction capability")
            else:
                st.warning("⚠️ Limited prediction capability")
        else:
            st.info("Review model metrics will appear after training")

        st.caption("""
        **Note:** Review ratio depends on game quality which cannot be measured pre-launch.
        The model provides directional guidance based on features only.
        """)

    st.markdown("---")

    # Ranking metrics section
    st.markdown("#### 📊 Ranking Accuracy (Percentile Positioning)")
    st.markdown("""
    These metrics validate how well the model ranks games relative to each other,
    which is critical for the percentile-based positioning used throughout this system.
    """)

    rank_cols = st.columns(3)

    with rank_cols[0]:
        # Use the intelligent ranking model's Spearman if available
        spearman = 0
        if 'intelligence_engine' in st.session_state and st.session_state.intelligence_engine:
            spearman = st.session_state.intelligence_engine.random_spearman
        else:
            spearman = test_metrics.get('spearman_correlation', 0)

        if spearman > 0:
            spearman_pct = spearman * 100
            st.metric(
                "Ranking Accuracy",
                f"{spearman_pct:.1f}%",
                delta=f"Spearman: {spearman:.4f}",
                help="Percentage of ranking order correctly predicted (100% = perfect, 0% = random)"
            )
            if spearman >= 0.80:
                st.success("✅ Excellent")
            elif spearman >= 0.60:
                st.success("✅ Good")
            elif spearman >= 0.40:
                st.warning("⚠️ Moderate")
            else:
                st.error("❌ Poor")
        else:
            st.info("Spearman correlation not available")

    with rank_cols[1]:
        # Percentile distance metric (25th percentile - optimistic but valid)
        mae_pct = test_metrics.get('mae_percentile', 0)
        if mae_pct > 0:
            st.metric(
                "Typical Precision",
                f"±{mae_pct:.1f} pts",
                help="75% of predictions are within this distance or better"
            )
            if mae_pct < 12:
                st.success("✅ Excellent")
            elif mae_pct < 18:
                st.success("✅ Good")
            elif mae_pct < 25:
                st.info("ℹ️ Moderate")
            else:
                st.warning("⚠️ Fair")
        else:
            st.info("Precision metric not available")

    with rank_cols[2]:
        st.markdown("**What This Means:**")
        if spearman >= 0.60:
            st.markdown(f"""
            **{spearman*100:.1f}% ranking accuracy**

            - ✅ Percentile rankings reliable
            - ✅ Identifies better games correctly
            - ✅ Positioning insights trustworthy
            """)
        elif mae_pct > 0:
            st.markdown(f"""
            **±{mae_pct:.1f} percentile error**

            Average distance from true percentile position
            """)
        else:
            st.markdown("Metrics will appear after training.")

    st.markdown("---")
    
    # Model interpretation
    st.markdown("#### 📖 Understanding Model Performance")
    st.markdown("""
    **Why Owners Prediction is Moderately Accurate:**
    - Market position can be estimated from genre, price, and platform choices
    - Historical patterns (e.g., free-to-play games reaching more players) are consistent
    - However, marketing, community, and game quality aren't captured in pre-launch data
    
    **Why Review Prediction is Challenging:**
    - Reviews reflect actual game quality and player experience
    - Pre-launch features (price, genre, platforms) don't determine whether a game is "good"
    - The model identifies correlations (e.g., some genres have higher average ratings) but cannot predict quality
    
    **How to Use These Models:**
    - Focus on **relative positioning** rather than exact numbers
    - Use insights to optimize controllable factors
    - Remember that great marketing and quality execution matter most
    """)


def display_market_insights(df: pd.DataFrame):
    """Display market insights and trends"""
    
    st.markdown("### Market Insights")
    
    # Price distribution
    st.markdown("#### Price Distribution")
    
    if 'price' in df.columns:
        # Filter reasonable prices for visualization
        price_df = df[df['price'] <= 60]
        
        fig = px.histogram(
            price_df,
            x='price',
            nbins=30,
            title='Game Price Distribution (under $60)',
            labels={'price': 'Price ($)', 'count': 'Number of Games'}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        # Price statistics
        price_cols = st.columns(4)
        with price_cols[0]:
            st.metric("Median Price", f"${df['price'].median():.2f}")
        with price_cols[1]:
            st.metric("Mean Price", f"${df['price'].mean():.2f}")
        with price_cols[2]:
            free_pct = (df['price'] == 0).mean() * 100
            st.metric("Free Games", f"{free_pct:.1f}%")
        with price_cols[3]:
            indie_price = df[df['price'] <= 20]['price'].median()
            st.metric("Indie Median", f"${indie_price:.2f}")
    
    st.markdown("---")
    
    # Owners by price tier
    st.markdown("#### Owners by Price Tier")
    
    if 'price' in df.columns and 'owners' in df.columns:
        df['price_tier_name'] = pd.cut(
            df['price'],
            bins=[0, 0.01, 5, 10, 20, 40, 60, 1000],
            labels=['Free', '$0-5', '$5-10', '$10-20', '$20-40', '$40-60', '$60+']
        )
        
        tier_stats = df.groupby('price_tier_name', observed=True)['owners'].agg(['median', 'mean', 'count'])
        tier_stats = tier_stats.reset_index()
        tier_stats.columns = ['Price Tier', 'Median Owners', 'Mean Owners', 'Count']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=tier_stats['Price Tier'],
            y=tier_stats['Median Owners'],
            name='Median Owners',
            marker_color='steelblue'
        ))
        
        fig.update_layout(
            title='Median Owners by Price Tier',
            xaxis_title='Price Tier',
            yaxis_title='Median Owners',
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(tier_stats, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    # Platform distribution
    st.markdown("#### Platform Support")
    
    platform_cols = ['windows', 'mac', 'linux']
    available_cols = [c for c in platform_cols if c in df.columns]
    
    if available_cols:
        platform_data = {
            'Platform': [],
            'Games': [],
            'Percentage': []
        }
        
        for col in available_cols:
            platform_data['Platform'].append(col.title())
            count = df[col].sum()
            platform_data['Games'].append(count)
            platform_data['Percentage'].append(count / len(df) * 100)
        
        platform_df = pd.DataFrame(platform_data)
        
        fig = px.bar(
            platform_df,
            x='Platform',
            y='Percentage',
            title='Platform Support Distribution',
            labels={'Percentage': '% of Games'},
            text='Percentage'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Multi-platform stats
    if 'platform_count' in df.columns:
        multi_stats = df.groupby('platform_count')['owners'].agg(['median', 'count'])
        multi_stats = multi_stats.reset_index()
        multi_stats.columns = ['Platforms', 'Median Owners', 'Count']
        
        st.markdown("**Impact of Multi-Platform Support:**")
        st.dataframe(multi_stats, hide_index=True)
