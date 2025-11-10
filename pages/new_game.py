"""
New Game prediction page
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st

from models import load_steam_data, train_models
from recommendations import generate_data_driven_recommendations
from persistence import save_game_configuration, SAVED_GAMES_FILE


def new_game_page():
    """New Game prediction page"""
    st.markdown('<h2 class="sub-header">🎮 New Game Prediction</h2>', unsafe_allow_html=True)

    # Debug info - show saved games status
    with st.expander("🔍 Debug: Saved Games Status"):
        st.write(f"**Games in memory:** {len(st.session_state.saved_games)} games, {len(st.session_state.configurations)} total configurations")
        if st.session_state.saved_games:
            for game_name, configs in st.session_state.saved_games.items():
                st.write(f"  - {game_name}: {len(configs)} configuration(s)")

        if os.path.exists(SAVED_GAMES_FILE):
            file_size = os.path.getsize(SAVED_GAMES_FILE)
            st.write(f"**File on disk:** `{os.path.abspath(SAVED_GAMES_FILE)}` ({file_size} bytes)")
            with open(SAVED_GAMES_FILE, 'r') as f:
                data = json.load(f)
                st.write(f"**In file:** {len(data.get('saved_games', {}))} games, {len(data.get('configurations', []))} configurations")
        else:
            st.write(f"**File on disk:** Not found at `{os.path.abspath(SAVED_GAMES_FILE)}`")

    # Load data and models
    df, loader = load_steam_data()
    
    if not st.session_state.models_trained:
        with st.spinner("Training prediction models on Steam data..."):
            st.session_state.models = train_models(df, loader)
            st.session_state.models_trained = True
    
    models = st.session_state.models
    
    # Create form layout
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("📝 Basic Information")
        game_name = st.text_input("Game Name", value="My New Game", key="game_name_input")
        price = st.slider("Price ($)", 0.0, 59.99, 19.99, 0.01, key="price_slider")
        is_free = st.checkbox("Free to Play", value=price == 0, key="free_checkbox")
        if is_free:
            price = 0.0
        required_age = st.selectbox("Required Age", [0, 12, 16, 18], key="age_select")
    
    with col2:
        st.subheader("🖥️ Platforms")
        windows = st.checkbox("Windows", value=True, key="windows_check")
        mac = st.checkbox("Mac", value=False, key="mac_check")
        linux = st.checkbox("Linux", value=False, key="linux_check")
        
        st.subheader("📅 Release")
        release_month = st.slider("Release Month", 1, 12, 6, key="month_slider")
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        st.write(f"Selected: {month_names[release_month - 1]}")
    
    with col3:
        st.subheader("🏷️ Tags/Genres")
        # Get available tags from the data
        available_tags = [col.replace('tag_', '') for col in df.columns if col.startswith('tag_')][:10]
        
        tag_values = {}
        for tag in available_tags:
            tag_values[f'tag_{tag}'] = st.checkbox(tag.replace('_', ' ').title(), 
                                                   key=f"tag_{tag}_check")
    
    # Prepare input features
    input_features = {
        'price': price,
        'release_month': release_month,
        'windows': int(windows),
        'mac': int(mac),
        'linux': int(linux),
        'is_free': int(is_free),
        'required_age': required_age
    }
    
    # Add tag values
    for tag_col, value in tag_values.items():
        input_features[tag_col] = int(value)
    
    # Ensure all model features are present
    for col in models['feature_cols']:
        if col not in input_features:
            input_features[col] = 0
    
    # Prediction button
    col1, col2 = st.columns([1, 3])

    with col1:
        predict_button = st.button("🔮 Predict", type="primary", use_container_width=True)

    # Handle predictions
    if predict_button:
        # Prepare data for prediction
        X_pred = pd.DataFrame([input_features])[models['feature_cols']]

        # Make predictions
        owners_pred_log = models['owners_model'].predict(X_pred)[0]
        review_pred = models['review_model'].predict(X_pred)[0]

        # Convert from log space back to original scale
        if models.get('uses_log_transform', False):
            owners_pred = np.expm1(owners_pred_log)  # Inverse of log1p
        else:
            owners_pred = owners_pred_log

        # Ensure predictions are reasonable
        owners_pred = max(100, owners_pred)
        review_pred = np.clip(review_pred, 0.1, 0.99)

        # Calculate prediction ranges (±40% for owners based on model uncertainty)
        owners_lower = int(owners_pred * 0.6)
        owners_upper = int(owners_pred * 1.4)

        # Format owners as range string
        def format_owners_range(lower, upper):
            """Format owners count into readable range"""
            if upper < 1000:
                return f"{lower:,} - {upper:,}"
            elif upper < 1000000:
                return f"{lower//1000:,}K - {upper//1000:,}K"
            else:
                return f"{lower//1000000:.1f}M - {upper//1000000:.1f}M"

        owners_range_str = format_owners_range(owners_lower, owners_upper)

        # Store predictions
        st.session_state.current_predictions = {
            'game_name': game_name,
            'owners': owners_pred,
            'owners_lower': owners_lower,
            'owners_upper': owners_upper,
            'owners_range_str': owners_range_str,
            'review_ratio': review_pred,
            'features': input_features
        }

        # Set flag to show predictions
        st.session_state.show_predictions = True

    # Display results (outside the button click, so they persist)
    if st.session_state.show_predictions and st.session_state.current_predictions:
        # Extract predictions from session state
        preds = st.session_state.current_predictions
        owners_pred = preds['owners']
        review_pred = preds['review_ratio']

        st.markdown("---")
        st.markdown('<h3 style="color: #4ECDC4;">📊 Prediction Results</h3>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="Predicted Owners",
                value=f"{int(owners_pred):,}",
                delta=f"±{int(owners_pred * 0.15):,}"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="Predicted Review Ratio",
                value=f"{review_pred:.1%}",
                delta=f"±{review_pred * 0.08:.1%}"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # ============================================================================
        # IMPROVEMENT INSIGHTS - Show what improves outcomes
        # ============================================================================
        if 'feature_impacts' in models:
            st.markdown("---")
            st.markdown('<h3 style="color: #4ECDC4;">📈 What Improves Your Game\'s Success?</h3>', unsafe_allow_html=True)
            st.info("💡 These insights show how each feature impacts success, based on analysis of all games in the dataset.")

            feature_impacts = models['feature_impacts']

            # Split into two columns for owners and reviews
            col_own, col_rev = st.columns(2)

            with col_own:
                st.markdown("**🎮 Features That Improve Owners**")

                # Get top 10 features sorted by owners improvement
                sorted_by_owners = sorted(
                    [(k, v) for k, v in feature_impacts.items() if v.get('owners_improvement_pct', 0) > 0],
                    key=lambda x: x[1].get('owners_improvement_pct', 0),
                    reverse=True
                )[:10]

                if sorted_by_owners:
                    # Create dataframe for chart
                    chart_data = pd.DataFrame([
                        {
                            'Feature': feat.replace('tag_', '').replace('_', ' ').title(),
                            'Improvement %': impact['owners_improvement_pct']
                        }
                        for feat, impact in sorted_by_owners
                    ])

                    # Bar chart
                    fig_owners = px.bar(
                        chart_data,
                        x='Improvement %',
                        y='Feature',
                        orientation='h',
                        title='Top 10 Features for Increasing Owners',
                        color='Improvement %',
                        color_continuous_scale='Blues'
                    )
                    fig_owners.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_owners, use_container_width=True)

                    # Show details for top 3
                    st.markdown("**Top 3 Details:**")
                    for feat, impact in sorted_by_owners[:3]:
                        feat_name = feat.replace('tag_', '').replace('_', ' ').title()
                        pct = impact['owners_improvement_pct']

                        if impact['type'] == 'binary':
                            with_val = impact['owners_with']
                            without_val = impact['owners_without']
                            st.markdown(f"- **{feat_name}**: +{pct}% ({without_val:,} → {with_val:,} owners)")
                        else:
                            top_val = impact['owners_top25']
                            bottom_val = impact['owners_bottom25']
                            st.markdown(f"- **{feat_name}**: +{pct}% (top 25% vs bottom 25%: {bottom_val:,} → {top_val:,})")
                else:
                    st.info("No positive improvements found for owners.")

            with col_rev:
                st.markdown("**⭐ Features That Improve Review Ratio**")

                # Get top 10 features sorted by review improvement
                sorted_by_reviews = sorted(
                    [(k, v) for k, v in feature_impacts.items() if v.get('reviews_improvement_pct', 0) > 0],
                    key=lambda x: x[1].get('reviews_improvement_pct', 0),
                    reverse=True
                )[:10]

                if sorted_by_reviews:
                    # Create dataframe for chart
                    chart_data = pd.DataFrame([
                        {
                            'Feature': feat.replace('tag_', '').replace('_', ' ').title(),
                            'Improvement %': impact['reviews_improvement_pct']
                        }
                        for feat, impact in sorted_by_reviews
                    ])

                    # Bar chart
                    fig_reviews = px.bar(
                        chart_data,
                        x='Improvement %',
                        y='Feature',
                        orientation='h',
                        title='Top 10 Features for Improving Review Ratio',
                        color='Improvement %',
                        color_continuous_scale='Greens'
                    )
                    fig_reviews.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_reviews, use_container_width=True)

                    # Show details for top 3
                    st.markdown("**Top 3 Details:**")
                    for feat, impact in sorted_by_reviews[:3]:
                        feat_name = feat.replace('tag_', '').replace('_', ' ').title()
                        pct = impact['reviews_improvement_pct']

                        if impact['type'] == 'binary':
                            with_val = impact['reviews_with']
                            without_val = impact['reviews_without']
                            st.markdown(f"- **{feat_name}**: +{pct}% ({without_val:.1%} → {with_val:.1%} ratio)")
                        else:
                            top_val = impact['reviews_top25']
                            bottom_val = impact['reviews_bottom25']
                            st.markdown(f"- **{feat_name}**: +{pct}% (top 25% vs bottom 25%: {bottom_val:.1%} → {top_val:.1%})")
                else:
                    st.info("No positive improvements found for review ratio.")

        # Generate data-driven recommendations
        recommendations = generate_data_driven_recommendations(
            st.session_state.current_predictions,
            input_features,
            st.session_state.data_analysis
        )
        
        if recommendations:
            st.markdown('<h3 style="color: #4ECDC4;">💡 Data-Driven Recommendations</h3>', unsafe_allow_html=True)
            
            for rec in recommendations:
                priority_color = {
                    'Critical': '🔴',
                    'High': '🟠',
                    'Medium': '🟡',
                    'Low': '🟢',
                    'Info': '🔵'
                }
                
                confidence_indicator = {
                    'Very Strong': '⭐⭐⭐⭐⭐',
                    'Strong': '⭐⭐⭐⭐',
                    'Moderate': '⭐⭐⭐',
                    'Weak': '⭐⭐',
                    'Model Metric': '📊'
                }
                
                st.markdown(f"""
                <div class="recommendation-box">
                    <strong>{priority_color.get(rec['priority'], '⚪')} {rec['type']}</strong> 
                    {confidence_indicator.get(rec.get('data_confidence', ''), '')}<br>
                    {rec['message']}
                </div>
                """, unsafe_allow_html=True)
        
        # Save configuration button
        st.markdown("---")

        # Show success message from previous save (persists across reruns)
        if st.session_state.last_saved_game:
            st.success(f"✅ Last saved: **{st.session_state.last_saved_game}** - Total configurations: {len(st.session_state.saved_games.get(st.session_state.last_saved_game, []))}")

        if st.button("💾 Save This Configuration", use_container_width=True, type="primary"):
            # Use the game name and features from the predictions
            saved_name = save_game_configuration(
                preds['game_name'],
                preds['features'],
                st.session_state.current_predictions
            )
            # Set the flag for next render
            st.session_state.last_saved_game = saved_name

            # Show immediate feedback
            st.balloons()
            st.toast(f"✅ Saved {saved_name}!", icon="✅")

            # Show file location
            if os.path.exists(SAVED_GAMES_FILE):
                file_size = os.path.getsize(SAVED_GAMES_FILE)
                st.info(f"💾 Saved to: `{os.path.abspath(SAVED_GAMES_FILE)}` ({file_size} bytes)")

            # Rerun to show success message
            st.rerun()

