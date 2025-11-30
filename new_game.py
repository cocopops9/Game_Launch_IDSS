"""
New Game Analysis Page - Intelligent Decision Support Interface
Focuses on actionable insights rather than raw predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from models import load_steam_data, train_models
from intelligence_engine import IntelligenceEngine, create_intelligence_engine
from persistence import save_game_configuration


def new_game_page():
    """Main page for new game analysis"""
    
    st.markdown("## 🎮 Analyze Your Game Configuration")
    st.markdown("""
    Configure your game's parameters below to see where it would likely position 
    in the Steam market and get actionable recommendations for improvement.
    """)
    
    # Load data and models
    if 'models' not in st.session_state or not st.session_state.get('models_trained'):
        with st.spinner("Loading market data and training models..."):
            df, loader = load_steam_data()
            models = train_models(df, loader)
            st.session_state.models = models
            st.session_state.df = df
            st.session_state.models_trained = True
    
    # Initialize intelligence engine
    if 'intelligence_engine' not in st.session_state:
        st.session_state.intelligence_engine = create_intelligence_engine(
            st.session_state.df, 
            st.session_state.models
        )
    
    engine = st.session_state.intelligence_engine
    
    # --- INPUT SECTION ---
    st.markdown("### ⚙️ Game Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        game_name = st.text_input(
            "Game Name",
            value="My Awesome Game",
            help="Name for saving this configuration"
        )
        
        price = st.slider(
            "Price (USD)",
            min_value=0.0,
            max_value=59.99,
            value=14.99,
            step=0.01,
            format="$%.2f",
            help="Set to $0 for free-to-play"
        )
        
        platforms = st.multiselect(
            "Platforms",
            options=["windows", "mac", "linux"],
            default=["windows"],
            help="Which platforms will you support?"
        )
        
        achievements = st.number_input(
            "Number of Achievements",
            min_value=0,
            max_value=500,
            value=0,
            help="Steam achievements (0 if none)"
        )
    
    with col2:
        release_month = st.selectbox(
            "Release Month",
            options=list(range(1, 13)),
            index=10,  # November default
            format_func=lambda x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][x-1],
            help="Planned release month"
        )
        
        genres = st.multiselect(
            "Primary Genres",
            options=['Indie', 'Action', 'Adventure', 'Strategy', 'Simulation', 
                    'RPG', 'Casual', 'Sports', 'Racing', 'Puzzle', 'Horror',
                    'Free to Play', 'Early Access', 'Massively Multiplayer'],
            default=['Indie'],
            help="Select up to 3-4 primary genres"
        )
        
        tags = st.multiselect(
            "Key Tags",
            options=['Singleplayer', 'Multiplayer', 'Co-op', 'Open World', 'Survival',
                    'Roguelike', 'Sandbox', 'Story Rich', 'Atmospheric', 'Horror',
                    'FPS', 'Platformer', 'Puzzle', 'VR', 'Pixel Graphics', 'Retro',
                    'Fantasy', 'Sci-fi', 'Anime', 'Tower Defense'],
            default=['Singleplayer'],
            help="Key gameplay tags"
        )
        
        required_age = st.selectbox(
            "Age Rating",
            options=[0, 13, 16, 18],
            index=0,
            format_func=lambda x: "Everyone" if x == 0 else f"{x}+",
            help="Minimum age requirement"
        )
    
    # Steam Features
    st.markdown("#### Steam Features")
    feature_cols = st.columns(4)
    
    categories = []
    with feature_cols[0]:
        if st.checkbox("Steam Achievements", value=achievements > 0):
            categories.append("Steam Achievements")
    with feature_cols[1]:
        if st.checkbox("Steam Trading Cards"):
            categories.append("Steam Trading Cards")
    with feature_cols[2]:
        if st.checkbox("Steam Cloud"):
            categories.append("Steam Cloud")
    with feature_cols[3]:
        if st.checkbox("Controller Support"):
            categories.append("Full controller support")
    
    # Build features dict
    features = {
        'price': price,
        'platforms': platforms,
        'genres': genres,
        'tags': tags,
        'categories': categories,
        'release_month': release_month,
        'achievements': achievements,
        'required_age': required_age
    }
    
    # --- ANALYSIS BUTTON ---
    st.markdown("---")
    
    analyze_col, save_col = st.columns([3, 1])
    
    with analyze_col:
        analyze_clicked = st.button("🔍 Analyze Market Position", type="primary", use_container_width=True)
    
    # --- RESULTS SECTION ---
    if analyze_clicked or st.session_state.get('show_predictions'):
        st.session_state.show_predictions = True
        
        with st.spinner("Analyzing market position..."):
            analysis = engine.analyze_game(features)
        
        st.session_state.current_analysis = analysis
        st.session_state.current_features = features
        
        display_analysis_results(analysis, features)
        
        # Save button
        st.markdown("---")
        save_col1, save_col2 = st.columns([3, 1])
        with save_col1:
            if st.button("💾 Save This Configuration", use_container_width=True):
                predictions = {
                    'percentile': analysis['positioning']['overall_percentile'],
                    'tier': analysis['positioning']['tier_name'],
                    'readiness_score': analysis['success_factors']['readiness_score'],
                    'raw_owners': analysis['raw_prediction']['owners']
                }
                save_game_configuration(game_name, features, predictions)
                st.success(f"✅ Saved configuration for '{game_name}'")
                st.session_state.last_saved_game = game_name


def display_analysis_results(analysis: dict, features: dict):
    """Display the intelligent analysis results"""
    
    positioning = analysis['positioning']
    insights = analysis['insights']
    improvements = analysis['improvements']
    risks = analysis['risks']
    success_factors = analysis['success_factors']

    # --- MARKET POSITION DASHBOARD ---
    st.markdown("## 📊 Market Position Analysis")

    # Main positioning metrics
    metric_cols = st.columns(3)
    
    percentile = positioning['overall_percentile']
    tier = positioning['tier_name'].replace('_', ' ').title()
    
    with metric_cols[0]:
        st.metric(
            "Market Percentile",
            f"{percentile:.0f}th",
            delta=f"Top {100-percentile:.0f}%" if percentile >= 50 else f"Bottom {percentile:.0f}%",
            delta_color="normal" if percentile >= 50 else "inverse"
        )
    
    with metric_cols[1]:
        st.metric(
            "Success Tier",
            tier,
            help="Based on predicted reach compared to all Steam games"
        )
    
    with metric_cols[2]:
        st.metric(
            "Readiness Score",
            f"{success_factors['readiness_score']}%",
            help="How many success factors are present"
        )

    # Visual percentile gauge
    st.markdown("### Your Market Position")
    fig = create_percentile_gauge(percentile)
    st.plotly_chart(fig, use_container_width=True)
    
    # --- KEY INSIGHTS ---
    st.markdown("## 💡 Key Insights")
    
    for insight in insights[:4]:
        icon = get_insight_icon(insight.action_type)
        color = get_insight_color(insight.action_type)
        
        st.markdown(f"""
        <div style="background-color: {color}; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
            <strong>{icon} {insight.title}</strong><br>
            <span style="color: #444;">{insight.message}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # --- IMPROVEMENT SCENARIOS ---
    st.markdown("## 🚀 Improvement Opportunities")
    
    if improvements:
        st.markdown("""
        These changes could improve your market position. 
        Percentile gains are based on analysis of similar games in the dataset.
        """)
        
        # Create improvement chart
        fig = create_improvement_chart(improvements)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detail each improvement
        for i, imp in enumerate(improvements[:3], 1):
            with st.expander(f"**{i}. {imp.change_description}** (+{imp.percentile_gain:.1f} percentile points)"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Position", f"{imp.current_percentile:.0f}th")
                with col2:
                    st.metric("New Position", f"{imp.new_percentile:.0f}th")
                with col3:
                    st.metric("Similar Games", f"{imp.similar_games_count:,}")
                
                confidence_text = {
                    'high': '✅ High confidence - strong historical evidence',
                    'medium': '⚠️ Medium confidence - based on general patterns',
                    'low': '❓ Low confidence - limited data available'
                }
                st.caption(confidence_text.get(imp.confidence, ''))
    else:
        st.info("Your configuration already incorporates many best practices! Minor optimizations may still be possible.")
    
    # --- RISK ASSESSMENT ---
    st.markdown("## ⚠️ Risk Assessment")
    
    if risks:
        for risk in risks:
            level_colors = {
                'critical': '🔴',
                'high': '🟠',
                'moderate': '🟡',
                'low': '🟢'
            }
            icon = level_colors.get(risk['level'], '⚪')
            
            with st.expander(f"{icon} **{risk['title']}** ({risk['level'].title()} Risk)"):
                st.markdown(f"**Category:** {risk['category']}")
                st.markdown(risk['description'])
                st.markdown(f"**Recommendation:** {risk['recommendation']}")
    else:
        st.success("✅ No significant risks identified with this configuration.")
    
    # --- SUCCESS FACTORS ---
    st.markdown("## ✅ Success Factors Checklist")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Present ✅")
        for factor in success_factors['present']:
            impact_badge = get_impact_badge(factor['impact'])
            st.markdown(f"- **{factor['factor']}** {impact_badge}")
            st.caption(f"  ↳ {factor['reason']}")
    
    with col2:
        st.markdown("### Missing ❌")
        if success_factors['missing']:
            for factor in success_factors['missing']:
                impact_badge = get_impact_badge(factor['impact'])
                st.markdown(f"- **{factor['factor']}** {impact_badge}")
                st.caption(f"  ↳ {factor['reason']}")
        else:
            st.success("All major success factors are present!")
    
    # --- CONFIDENCE DISCLAIMER ---
    st.markdown("---")
    st.caption(f"**Analysis Note:** {analysis['confidence_statement']}")


def create_percentile_gauge(percentile: float) -> go.Figure:
    """Create a visual gauge showing market position"""
    
    # Determine color based on percentile
    if percentile >= 80:
        color = "#28a745"  # Green
    elif percentile >= 60:
        color = "#17a2b8"  # Blue
    elif percentile >= 40:
        color = "#ffc107"  # Yellow
    elif percentile >= 20:
        color = "#fd7e14"  # Orange
    else:
        color = "#dc3545"  # Red
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=percentile,
        title={'text': "Market Percentile", 'font': {'size': 20}},
        delta={'reference': 50, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#ffebee'},
                {'range': [20, 40], 'color': '#fff3e0'},
                {'range': [40, 60], 'color': '#fffde7'},
                {'range': [60, 80], 'color': '#e3f2fd'},
                {'range': [80, 100], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': percentile
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_improvement_chart(improvements) -> go.Figure:
    """Create a chart showing improvement scenarios"""
    
    if not improvements:
        return go.Figure()
    
    scenarios = [imp.change_description[:30] + "..." if len(imp.change_description) > 30 
                 else imp.change_description for imp in improvements]
    gains = [imp.percentile_gain for imp in improvements]
    current = [imp.current_percentile for imp in improvements]
    new = [imp.new_percentile for imp in improvements]
    
    fig = go.Figure()
    
    # Current position bars
    fig.add_trace(go.Bar(
        name='Current Position',
        x=scenarios,
        y=current,
        marker_color='lightgray',
        text=[f'{v:.0f}th' for v in current],
        textposition='inside'
    ))
    
    # Potential gain bars
    fig.add_trace(go.Bar(
        name='Potential Gain',
        x=scenarios,
        y=gains,
        marker_color='#28a745',
        text=[f'+{v:.1f}' for v in gains],
        textposition='inside'
    ))
    
    fig.update_layout(
        barmode='stack',
        title='Improvement Potential by Scenario',
        yaxis_title='Market Percentile',
        xaxis_title='',
        height=300,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig


def get_insight_icon(action_type: str) -> str:
    """Get icon for insight type"""
    icons = {
        'success': '✅',
        'opportunity': '🚀',
        'warning': '⚠️',
        'info': 'ℹ️'
    }
    return icons.get(action_type, '📌')


def get_insight_color(action_type: str) -> str:
    """Get background color for insight type"""
    colors = {
        'success': '#d4edda',
        'opportunity': '#cce5ff',
        'warning': '#fff3cd',
        'info': '#e2e3e5'
    }
    return colors.get(action_type, '#f8f9fa')


def get_impact_badge(impact: str) -> str:
    """Get a badge for impact level"""
    badges = {
        'high': '🔴 High Impact',
        'medium': '🟡 Medium Impact',
        'low': '🟢 Low Impact'
    }
    return badges.get(impact, '')
