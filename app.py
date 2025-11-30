"""
Game Launch Decision Support System (IDSS) - Main Application
An Intelligent Decision Support System using Steam data for game developers

Key Features:
- Market position analysis (percentile-based, not absolute predictions)
- Improvement opportunities with actionable scenarios
- Risk assessment and success factor analysis
- Competitive positioning insights
"""

import streamlit as st

# Import page modules
from new_game import new_game_page
from my_games import my_games_page
from data_analysis import data_analysis_page
from persistence import initialize_persistence


# Set page configuration
st.set_page_config(
    page_title="Game Launch Decision Support System",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'saved_games' not in st.session_state:
    st.session_state.saved_games = {}
if 'current_predictions' not in st.session_state:
    st.session_state.current_predictions = {}
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'data_analysis' not in st.session_state:
    st.session_state.data_analysis = {}
if 'configurations' not in st.session_state:
    st.session_state.configurations = []
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'last_saved_game' not in st.session_state:
    st.session_state.last_saved_game = None
if 'show_predictions' not in st.session_state:
    st.session_state.show_predictions = False

# Initialize persistence (load saved games)
initialize_persistence(st.session_state)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        padding: 1.5rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .insight-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    .opportunity-box {
        background-color: #cce5ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    
    /* Better tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application function"""
    # Header
    st.markdown('<h1 class="main-header">🎮 Game Launch Decision Support System</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p class="sub-header">
    Understand where your game fits in the Steam market and discover actionable improvements
    </p>
    """, unsafe_allow_html=True)

    # Value proposition
    with st.expander("ℹ️ How This System Helps You", expanded=False):
        st.markdown("""
        **This is an Intelligent Decision Support System (IDSS)** that helps game developers make data-driven decisions:
        
        **What We Provide:**
        - 📊 **Market Positioning** - See where your game would rank among 27,000+ Steam games
        - 🚀 **Improvement Scenarios** - Discover specific changes that could boost your position
        - ⚠️ **Risk Assessment** - Identify potential issues before launch
        - ✅ **Success Factors** - Check which best practices you're following
        - 📈 **Competitive Analysis** - Compare against similar games in your niche
        
        **What We DON'T Provide:**
        - ❌ Exact sales predictions (too many unpredictable factors)
        - ❌ Guaranteed outcomes (marketing and quality matter more)
        - ❌ Review score predictions (quality can't be measured pre-launch)
        
        **Philosophy:** Focus on **relative positioning** and **actionable improvements**, 
        not on unreliable absolute predictions.
        """)

    # Navigation
    st.markdown("---")

    # Create tabs for navigation
    tab1, tab2, tab3 = st.tabs([
        "🎮 Analyze New Game", 
        "📚 My Configurations", 
        "📊 Market Data"
    ])

    with tab1:
        new_game_page()

    with tab2:
        my_games_page()

    with tab3:
        data_analysis_page()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #888;">
        <p><strong>Game Launch IDSS v3.0</strong> | Intelligent Decision Support</p>
        <p style="font-size: 0.85rem;">
        Powered by machine learning on 27,000+ Steam games | 
        Focusing on insights over predictions
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
