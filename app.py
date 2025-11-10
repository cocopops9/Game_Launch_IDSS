"""
Game Launch Decision Support System (IDSS) - Main Application
Uses actual Steam data and provides data-driven recommendations
"""

import streamlit as st

# Import page modules
from pages import new_game_page, my_games_page, data_analysis_page
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
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        padding: 2rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .recommendation-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF6B6B;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application function"""
    # Header
    st.markdown('<h1 class="main-header">🎮 Game Launch Decision Support System</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: center; font-size: 1.2rem;">
    Data-driven predictions and recommendations based on real Steam data analysis
    </p>
    """, unsafe_allow_html=True)

    # Navigation
    st.markdown("---")

    # Create tabs for navigation
    tab1, tab2, tab3 = st.tabs(["🎮 New Game", "📚 My Games", "📊 Data Analysis"])

    with tab1:
        new_game_page()

    with tab2:
        my_games_page()

    with tab3:
        data_analysis_page()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #888;">
        <p>Game Launch IDSS v2.0 | Powered by Real Steam Data Analysis</p>
        <p style="font-size: 0.9rem;">Using GradientBoosting & LightGBM on actual game data</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
