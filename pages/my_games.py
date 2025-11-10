"""
My Games page - view and manage saved configurations
"""

import pandas as pd
import streamlit as st

from persistence import delete_configuration, delete_game


def my_games_page():
    """My Games page - view and manage saved configurations"""
    st.markdown('<h2 class="sub-header">📚 My Saved Games & Configurations</h2>', unsafe_allow_html=True)

    if not st.session_state.saved_games:
        st.info("No saved games yet. Go to 'New Game' to create and save game configurations.")
        return

    # Display games and their configurations
    st.subheader("🎮 Saved Games")

    # Create a copy of items to iterate safely during deletion
    games_list = list(st.session_state.saved_games.items())

    for game_name, configs in games_list:
        # Skip if game was deleted
        if game_name not in st.session_state.saved_games:
            continue

        with st.expander(f"📁 {game_name} ({len(configs)} configurations)", expanded=False):
            # Delete entire game button
            col_del1, col_del2 = st.columns([3, 1])
            with col_del2:
                if st.button(f"🗑️ Delete Game", key=f"delete_game_{game_name}", type="secondary", use_container_width=True):
                    if delete_game(game_name):
                        st.success(f"✅ Deleted game '{game_name}' and all its configurations!")
                        st.rerun()

            st.markdown("---")

            # Display each configuration
            for idx, config in enumerate(configs, 1):
                col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 1])

                with col1:
                    st.write(f"**Config #{idx}**")

                with col2:
                    st.write(f"💰 ${config['features']['price']:.2f}")
                    platforms = []
                    if config['features'].get('windows', 0): platforms.append('Win')
                    if config['features'].get('mac', 0): platforms.append('Mac')
                    if config['features'].get('linux', 0): platforms.append('Linux')
                    st.write(f"🖥️ {'/'.join(platforms) if platforms else 'None'}")

                with col3:
                    st.write(f"👥 {int(config['predictions']['owners']):,} owners")
                    st.write(f"⭐ {config['predictions']['review_ratio']:.1%} reviews")

                with col4:
                    st.write(f"📅 {config['timestamp']}")

                with col5:
                    # Delete individual configuration button
                    if st.button("🗑️", key=f"delete_config_{config['config_id']}", help="Delete this configuration"):
                        if delete_configuration(game_name, config['config_id']):
                            st.success(f"✅ Deleted configuration!")
                            st.rerun()

                st.markdown("---")
    
    # Display ranked configurations (all games)
    st.subheader("🏆 All Configurations Ranking")
    
    if st.session_state.configurations:
        # Create ranking table
        ranking_data = []
        for config in st.session_state.configurations:
            ranking_data.append({
                'Game': config['game_name'],
                'Config ID': config['config_id'].split('_')[-1],
                'Price': f"${config['features']['price']:.2f}",
                'Platforms': '/'.join([p for p in ['Win', 'Mac', 'Linux'] 
                                      if config['features'].get(p.lower() if p != 'Win' else 'windows', 0)]),
                'Release Month': config['features'].get('release_month', 'N/A'),
                'Predicted Owners': int(config['predictions']['owners']),
                'Review Ratio': f"{config['predictions']['review_ratio']:.1%}",
                'Timestamp': config['timestamp']
            })
        
        df_ranking = pd.DataFrame(ranking_data)
        
        # Sort by predicted owners
        df_ranking = df_ranking.sort_values('Predicted Owners', ascending=False)
        df_ranking['Rank'] = range(1, len(df_ranking) + 1)
        
        # Reorder columns
        column_order = ['Rank', 'Game', 'Config ID', 'Price', 'Platforms', 
                       'Release Month', 'Predicted Owners', 'Review Ratio', 'Timestamp']
        df_ranking = df_ranking[column_order]
        
        # Display with formatting
        st.dataframe(
            df_ranking,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                "Predicted Owners": st.column_config.NumberColumn("Predicted Owners", format="%d"),
            }
        )
        
        # Summary statistics
        st.subheader("📊 Configuration Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Configurations", len(st.session_state.configurations))
        
        with col2:
            st.metric("Unique Games", len(st.session_state.saved_games))
        
        with col3:
            avg_owners = df_ranking['Predicted Owners'].mean()
            st.metric("Avg Predicted Owners", f"{int(avg_owners):,}")
        
        with col4:
            best_config = df_ranking.iloc[0]
            st.metric("Best Configuration", f"{best_config['Game']} #{best_config['Config ID']}")

