"""
My Games Page - View and compare saved game configurations
"""

import streamlit as st
import pandas as pd
from persistence import delete_configuration, delete_game


def my_games_page():
    """Display saved game configurations"""
    
    st.markdown("## 📚 My Saved Configurations")
    
    if not st.session_state.saved_games:
        st.info("No saved configurations yet. Go to the 'New Game' tab to analyze and save a game configuration.")
        return
    
    # Summary metrics
    total_configs = len(st.session_state.configurations)
    total_games = len(st.session_state.saved_games)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Games", total_games)
    with col2:
        st.metric("Total Configurations", total_configs)
    
    st.markdown("---")
    
    # Create ranking table
    if st.session_state.configurations:
        st.markdown("### 📊 Configuration Rankings")
        st.caption("Sorted by market percentile (higher = better position)")
        
        # Build comparison dataframe
        data = []
        for config in st.session_state.configurations:
            predictions = config.get('predictions', {})
            features = config.get('features', {})
            
            data.append({
                'Game': config.get('game_name', 'Unknown'),
                'Percentile': predictions.get('percentile', 0),
                'Tier': predictions.get('tier', 'unknown').replace('_', ' ').title(),
                'Readiness': f"{predictions.get('readiness_score', 0)}%",
                'Price': f"${features.get('price', 0):.2f}",
                'Platforms': ', '.join(features.get('platforms', [])),
                'Genres': ', '.join(features.get('genres', [])[:2]),
                'Saved': config.get('timestamp', ''),
                'config_id': config.get('config_id', '')
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Percentile', ascending=False)
        
        # Display as styled table
        display_df = df[['Game', 'Percentile', 'Tier', 'Readiness', 'Price', 'Platforms', 'Genres', 'Saved']]
        display_df['Percentile'] = display_df['Percentile'].apply(lambda x: f"{x:.0f}th")
        
        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                'Game': st.column_config.TextColumn('Game', width='medium'),
                'Percentile': st.column_config.TextColumn('Market Position', width='small'),
                'Tier': st.column_config.TextColumn('Tier', width='small'),
                'Readiness': st.column_config.TextColumn('Ready', width='small'),
                'Price': st.column_config.TextColumn('Price', width='small'),
                'Platforms': st.column_config.TextColumn('Platforms', width='medium'),
                'Genres': st.column_config.TextColumn('Genres', width='medium'),
                'Saved': st.column_config.TextColumn('Saved', width='medium')
            }
        )
    
    st.markdown("---")
    
    # Individual game sections
    st.markdown("### 🎮 Game Details")
    
    for game_name, configs in st.session_state.saved_games.items():
        with st.expander(f"**{game_name}** ({len(configs)} configuration{'s' if len(configs) > 1 else ''})"):
            
            for i, config in enumerate(configs):
                predictions = config.get('predictions', {})
                features = config.get('features', {})
                
                st.markdown(f"#### Configuration {i+1}")
                st.caption(f"Saved: {config.get('timestamp', 'Unknown')}")
                
                # Metrics row
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Market Position", f"{predictions.get('percentile', 0):.0f}th percentile")
                with metric_cols[1]:
                    st.metric("Tier", predictions.get('tier', 'unknown').replace('_', ' ').title())
                with metric_cols[2]:
                    st.metric("Readiness", f"{predictions.get('readiness_score', 0)}%")
                with metric_cols[3]:
                    st.metric("Price", f"${features.get('price', 0):.2f}")
                
                # Features summary
                st.markdown("**Configuration Details:**")
                detail_cols = st.columns(3)
                with detail_cols[0]:
                    st.markdown(f"- **Platforms:** {', '.join(features.get('platforms', []))}")
                    st.markdown(f"- **Achievements:** {features.get('achievements', 0)}")
                with detail_cols[1]:
                    st.markdown(f"- **Genres:** {', '.join(features.get('genres', []))}")
                    st.markdown(f"- **Tags:** {', '.join(features.get('tags', [])[:3])}")
                with detail_cols[2]:
                    st.markdown(f"- **Release Month:** {features.get('release_month', 'Not set')}")
                    st.markdown(f"- **Age Rating:** {features.get('required_age', 0)}+")
                
                # Delete button
                if st.button(f"🗑️ Delete", key=f"delete_{config.get('config_id')}"):
                    delete_configuration(game_name, config.get('config_id'))
                    st.success(f"Deleted configuration")
                    st.rerun()
                
                st.markdown("---")
            
            # Delete entire game button
            if st.button(f"🗑️ Delete All '{game_name}' Configurations", key=f"delete_game_{game_name}"):
                delete_game(game_name)
                st.success(f"Deleted all configurations for '{game_name}'")
                st.rerun()
    
    # Export section
    st.markdown("---")
    st.markdown("### 📤 Export Data")
    
    if st.button("📋 Export All Configurations as CSV"):
        if st.session_state.configurations:
            export_data = []
            for config in st.session_state.configurations:
                predictions = config.get('predictions', {})
                features = config.get('features', {})
                export_data.append({
                    'game_name': config.get('game_name'),
                    'timestamp': config.get('timestamp'),
                    'percentile': predictions.get('percentile'),
                    'tier': predictions.get('tier'),
                    'readiness_score': predictions.get('readiness_score'),
                    'price': features.get('price'),
                    'platforms': '|'.join(features.get('platforms', [])),
                    'genres': '|'.join(features.get('genres', [])),
                    'tags': '|'.join(features.get('tags', [])),
                    'achievements': features.get('achievements'),
                    'release_month': features.get('release_month')
                })
            
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name="game_configurations.csv",
                mime="text/csv"
            )
