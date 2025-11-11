"""
Data persistence functions for saving and loading game configurations
"""

import os
import json
import time
import streamlit as st
from datetime import datetime
from src.utils import convert_numpy_types


# File location for saved games
SAVED_GAMES_FILE = "saved_games.json"


def migrate_old_config_ids(saved_games, configurations):
    """Migrate old config_id format to new timestamp-based format"""
    migrated = False

    # Fix config IDs in saved_games
    for game_name, configs in saved_games.items():
        for i, config in enumerate(configs):
            old_id = config.get('config_id', '')
            # Check if it's the old format (e.g., "Game_config_1")
            if old_id.endswith(f'_config_{i+1}') or '_config_' in old_id and not old_id.split('_')[-1].isdigit() or len(old_id.split('_')[-1]) < 10:
                # Generate new unique ID
                new_id = f"{game_name}_{int(time.time() * 1000)}_{i}"
                config['config_id'] = new_id
                migrated = True

    # Fix config IDs in configurations list
    for config in configurations:
        game_name = config.get('game_name', '')
        old_id = config.get('config_id', '')
        # Check if old format
        if '_config_' in old_id and not old_id.split('_')[-1].isdigit() or (old_id.split('_')[-1].isdigit() and len(old_id.split('_')[-1]) < 10):
            # Find matching config in saved_games to get the new ID
            if game_name in saved_games:
                for saved_config in saved_games[game_name]:
                    if saved_config.get('timestamp') == config.get('timestamp'):
                        config['config_id'] = saved_config['config_id']
                        break

    return saved_games, configurations, migrated


def load_saved_games():
    """Load saved games from JSON file and migrate old IDs if needed"""
    if os.path.exists(SAVED_GAMES_FILE):
        try:
            with open(SAVED_GAMES_FILE, 'r') as f:
                data = json.load(f)
                saved_games = data.get('saved_games', {})
                configurations = data.get('configurations', [])

                # Migrate old config_ids to new format
                saved_games, configurations, migrated = migrate_old_config_ids(saved_games, configurations)

                if migrated:
                    print("🔄 Migrated old config IDs to new timestamp-based format")
                    # Save the migrated data
                    data = {
                        'saved_games': saved_games,
                        'configurations': configurations
                    }
                    with open(SAVED_GAMES_FILE, 'w') as f:
                        json.dump(data, f, indent=2)

                return saved_games, configurations
        except (json.JSONDecodeError, IOError) as e:
            st.warning(f"⚠️ Could not load saved games: {e}")
            return {}, []
    return {}, []


def save_games_to_file():
    """Save games to JSON file"""
    try:
        # Convert numpy types to native Python types
        saved_games = convert_numpy_types(st.session_state.saved_games)
        configurations = convert_numpy_types(st.session_state.configurations)

        data = {
            'saved_games': saved_games,
            'configurations': configurations
        }

        with open(SAVED_GAMES_FILE, 'w') as f:
            json.dump(data, f, indent=2)

        # Verify file was created
        if os.path.exists(SAVED_GAMES_FILE):
            file_size = os.path.getsize(SAVED_GAMES_FILE)
            print(f"✅ Saved games to {SAVED_GAMES_FILE} ({file_size} bytes)")
            return True
        else:
            st.error(f"❌ File {SAVED_GAMES_FILE} was not created")
            return False
    except Exception as e:
        st.error(f"❌ Error saving games: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def save_game_configuration(game_name, features, predictions):
    """Save game configuration with unique ID"""
    original_name = game_name

    # Check if game already exists in saved games
    if original_name not in st.session_state.saved_games:
        st.session_state.saved_games[original_name] = []

    # Generate UNIQUE config_id using timestamp to avoid duplicates after deletions
    # Format: gamename_timestamp (e.g., "My Game_1699123456789")
    unique_id = f"{original_name}_{int(time.time() * 1000)}"

    # Create configuration entry
    config = {
        'config_id': unique_id,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'game_name': original_name,
        'features': features.copy(),
        'predictions': predictions.copy()
    }

    # Add to game's configurations
    st.session_state.saved_games[original_name].append(config)

    # Also add to global configurations list for ranking
    st.session_state.configurations.append(config)

    # Persist to file
    save_games_to_file()

    return original_name


def delete_configuration(game_name, config_id):
    """Delete a specific configuration"""
    try:
        # Find and remove from game's configuration list
        if game_name in st.session_state.saved_games:
            st.session_state.saved_games[game_name] = [
                c for c in st.session_state.saved_games[game_name]
                if c['config_id'] != config_id
            ]

            # If no configurations left for this game, remove the game entirely
            if not st.session_state.saved_games[game_name]:
                del st.session_state.saved_games[game_name]
                # Clear last_saved_game if it was this game
                if st.session_state.last_saved_game == game_name:
                    st.session_state.last_saved_game = None

        # Remove from global configurations list
        st.session_state.configurations = [
            c for c in st.session_state.configurations
            if c['config_id'] != config_id
        ]

        # Persist to file
        save_games_to_file()
        return True
    except Exception as e:
        st.error(f"❌ Error deleting configuration: {e}")
        return False


def delete_game(game_name):
    """Delete an entire game and all its configurations"""
    try:
        # Get all config IDs for this game
        if game_name in st.session_state.saved_games:
            config_ids = [c['config_id'] for c in st.session_state.saved_games[game_name]]

            # Remove game from saved_games
            del st.session_state.saved_games[game_name]

            # Clear last_saved_game if it was this game
            if st.session_state.last_saved_game == game_name:
                st.session_state.last_saved_game = None

            # Remove all configurations from global list
            st.session_state.configurations = [
                c for c in st.session_state.configurations
                if c['config_id'] not in config_ids
            ]

            # Persist to file
            save_games_to_file()
            return True
    except Exception as e:
        st.error(f"❌ Error deleting game: {e}")
        return False


def initialize_persistence(session_state):
    """Initialize persistence by loading saved games from file"""
    if not session_state.get('data_loaded', False):
        saved_games, configurations = load_saved_games()
        if saved_games:
            session_state.saved_games = saved_games
            session_state.configurations = configurations
        session_state.data_loaded = True
