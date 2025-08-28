from transform_player_stats import calculate_player_stats, get_distinct_players
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from functools import lru_cache
import os
import json
import hashlib

CACHE_DIR = 'data/cache/pca'

def ensure_cache_dir():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

def generate_cache_filename(player_name, max_date, current_season):
    # Create a unique identifier based on input parameters
    unique_string = f"{player_name}_{max_date}_{current_season}"
    hashed = hashlib.md5(unique_string.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{hashed}.json")

def save_to_cache(player_name, max_date, current_season, data):
    ensure_cache_dir()
    filename = generate_cache_filename(player_name, max_date, current_season)
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_from_cache(player_name, max_date, current_season):
    filename = generate_cache_filename(player_name, max_date, current_season)
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

@lru_cache(maxsize=None)
def run_pca(metric_type, max_date, current_season="2025", log_queue=None):
    
    players = get_distinct_players()
    stacked_stats = []

    for player in players:
        player_vector = calculate_player_stats(player, max_date, current_season)
        
        if player_vector and player_vector.get('GP', 0) > 5:
            stacked_stats.append(player_vector)
        else:
            player_vector = calculate_player_stats(player, None, str(int(current_season) - 1))
            if player_vector:
                stacked_stats.append(player_vector)

    # Filter out None values
    stacked_stats = [stat for stat in stacked_stats if stat is not None]

    # Check if stacked_stats is empty after filtering
    if not stacked_stats:
        return None, None

    # Convert stacked_stats to a DataFrame
    df = pd.DataFrame(stacked_stats)

    # Select specific fields for PCA
    if metric_type == "offense":
        selected_fields = ['PPG', 'APG', 'TS%', 'TO']
    elif metric_type == "defense":
        selected_fields = ['BPG', 'SPG', 'FPG']
    elif metric_type == "shot_selection":
        selected_fields = ['3PR', 'FTR']
    elif metric_type == "efficiency":
        selected_fields = ['TS%', 'eFG%', 'TO']
    else:
        raise ValueError(f"Invalid metric type: {metric_type}")
    
    numeric_df = df[selected_fields]

    # Remove rows with NaN values and keep track of valid indices
    valid_indices = numeric_df.dropna().index
    numeric_df = numeric_df.loc[valid_indices]

    # Update stacked_stats to match the cleaned data
    stacked_stats = [stacked_stats[i] for i in valid_indices]

    # Convert DataFrame to numpy array
    X = numeric_df.values

    # Check if X is empty
    if X.size == 0:
        return None, None

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create and fit PCA
    pca = PCA(n_components=1)
    pca_result = pca.fit_transform(X_scaled)

    # Create a mapping of player names to PCA results
    player_pca_mapping = {player['PLAYER_NAME']: {metric_type: pca_result[i][0]} for i, player in enumerate(stacked_stats)}

    return player_pca_mapping, stacked_stats

def get_player_pca_score(player_name, max_date, current_season="2025", log_queue=None):
    # Attempt to load from cache
    cached_data = load_from_cache(player_name, max_date, current_season)
    if cached_data:
        return (
            cached_data.get('offense', 0.0),
            cached_data.get('defense', 0.0),
            cached_data.get('shot_selection', 0.0),
            cached_data.get('efficiency', 0.0)
        )
    
    # If not cached, calculate the scores
    pca_result_offense, stacked_stats = run_pca("offense", max_date, current_season, log_queue)

    if pca_result_offense is None or stacked_stats is None:
        return 0.0, 0.0, 0.0, 0.0

    pca_result_defense, _ = run_pca("defense", max_date, current_season, log_queue)
    pca_result_shot_selection, _ = run_pca("shot_selection", max_date, current_season, log_queue)
    pca_result_efficiency, _ = run_pca("efficiency", max_date, current_season, log_queue)

    if not stacked_stats:
        return 0, 0, 0, 0  # Return default scores if no data is found

    try:
        player_pca_score = pca_result_offense.get(player_name, {}).get('offense', 0.0)
        player_pca_score_defense = pca_result_defense.get(player_name, {}).get('defense', 0.0)
        player_pca_score_shot_selection = pca_result_shot_selection.get(player_name, {}).get('shot_selection', 0.0)
        player_pca_score_efficiency = pca_result_efficiency.get(player_name, {}).get('efficiency', 0.0)
        
        # Prepare data to cache
        data_to_cache = {
            'offense': player_pca_score,
            'defense': player_pca_score_defense,
            'shot_selection': player_pca_score_shot_selection,
            'efficiency': player_pca_score_efficiency
        }
        
        # Save to cache
        save_to_cache(player_name, max_date, current_season, data_to_cache)
        
        return player_pca_score, player_pca_score_defense, player_pca_score_shot_selection, player_pca_score_efficiency
    except Exception as e:
        return 0.0, 0.0, 0.0, 0.0
