import pandas as pd
from datetime import datetime, timedelta
import os
import json
import hashlib

# Configuration - Change this to the desired season year
SEASON_YEAR = "2023-2024"  # Can be "2021-2022", "2022-2023", or "2023-2024"

def load_player_data(season_year):
    """Load player boxscore data for the specified season year."""
    # Map season year to file path
    file_mapping = {
        "2020-2021": "data/player_boxscores/historical/NBA-2020-2021-Player-BoxScore-Dataset.xlsx",
        "2021-2022": "data/player_boxscores/historical/NBA-2021-2022-Player-BoxScore-Dataset.xlsx",
        "2022-2023": "data/player_boxscores/historical/NBA-2022-2023-Player-BoxScore-Dataset.xlsx", 
        "2023-2024": "data/player_boxscores/historical/NBA-2023-2024-Player-BoxScore-Dataset.xlsx",
        "2024-2025": "data/player_boxscores/historical/NBA-2024-2025-Player-BoxScore-Dataset.xlsx"
    }
    
    if season_year not in file_mapping:
        raise ValueError(f"Season year {season_year} not supported. Available options: {list(file_mapping.keys())}")
    
    file_path = file_mapping[season_year]
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    return df

# Load data for the configured season year
df = load_player_data(SEASON_YEAR)

# Cache Configuration
CACHE_DIR = 'data/cache/player_stats'

def ensure_cache_dir():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

def generate_cache_filename(player_name, max_date, current_season):
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

def calculate_player_stats(player_name, max_date=None, current_season=None):
    """
    Calculates and returns aggregated player statistics for the configured season.
    Implements caching to avoid redundant calculations.
    """
    # Use provided season or fall back to global SEASON_YEAR
    season_to_use = current_season if current_season else SEASON_YEAR
    
    # Attempt to load from cache
    cached_data = load_from_cache(player_name, max_date, season_to_use)
    if cached_data:
        return cached_data

    # If current_season is provided and different from loaded SEASON_YEAR, load that season's data
    if current_season and current_season != SEASON_YEAR:
        try:
            # Handle both single year (e.g., "2024") and full format (e.g., "2023-2024")
            if "-" in current_season:
                # Already in proper format (e.g., "2023-2024")
                season_format = current_season
            else:
                # Convert single year to season format (e.g., "2024" -> "2023-2024")
                season_year = int(current_season)
                season_format = f"{season_year-1}-{current_season}"
            
            temp_df = load_player_data(season_format)
            player_df = temp_df[temp_df['PLAYER \nFULL NAME'] == player_name]
        except (ValueError, FileNotFoundError):
            # Fallback to current loaded data if season loading fails
            player_df = df[df['PLAYER \nFULL NAME'] == player_name]
    else:
        # Get player data from the loaded DataFrame
        player_df = df[df['PLAYER \nFULL NAME'] == player_name]
    
    if len(player_df) == 0:
        return None
        
    # If max_date is provided, filter the games up to but not including that date
    if max_date is not None:
        player_df = player_df[pd.to_datetime(player_df['DATE']) < pd.to_datetime(max_date)]
    
    # Select only numeric columns for summing
    numeric_cols = ['MIN', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA', 'OR', 'DR', 'TOT', 'A', 'PF', 'ST', 'TO', 'BL', 'PTS']
    
    # Sum up all numeric columns
    stats_sum = player_df[numeric_cols].sum().to_dict()
    
    # Calculate average for percentage-based stats (usage rate)
    stats_sum['USAGE_RATE'] = player_df['USAGE \nRATE (%)'].mean() if len(player_df) > 0 else None
    # Add player name to the beginning of the dictionary
    stats_sum = {
                'PLAYER_NAME': player_name
              , '2P': round(stats_sum['FG'] - stats_sum['3P'], 3)
              , 'GP': len(player_df)
              , 'MPG': round(stats_sum['MIN'] / len(player_df), 3) if len(player_df) != 0 else None
              , '3P%': round(stats_sum['3P'] / stats_sum['3PA'], 3) if stats_sum['3PA'] != 0 else None
              , 'FT%': round(stats_sum['FT'] / stats_sum['FTA'], 3) if stats_sum['FTA'] != 0 else None
              , 'eFG%': round((stats_sum['FG'] + 0.5 * stats_sum['3P']) / stats_sum['FGA'], 3) if stats_sum['FGA'] != 0 else None
              , 'TS%': round(stats_sum['PTS'] / (2 * (stats_sum['FGA'] + 0.44 * stats_sum['FTA'])), 3) if (stats_sum['FGA'] + 0.44 * stats_sum['FTA']) != 0 else None
              , '3PR': round(stats_sum['3PA'] / stats_sum['FGA'], 3) if stats_sum['FGA'] != 0 else None
              , 'FTR': round(stats_sum['FTA'] / (stats_sum['FGA'] + stats_sum['FTA']), 3) if (stats_sum['FGA'] + stats_sum['FTA']) != 0 else None
              , 'PFFT': round(stats_sum['FT'] / stats_sum['PTS'], 3) if stats_sum['PTS'] != 0 else None
              , 'PPG': round(stats_sum['PTS'] / len(player_df), 3) if len(player_df) != 0 else None
              , 'RPG': round((stats_sum['OR'] + stats_sum['DR']) / len(player_df), 3) if len(player_df) != 0 else None
              , 'DRPG': round(stats_sum['DR'] / len(player_df), 3) if len(player_df) != 0 else None
              , 'ORPG': round(stats_sum['OR'] / len(player_df), 3) if len(player_df) != 0 else None
              , 'APG': round(stats_sum['A'] / len(player_df), 3) if len(player_df) != 0 else None
              , 'SPG': round(stats_sum['ST'] / len(player_df), 3) if len(player_df) != 0 else None
              , 'BPG': round(stats_sum['BL'] / len(player_df), 3) if len(player_df) != 0 else None
              , 'BPM': round(stats_sum['BL'] / stats_sum['MIN'], 3) if stats_sum['MIN'] != 0 else None
              , 'SPM': round(stats_sum['ST'] / stats_sum['MIN'], 3) if stats_sum['MIN'] != 0 else None
              , 'TPG': round(stats_sum['TO'] / len(player_df), 3) if len(player_df) != 0 else None
              , 'FPG': round(stats_sum['PF'] / len(player_df), 3) if len(player_df) != 0 else None
              , 'FPM': round(stats_sum['PF'] / stats_sum['MIN'], 3) if stats_sum['MIN'] != 0 else None
              , 'USAGE_RATE': round(stats_sum['USAGE_RATE'], 3) if stats_sum['USAGE_RATE'] is not None else None
              , **{k: round(v, 3) if isinstance(v, (int, float)) else v for k, v in stats_sum.items()}}
    
    # Save the calculated stats to cache
    save_to_cache(player_name, max_date, season_to_use, stats_sum)
    
    return stats_sum

def get_distinct_players():
    """Get all distinct player names from the loaded dataset."""
    return df['PLAYER \nFULL NAME'].unique()

def get_current_season():
    """Get the currently configured season year."""
    return SEASON_YEAR

def set_season_year(season_year):
    """
    Change the season year and reload the data.
    
    Args:
        season_year (str): Season year in format "YYYY-YYYY" (e.g., "2023-2024")
    """
    global SEASON_YEAR, df
    SEASON_YEAR = season_year
    df = load_player_data(SEASON_YEAR)
    print(f"Loaded data for season: {SEASON_YEAR}")