import pandas as pd
from get_team_city import get_team_city_LA
from datetime import datetime, timedelta
import os

# Configuration - Change this to the desired season year
SEASON_YEAR = "2023-2024"  # Can be "2022-2023" or "2023-2024"

def load_team_data(season_year):
    """Load team boxscore data for the specified season year."""
    # Map season year to file path
    file_mapping = {
        "2022-2023": "data/team_boxscores/historical/2022-2023_NBA_Box_Score_Team-Stats.xlsx",
        "2023-2024": "data/team_boxscores/historical/2023-2024_NBA_Box_Score_Team-Stats.xlsx"
    }
    
    if season_year not in file_mapping:
        raise ValueError(f"Season year {season_year} not supported. Available options: {list(file_mapping.keys())}")
    
    file_path = file_mapping[season_year]
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.replace('\n', ' ')
    df.columns = df.columns.str.strip()
    df['DATE'] = pd.to_datetime(df['DATE'])
    return df

# Load data for the configured season year
df = load_team_data(SEASON_YEAR)

def generate_team_stats(team_name, target_date=None):
    """
    Calculate average PACE, OEFF, and DEFF for a team up to a specific date.
    Uses the configured season data.
    
    Args:
        team_name (str): Name of the team
        target_date (str or None): Date in format 'YYYY-MM-DD', or None/empty for all games
    
    Returns:
        dict: Dictionary containing average PACE, OEFF, and DEFF
    """
    team_city = get_team_city_LA(team_name)

    # If target_date is empty or None, use all games
    if not target_date:
        mask = (df['TEAM'] == team_city)
        team_data = df[mask]
        last_game_date = team_data['DATE'].max() if len(team_data) > 0 else None
        rest_days = None
    else:
        # Convert target_date to datetime
        target_date_dt = pd.to_datetime(target_date)
        mask = (df['TEAM'] == team_city) & (df['DATE'] < target_date_dt)
        team_data = df[mask]
        last_game_date = team_data['DATE'].max() if len(team_data) > 0 else None
        rest_days = (target_date_dt - last_game_date).days if last_game_date is not None else None

    if len(team_data) == 0:
        return None

    stats = {
        'TEAM_NAME': team_name,
        'SEASON': SEASON_YEAR,
        'OEFF': round(team_data['OEFF'].median(), 1),
        'DEFF': round(team_data['DEFF'].median(), 1),
        'PACE': round(team_data['PACE'].median(), 1),
        'GAMES_PLAYED': len(team_data),
        'REST_DAYS': rest_days
    }
    return stats

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
    df = load_team_data(SEASON_YEAR)
    print(f"Loaded team data for season: {SEASON_YEAR}")

def get_available_teams():
    """Get all unique team names from the loaded dataset."""
    return df['TEAM'].unique()

# Example usage (uncomment to test):
# stats = generate_team_stats("New York Knicks", "2024-12-01")
# print(stats)