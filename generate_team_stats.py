import pandas as pd
from get_team_city import get_team_city
from datetime import datetime, timedelta
import os

# Configuration (will be set dynamically by config.settings)
SEASON_YEAR = "2023-2024"  # Default, overridden by global config

# Cache for loaded team data to avoid repeated file loading
_team_data_cache = {}

def load_team_data(season_year=None):
    """Load team boxscore data for the specified season year."""
    # Use global config if no season_year provided
    if season_year is None:
        from config.settings import config
        season_year = config.season_year
    """Load team boxscore data for the specified season year."""
    # Map season year to file path
    file_mapping = {
        "2021-2022": "data/team_boxscores/historical/2021-2022_NBA_Box_Score_Team-Stats.xlsx",
        "2022-2023": "data/team_boxscores/historical/2022-2023_NBA_Box_Score_Team-Stats.xlsx",
        "2023-2024": "data/team_boxscores/historical/2023-2024_NBA_Box_Score_Team-Stats.xlsx",
        "2024-2025": "data/team_boxscores/historical/2024-2025_NBA_Box_Score_Team-Stats.xlsx"
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

# Data will be loaded dynamically when needed based on global config
df = None
_team_data_cache = {}

def generate_team_stats(team_name, target_date=None, fallback_season=None):
    """Generate team stats for a given team and date."""
    global df
    
    # Load data if not already loaded
    if df is None:
        from config.settings import config
        season_to_use = fallback_season or config.season_year
        df = load_team_data(season_to_use)
        _team_data_cache[season_to_use] = df
    """
    Calculate average PACE, OEFF, and DEFF for a team up to a specific date.
    Uses the configured season data, or fallback season if specified.
    
    Args:
        team_name (str): Name of the team
        target_date (str or None): Date in format 'YYYY-MM-DD', or None/empty for all games
        fallback_season (str or None): Season to use for fallback (e.g., "2022-2023")
    
    Returns:
        dict: Dictionary containing average PACE, OEFF, and DEFF
    """
    team_city = get_team_city(team_name)

    # Determine which dataset to use
    if fallback_season and fallback_season != SEASON_YEAR:
        # Try to get fallback season data from cache first
        if fallback_season in _team_data_cache:
            data_source = _team_data_cache[fallback_season]
            season_label = fallback_season
            # print(f"✅ Using cached {fallback_season} data for {team_name}")
        else:
            # Load fallback season data and cache it
            try:
                fallback_df = load_team_data(fallback_season)
                _team_data_cache[fallback_season] = fallback_df  # Cache for future use
                data_source = fallback_df
                season_label = fallback_season
                print(f"📁 Loaded and cached {fallback_season} data for future use")
            except (ValueError, FileNotFoundError):
                # If fallback season not available, create reasonable default stats
                print(f"🔄 Fallback season {fallback_season} not available, using estimated defaults for early season")
                return {
                    'TEAM_NAME': team_name,
                    'SEASON': fallback_season, 
                    'GAMES_PLAYED': 0,
                    'OEFF': 110.0,  # League average estimates
                    'DEFF': 110.0,
                    'PACE': 100.0,
                    'REST_DAYS': 10,  # Well-rested at season start
                    'USING_PRIOR_SEASON': True,
                    'FALLBACK_REASON': f'No {fallback_season} team data available - using league averages'
                }
    else:
        data_source = df
        season_label = SEASON_YEAR

    # If target_date is empty or None, use all games from the data source
    if not target_date:
        mask = (data_source['TEAM'] == team_city)
        team_data = data_source[mask]
        last_game_date = team_data['DATE'].max() if len(team_data) > 0 else None
        rest_days = None
    else:
        # Convert target_date to datetime
        target_date_dt = pd.to_datetime(target_date)
        mask = (data_source['TEAM'] == team_city) & (data_source['DATE'] < target_date_dt)
        team_data = data_source[mask]
        last_game_date = team_data['DATE'].max() if len(team_data) > 0 else None
        rest_days = (target_date_dt - last_game_date).days if last_game_date is not None else None

    if len(team_data) == 0:
        return None

    stats = {
        'TEAM_NAME': team_name,
        'SEASON': season_label,
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

def clear_team_data_cache():
    """Clear the cached team data to free memory."""
    global _team_data_cache
    cache_size = len(_team_data_cache)
    _team_data_cache.clear()
    print(f"🗑️ Cleared team data cache ({cache_size} seasons)")
    return cache_size

def get_cache_info():
    """Get information about cached team data."""
    return {
        'cached_seasons': list(_team_data_cache.keys()),
        'cache_size': len(_team_data_cache),
        'memory_usage_mb': sum(df.memory_usage(deep=True).sum() for df in _team_data_cache.values()) / 1024**2
    }

# Example usage (uncomment to test):
# stats = generate_team_stats("New York Knicks", "2024-12-01")
# print(stats)