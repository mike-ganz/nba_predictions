import pandas as pd
import numpy as np
from get_team_city import get_team_city
from datetime import datetime, timedelta
import os
import json
import hashlib

# Configuration (will be set dynamically by config.settings)
SEASON_YEAR = "2023-2024"  # Default, overridden by global config
ROLLING_WINDOW = 10  # Number of games for rolling averages
MIN_GAMES = 3  # Minimum games required for stats

# Cache for loaded team data to avoid repeated file loading
_team_data_cache = {}

# Cache for calculated team stats (to avoid recalculating)
CACHE_DIR = 'data/cache/team_stats'
_stats_cache = {}

def load_team_data(season_year=None):
    """Load team boxscore data for the specified season year."""
    # Use global config if no season_year provided
    if season_year is None:
        from config.settings import config
        season_year = config.season_year
    """Load team boxscore data for the specified season year."""
    # Map season year to file path
    file_mapping = {
        "2020-2021": "data/team_boxscores/historical/2020-2021_NBA_Box_Score_Team-Stats.xlsx",
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

def ensure_cache_dir():
    """Ensure the cache directory exists."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

def get_cache_key(team_name, target_date, season_year):
    """Generate a cache key for team stats."""
    key_str = f"{team_name}_{target_date}_{season_year}"
    return hashlib.md5(key_str.encode()).hexdigest()

def save_to_cache(cache_key, stats_dict):
    """Save calculated stats to cache."""
    ensure_cache_dir()
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    with open(cache_file, 'w') as f:
        json.dump(stats_dict, f)
    _stats_cache[cache_key] = stats_dict

def load_from_cache(cache_key):
    """Load stats from cache if available."""
    # Check memory cache first
    if cache_key in _stats_cache:
        return _stats_cache[cache_key]
    
    # Check disk cache
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            stats_dict = json.load(f)
            _stats_cache[cache_key] = stats_dict  # Add to memory cache
            return stats_dict
    
    return None

def calculate_enhanced_rates(team_games_df):
    """
    Calculate enhanced team rate statistics from team boxscore data.
    
    Args:
        team_games_df: DataFrame with team boxscore data (sorted by date, recent last)
        
    Returns:
        dict: Dictionary with calculated rates including TOr and DRr
    """
    if len(team_games_df) == 0:
        return None
    
    # Calculate rates for each game
    rates_data = []
    
    for _, game in team_games_df.iterrows():
        # Basic validation
        fga = game.get('FGA', 0)
        fgm = game.get('FG', 0)
        three_pa = game.get('3PA', 0)
        fta = game.get('FTA', 0)
        off_reb = game.get('OR', 0)
        def_reb = game.get('DR', 0)
        assists = game.get('A', 0)
        turnovers = game.get('TO', 0)
        
        # Skip if critical data is missing
        if fga == 0:
            continue
        
        # 3-point attempt rate: 3PA / FGA
        three_par = three_pa / fga if fga > 0 else 0
        
        # Free throw rate: FTA / FGA
        ftr = fta / fga if fga > 0 else 0
        
        # Offensive rebound rate: OR / (OR + DR)
        # Note: In real calculation this should be OR / (OR + opponent_DR)
        # but we don't have opponent stats in single row, so approximate
        total_reb = off_reb + def_reb
        orr = off_reb / total_reb if total_reb > 0 else 0
        
        # Defensive rebound rate: DR / (DR + OR)
        # Note: Similar to ORr, ideally should be DR / (DR + opponent_OR)
        drr = def_reb / total_reb if total_reb > 0 else 0
        
        # Assist rate: AST / FGM
        astr = assists / fgm if fgm > 0 else 0
        
        # Turnover rate: TO / Possessions (approximate with FGA + 0.44*FTA + TO)
        # This is a per-possession turnover rate
        approx_possessions = fga + 0.44 * fta + turnovers
        tor = turnovers / approx_possessions if approx_possessions > 0 else 0
        
        rates_data.append({
            'OEFF': game.get('OEFF', 110.0),
            'DEFF': game.get('DEFF', 110.0),
            'PACE': game.get('PACE', 100.0),
            '3PAr': three_par,
            'FTr': ftr,
            'ORr': orr,
            'DRr': drr,
            'ASTr': astr,
            'TOr': tor
        })
    
    if len(rates_data) == 0:
        return None
    
    # Convert to DataFrame for easy averaging
    rates_df = pd.DataFrame(rates_data)
    
    # Calculate rolling averages (use min of ROLLING_WINDOW or all available games)
    window = min(ROLLING_WINDOW, len(rates_df))
    recent_games = rates_df.tail(window)
    
    return {
        'OEFF': round(recent_games['OEFF'].mean(), 2),
        'DEFF': round(recent_games['DEFF'].mean(), 2),
        'PACE': round(recent_games['PACE'].mean(), 2),
        '3PAr': round(recent_games['3PAr'].mean(), 3),
        'FTr': round(recent_games['FTr'].mean(), 3),
        'ORr': round(recent_games['ORr'].mean(), 3),
        'DRr': round(recent_games['DRr'].mean(), 3),
        'ASTr': round(recent_games['ASTr'].mean(), 3),
        'TOr': round(recent_games['TOr'].mean(), 3),
        'games_in_window': len(recent_games)
    }

def generate_team_stats(team_name, target_date=None, fallback_season=None, use_cache=True):
    """
    Calculate enhanced team stats (10 values) for a given team and date using rolling averages.
    
    Args:
        team_name (str): Name of the team
        target_date (str or None): Date in format 'YYYY-MM-DD', or None/empty for all games
        fallback_season (str or None): Season to use for fallback (e.g., "2022-2023")
        use_cache (bool): Whether to use cached results (default True)
    
    Returns:
        dict: Dictionary containing:
            - OEFF: Offensive efficiency (points per 100 possessions)
            - DEFF: Defensive efficiency (points allowed per 100 possessions)
            - PACE: Possessions per 48 minutes
            - 3PAr: 3-point attempt rate (3PA / FGA)
            - FTr: Free throw rate (FTA / FGA)
            - ORr: Offensive rebound rate
            - DRr: Defensive rebound rate
            - ASTr: Assist rate (AST / FGM)
            - TOr: Turnover rate (TO / Possessions)
            - REST_DAYS: Days since last game
            - TEAM_NAME, SEASON, GAMES_PLAYED (metadata)
    """
    global df
    
    # Load data if not already loaded
    if df is None:
        from config.settings import config
        season_to_use = fallback_season or config.season_year
        df = load_team_data(season_to_use)
        _team_data_cache[season_to_use] = df
    
    # Check cache first
    season_to_use = fallback_season or SEASON_YEAR
    if use_cache:
        cache_key = get_cache_key(team_name, target_date or "all", season_to_use)
        cached_stats = load_from_cache(cache_key)
        if cached_stats is not None:
            return cached_stats
    
    team_city = get_team_city(team_name)

    # Determine which dataset to use
    if fallback_season and fallback_season != SEASON_YEAR:
        # Try to get fallback season data from cache first
        if fallback_season in _team_data_cache:
            data_source = _team_data_cache[fallback_season]
            season_label = fallback_season
        else:
            # Load fallback season data and cache it
            try:
                fallback_df = load_team_data(fallback_season)
                _team_data_cache[fallback_season] = fallback_df
                data_source = fallback_df
                season_label = fallback_season
            except (ValueError, FileNotFoundError):
                # If fallback season not available, create reasonable default stats
                default_stats = {
                    'TEAM_NAME': team_name,
                    'SEASON': fallback_season, 
                    'GAMES_PLAYED': 0,
                    'OEFF': 110.0,
                    'DEFF': 110.0,
                    'PACE': 100.0,
                    '3PAr': 0.38,  # League average
                    'FTr': 0.22,   # League average
                    'ORr': 0.25,   # League average
                    'DRr': 0.75,   # League average (complement of ORr)
                    'ASTr': 0.65,  # League average
                    'TOr': 0.13,   # League average (~13% of possessions)
                    'REST_DAYS': 10,
                    'USING_PRIOR_SEASON': True,
                    'FALLBACK_REASON': f'No {fallback_season} team data available'
                }
                if use_cache:
                    save_to_cache(cache_key, default_stats)
                return default_stats
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

    # Return None if no games found
    if len(team_data) == 0:
        return None

    # Check if we have minimum games required
    if len(team_data) < MIN_GAMES:
        # Not enough games - return league averages
        default_stats = {
            'TEAM_NAME': team_name,
            'SEASON': season_label,
            'GAMES_PLAYED': len(team_data),
            'OEFF': 110.0,
            'DEFF': 110.0,
            'PACE': 100.0,
            '3PAr': 0.38,
            'FTr': 0.22,
            'ORr': 0.25,
            'DRr': 0.75,
            'ASTr': 0.65,
            'TOr': 0.13,
            'REST_DAYS': rest_days,
            'WARNING': f'Only {len(team_data)} games available (minimum {MIN_GAMES})'
        }
        if use_cache:
            save_to_cache(cache_key, default_stats)
        return default_stats

    # Sort by date to ensure proper rolling window
    team_data = team_data.sort_values('DATE')

    # Calculate enhanced rates using rolling window
    enhanced_rates = calculate_enhanced_rates(team_data)
    
    if enhanced_rates is None:
        return None

    # Build final stats dictionary
    stats = {
        'TEAM_NAME': team_name,
        'SEASON': season_label,
        'GAMES_PLAYED': len(team_data),
        'OEFF': enhanced_rates['OEFF'],
        'DEFF': enhanced_rates['DEFF'],
        'PACE': enhanced_rates['PACE'],
        '3PAr': enhanced_rates['3PAr'],
        'FTr': enhanced_rates['FTr'],
        'ORr': enhanced_rates['ORr'],
        'DRr': enhanced_rates['DRr'],
        'ASTr': enhanced_rates['ASTr'],
        'TOr': enhanced_rates['TOr'],
        'REST_DAYS': rest_days,
        'ROLLING_WINDOW_SIZE': enhanced_rates['games_in_window']
    }
    
    # Cache the result
    if use_cache:
        save_to_cache(cache_key, stats)
    
    return stats

def get_team_stats_array(team_name, target_date=None, fallback_season=None, use_cache=True):
    """
    Get team stats as a compact 10-value array for training data.
    
    Args:
        team_name (str): Name of the team
        target_date (str or None): Date in format 'YYYY-MM-DD'
        fallback_season (str or None): Season to use for fallback
        use_cache (bool): Whether to use cached results
    
    Returns:
        list: [OEFF, DEFF, PACE, 3PAr, FTr, ORr, DRr, ASTr, TOr, REST]
              Returns None if team stats cannot be calculated
    """
    stats = generate_team_stats(team_name, target_date, fallback_season, use_cache)
    
    if stats is None:
        return None
    
    # Convert REST_DAYS to standardized format (0, 1, 2, 3+)
    rest_days = stats.get('REST_DAYS')
    if rest_days is None:
        rest_val = 2  # Default to well-rested
    elif rest_days >= 3:
        rest_val = 3  # 3+ days
    else:
        rest_val = int(rest_days)
    
    return [
        stats['OEFF'],
        stats['DEFF'],
        stats['PACE'],
        stats['3PAr'],
        stats['FTr'],
        stats['ORr'],
        stats['DRr'],
        stats['ASTr'],
        stats['TOr'],
        rest_val
    ]

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
    global _team_data_cache, _stats_cache
    data_cache_size = len(_team_data_cache)
    stats_cache_size = len(_stats_cache)
    _team_data_cache.clear()
    _stats_cache.clear()
    print(f"🗑️ Cleared team data cache ({data_cache_size} seasons, {stats_cache_size} stats entries)")
    return data_cache_size + stats_cache_size

def get_cache_info():
    """Get information about cached team data."""
    return {
        'cached_seasons': list(_team_data_cache.keys()),
        'data_cache_size': len(_team_data_cache),
        'stats_cache_size': len(_stats_cache),
        'data_memory_usage_mb': sum(df.memory_usage(deep=True).sum() for df in _team_data_cache.values()) / 1024**2 if _team_data_cache else 0
    }

# Example usage and tests
if __name__ == "__main__":
    print("=" * 80)
    print("TESTING ENHANCED TEAM STATS")
    print("=" * 80)
    
    # Test 1: Get stats for Lakers mid-season
    print("\n📊 Test 1: Lakers on 2024-02-15 (mid-season)")
    stats = generate_team_stats("LA Lakers", "2024-02-15")
    if stats:
        print(f"   OEFF: {stats['OEFF']}")
        print(f"   DEFF: {stats['DEFF']}")
        print(f"   PACE: {stats['PACE']}")
        print(f"   3PAr: {stats['3PAr']}")
        print(f"   FTr: {stats['FTr']}")
        print(f"   ORr: {stats['ORr']}")
        print(f"   DRr: {stats['DRr']}")
        print(f"   ASTr: {stats['ASTr']}")
        print(f"   TOr: {stats['TOr']}")
        print(f"   REST: {stats['REST_DAYS']} days")
        print(f"   Games in window: {stats['ROLLING_WINDOW_SIZE']}")
        
        # Test compact array format
        print("\n   Compact array format (10 values):")
        array = get_team_stats_array("LA Lakers", "2024-02-15")
        print(f"   {array}")
    
    # Test 2: Get stats for Celtics late season
    print("\n📊 Test 2: Boston Celtics on 2024-04-01 (late season)")
    stats = generate_team_stats("Boston Celtics", "2024-04-01")
    if stats:
        print(f"   3PAr: {stats['3PAr']} (should be high - modern offense)")
        print(f"   ASTr: {stats['ASTr']}")
        array = get_team_stats_array("Boston Celtics", "2024-04-01")
        print(f"   Compact: {array}")
    
    # Test 3: Early season (should handle gracefully)
    print("\n📊 Test 3: Nuggets on 2023-11-01 (early season)")
    stats = generate_team_stats("Denver Nuggets", "2023-11-01")
    if stats:
        print(f"   Games played: {stats['GAMES_PLAYED']}")
        print(f"   Rolling window: {stats.get('ROLLING_WINDOW_SIZE', 'N/A')}")
        if 'WARNING' in stats:
            print(f"   ⚠️  {stats['WARNING']}")
    
    # Cache info
    print("\n💾 Cache info:")
    info = get_cache_info()
    print(f"   Stats cached: {info['stats_cache_size']}")
    print(f"   Seasons loaded: {info['cached_seasons']}")
    
    print("\n✅ All tests complete!")
    print("=" * 80)