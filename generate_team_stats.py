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

SEASON_FALLBACK = {
    "2024-2025": "2023-2024",
    "2023-2024": "2022-2023",
    "2022-2023": "2021-2022",
    "2021-2022": "2020-2021",
}

# Cache for loaded team data to avoid repeated file loading
_team_data_cache = {}

# Cache for calculated team stats (to avoid recalculating)
CACHE_DIR = 'data/cache/team_stats'
_stats_cache = {}


def _load_season_frame(season_year: str) -> pd.DataFrame:
    """Return the team boxscore dataframe for the requested season."""

    global df

    if season_year in _team_data_cache:
        return _team_data_cache[season_year]

    frame = load_team_data(season_year)
    _team_data_cache[season_year] = frame

    # Maintain the legacy global reference for the default season
    if df is None and season_year == SEASON_YEAR:
        df = frame

    return frame


def _compute_rest_days(team_games_df: pd.DataFrame, target_date: pd.Timestamp | None) -> float | None:
    """Compute rest days using the current-season schedule prior to target_date.
    
    Returns the number of rest days (not calendar days). For example:
    - Last game: Nov 5, Current game: Nov 7 → 1 rest day (Nov 6)
    - Last game: Nov 6, Current game: Nov 7 → 0 rest days (back-to-back)
    """

    if target_date is None or team_games_df.empty:
        return None

    last_game_date = team_games_df['DATE'].max()
    if pd.isna(last_game_date):
        return None

    # Calculate rest days = calendar days - 1
    # (calendar days includes both game days, rest days does not)
    calendar_days = (target_date - last_game_date).days
    return float(max(0, calendar_days - 1))


def _assemble_stats(
    team_name: str,
    rate_payload: dict,
    *,
    season_used: str,
    source: str,
    rest_days: float | None,
    games_played: int,
    current_games: int,
    primary_season: str,
) -> dict:
    """Normalize the stats dictionary returned to callers."""

    stats = {
        'TEAM_NAME': team_name,
        'SEASON': season_used,
        'PRIMARY_SEASON': primary_season,
        'SEASON_USED': season_used,
        'STAT_SOURCE': source,
        'GAMES_PLAYED': games_played,
        'GAMES_CURRENT_SEASON': current_games,
        'REST_DAYS': rest_days,
        'ROLLING_WINDOW_SIZE': rate_payload.get('games_in_window', 0),
        'OEFF': rate_payload['OEFF'],
        'DEFF': rate_payload['DEFF'],
        'PACE': rate_payload['PACE'],
        '3PAr': rate_payload['3PAr'],
        'FTr': rate_payload['FTr'],
        'ORr': rate_payload['ORr'],
        'DRr': rate_payload['DRr'],
        'ASTr': rate_payload['ASTr'],
        'TOr': rate_payload['TOr'],
    }

    return stats

def load_team_data(season_year=None):
    """Load team boxscore data for the specified season year."""
    # Use global config if no season_year provided
    if season_year is None:
        from config.settings import config
        season_year = config.season_year
    
    # Map season year to file path (for historical seasons)
    file_mapping = {
        "2020-2021": "data/team_boxscores/historical/2020-2021_NBA_Box_Score_Team-Stats.xlsx",
        "2021-2022": "data/team_boxscores/historical/2021-2022_NBA_Box_Score_Team-Stats.xlsx",
        "2022-2023": "data/team_boxscores/historical/2022-2023_NBA_Box_Score_Team-Stats.xlsx",
        "2023-2024": "data/team_boxscores/historical/2023-2024_NBA_Box_Score_Team-Stats.xlsx",
        "2024-2025": "data/team_boxscores/historical/2024-2025_NBA_Box_Score_Team-Stats.xlsx"
    }
    
    # Check if it's a historical season
    if season_year in file_mapping:
        file_path = file_mapping[season_year]
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.replace('\n', ' ')
        df.columns = df.columns.str.strip()
        df['DATE'] = pd.to_datetime(df['DATE'])
        return df
    
    # For current season (e.g., 2025-2026), look in current/ directory
    # Find the most recent file
    from pathlib import Path
    import re
    
    current_dir = Path("data/team_boxscores/current")
    if current_dir.exists():
        files = list(current_dir.glob("*.xlsx"))
        if files:
            # Extract dates from filenames and find most recent
            dated_files = []
            for f in files:
                match = re.match(r'(\d{1,2})-(\d{1,2})-(\d{4})', f.name)
                if match:
                    month, day, year = match.groups()
                    try:
                        date = datetime(int(year), int(month), int(day))
                        dated_files.append((date, f))
                    except ValueError:
                        continue
            
            if dated_files:
                dated_files.sort(reverse=True)
                file_path = str(dated_files[0][1])
                
                df = pd.read_excel(file_path)
                df.columns = df.columns.str.replace('\n', ' ')
                df.columns = df.columns.str.strip()
                df['DATE'] = pd.to_datetime(df['DATE'])
                return df
    
    raise ValueError(f"Season year {season_year} not supported. Available options: {list(file_mapping.keys())} or place current season data in data/team_boxscores/current/")

# Data will be loaded dynamically when needed based on global config
df: pd.DataFrame | None = None

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
    Calculate enhanced team rate statistics from team boxscore data (VECTORIZED).
    
    Args:
        team_games_df: DataFrame with team boxscore data (sorted by date, recent last)
        
    Returns:
        dict: Dictionary with calculated rates including TOr and DRr
    """
    if len(team_games_df) == 0:
        return None
    
    # PERFORMANCE FIX: Vectorize all calculations instead of iterrows()
    # Filter out rows with missing FGA (critical data)
    df = team_games_df[team_games_df['FGA'] > 0].copy()
    
    if len(df) == 0:
        return None
    
    # Vectorized calculations (100x faster than loops!)
    df['3PAr'] = df['3PA'] / df['FGA']
    df['FTr'] = df['FTA'] / df['FGA']
    
    # Rebound rates
    total_reb = df['OR'] + df['DR']
    df['ORr'] = df['OR'] / total_reb.replace(0, 1)  # Avoid division by zero
    df['DRr'] = df['DR'] / total_reb.replace(0, 1)
    
    # Assist rate: AST / FGM
    df['ASTr'] = df['A'] / df['FG'].replace(0, 1)
    
    # Turnover rate: TO / Possessions
    approx_possessions = df['FGA'] + 0.44 * df['FTA'] + df['TO']
    df['TOr'] = df['TO'] / approx_possessions.replace(0, 1)
    
    # Calculate rolling averages (use min of ROLLING_WINDOW or all available games)
    window = min(ROLLING_WINDOW, len(df))
    recent_games = df.tail(window)
    
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
    
    primary_season = fallback_season or SEASON_YEAR

    # Pull the current-season frame first
    current_frame = _load_season_frame(primary_season)
    team_city = get_team_city(team_name)
    target_timestamp = pd.to_datetime(target_date) if target_date else None

    cache_key = get_cache_key(team_name, target_date or "all", primary_season)
    if use_cache:
        cached_stats = load_from_cache(cache_key)
        if cached_stats is not None:
            return cached_stats

    # Subset to games prior to target_date (or entire season if None)
    if target_timestamp is None:
        current_mask = current_frame['TEAM'] == team_city
    else:
        current_mask = (current_frame['TEAM'] == team_city) & (current_frame['DATE'] < target_timestamp)

    current_games = current_frame[current_mask].sort_values('DATE')
    current_count = len(current_games)

    rest_days = _compute_rest_days(current_games, target_timestamp)

    if current_count >= MIN_GAMES:
        rates = calculate_enhanced_rates(current_games)
        if rates is not None:
            stats = _assemble_stats(
                team_name,
                rates,
                season_used=primary_season,
                source="current",
                rest_days=rest_days,
                games_played=current_count,
                current_games=current_count,
                primary_season=primary_season,
            )
            if use_cache:
                save_to_cache(cache_key, stats)
            return stats

    # Not enough games – attempt prior-season fallback
    prior_season = SEASON_YEAR
    if primary_season in SEASON_FALLBACK:
        prior_season = SEASON_FALLBACK[primary_season]
    elif primary_season != SEASON_YEAR:
        prior_season = primary_season

    fallback_frame = _load_season_frame(prior_season)
    fallback_games = fallback_frame[fallback_frame['TEAM'] == team_city].sort_values('DATE')
    fallback_count = len(fallback_games)

    if fallback_count >= MIN_GAMES:
        fallback_rates = calculate_enhanced_rates(fallback_games)
        if fallback_rates is not None:
            stats = _assemble_stats(
                team_name,
                fallback_rates,
                season_used=prior_season,
                source="prior_season",
                rest_days=rest_days,
                games_played=fallback_count,
                current_games=current_count,
                primary_season=primary_season,
            )
            if use_cache:
                save_to_cache(cache_key, stats)
            return stats

    # Final fallback – use league averages but keep metadata useful
    league_rates = {
        'OEFF': 110.0,
        'DEFF': 110.0,
        'PACE': 100.0,
        '3PAr': 0.38,
        'FTr': 0.22,
        'ORr': 0.25,
        'DRr': 0.75,
        'ASTr': 0.65,
        'TOr': 0.13,
        'games_in_window': 0,
    }

    stats = _assemble_stats(
        team_name,
        league_rates,
        season_used=prior_season,
        source="league_average",
        rest_days=rest_days,
        games_played=fallback_count,
        current_games=current_count,
        primary_season=primary_season,
    )

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