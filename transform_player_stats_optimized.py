import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import hashlib
from concurrent.futures import ProcessPoolExecutor
import time

# Configuration (will be set dynamically by config.settings)
SEASON_YEAR = "2023-2024"  # Default, overridden by global config
CACHE_DIR = 'data/cache/player_stats'

def load_player_data(season_year=None):
    """Load player boxscore data for the specified season year."""
    # Use global config if no season_year provided
    if season_year is None:
        from config.settings import config
        season_year = config.season_year
    """Load player boxscore data for the specified season year."""
    file_mapping = {
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
    
    print(f"📊 Loading {file_path}...")
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    
    # 🚀 OPTIMIZATION: Pre-process dates once
    print("📅 Pre-processing dates...")
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    # 🚀 OPTIMIZATION: Sort by player and date for efficient processing
    print("🔄 Sorting data for optimal processing...")
    df = df.sort_values(['PLAYER \nFULL NAME', 'DATE'])
    
    print(f"✅ Loaded and optimized {len(df)} records")
    return df

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

def calculate_advanced_stats(stats_row):
    """Calculate advanced stats from basic stats."""
    result = stats_row.copy()
    
    # Avoid division by zero
    def safe_divide(numerator, denominator, default=None):
        return numerator / denominator if denominator != 0 else default
    
    # Calculate advanced metrics
    result['2P'] = round(stats_row['FG'] - stats_row['3P'], 3)
    result['MPG'] = safe_divide(stats_row['MIN'], stats_row['GP'])
    result['3P%'] = safe_divide(stats_row['3P'], stats_row['3PA'])
    result['FT%'] = safe_divide(stats_row['FT'], stats_row['FTA'])
    result['eFG%'] = safe_divide(stats_row['FG'] + 0.5 * stats_row['3P'], stats_row['FGA'])
    result['TS%'] = safe_divide(stats_row['PTS'], 2 * (stats_row['FGA'] + 0.44 * stats_row['FTA']))
    result['3PR'] = safe_divide(stats_row['3PA'], stats_row['FGA'])
    result['FTR'] = safe_divide(stats_row['FTA'], stats_row['FGA'] + stats_row['FTA'])
    result['PFFT'] = safe_divide(stats_row['FT'], stats_row['PTS'])
    result['PPG'] = safe_divide(stats_row['PTS'], stats_row['GP'])
    result['RPG'] = safe_divide(stats_row['OR'] + stats_row['DR'], stats_row['GP'])
    result['DRPG'] = safe_divide(stats_row['DR'], stats_row['GP'])
    result['ORPG'] = safe_divide(stats_row['OR'], stats_row['GP'])
    result['APG'] = safe_divide(stats_row['A'], stats_row['GP'])
    result['SPG'] = safe_divide(stats_row['ST'], stats_row['GP'])
    result['BPG'] = safe_divide(stats_row['BL'], stats_row['GP'])
    result['BPM'] = safe_divide(stats_row['BL'], stats_row['MIN'])
    result['SPM'] = safe_divide(stats_row['ST'], stats_row['MIN'])
    result['TPG'] = safe_divide(stats_row['TO'], stats_row['GP'])
    result['FPG'] = safe_divide(stats_row['PF'], stats_row['GP'])
    result['FPM'] = safe_divide(stats_row['PF'], stats_row['MIN'])
    
    # Round all numeric values
    for key, value in result.items():
        if isinstance(value, (int, float)) and not pd.isna(value):
            result[key] = round(value, 3)
    
    return result

def build_all_player_stats_vectorized(df, target_dates):
    """
    🚀 VECTORIZED: Build stats for ALL players for ALL target dates efficiently.
    
    This replaces thousands of individual calculations with a few vectorized operations.
    """
    print(f"🚀 Starting vectorized processing for {len(target_dates)} dates...")
    start_time = time.time()
    
    # Numeric columns for aggregation
    numeric_cols = ['MIN', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA', 'OR', 'DR', 'TOT', 'A', 'PF', 'ST', 'TO', 'BL', 'PTS']
    
    all_results = []
    
    for i, target_date in enumerate(target_dates):
        print(f"📅 Processing date {i+1}/{len(target_dates)}: {target_date}")
        
        # 🚀 VECTORIZED: Filter all data up to target date in one operation  
        # 🔧 FIX: Use < instead of <= to match individual method behavior
        filtered_df = df[df['DATE'] < pd.to_datetime(target_date)]
        
        # 🚀 VECTORIZED: Group by player and sum all stats at once
        player_stats = filtered_df.groupby('PLAYER \nFULL NAME').agg({
            **{col: 'sum' for col in numeric_cols},
            'USAGE \nRATE (%)': 'mean',  # Average usage rate (it's already a percentage)
            'DATE': 'count'  # Count games played
        }).rename(columns={'DATE': 'GP', 'USAGE \nRATE (%)': 'USAGE_RATE'})
        
        # 🚀 VECTORIZED: Calculate advanced stats for all players at once
        for player_name in player_stats.index:
            stats_row = player_stats.loc[player_name]
            stats_dict = stats_row.to_dict()
            stats_dict['PLAYER_NAME'] = player_name
            
            # Calculate advanced stats
            advanced_stats = calculate_advanced_stats(stats_dict)
            
            # Store result
            all_results.append({
                'player_name': player_name,
                'max_date': target_date,
                'stats': advanced_stats
            })
    
    elapsed = time.time() - start_time
    print(f"✅ Vectorized processing complete! {elapsed:.2f}s for {len(all_results)} player-date combinations")
    print(f"⚡ Speed: {len(all_results)/elapsed:.0f} calculations per second")
    
    return all_results

def batch_save_to_cache(results, current_season):
    """Save all results to cache efficiently."""
    print(f"💾 Saving {len(results)} cache files...")
    saved_count = 0
    
    for result in results:
        save_to_cache(
            result['player_name'], 
            result['max_date'], 
            current_season, 
            result['stats']
        )
        saved_count += 1
        
        if saved_count % 100 == 0:
            print(f"💾 Saved {saved_count}/{len(results)} cache files...")
    
    print(f"✅ Saved {saved_count} cache files!")

# Data will be loaded dynamically when needed based on global config
# Use a dictionary to cache multiple seasons instead of single global df
_season_data_cache = {}

# Backwards compatibility functions
def calculate_player_stats(player_name, max_date=None, current_season=None):
    """Backwards compatible function - tries cache first, falls back to old method."""
    if current_season:
        season_to_use = current_season
    else:
        from config.settings import config
        season_to_use = config.season_year
    
    # Try cache first
    cached_data = load_from_cache(player_name, max_date, season_to_use)
    if cached_data:
        return cached_data
    
    # Fall back to single calculation (slower)
    print(f"⚠️ Cache miss for {player_name} on {max_date} - calculating individually")
    
    # Load data using season cache to avoid repeated loading
    global _season_data_cache
    if season_to_use not in _season_data_cache:
        print(f"📊 Loading {season_to_use} data (first time for this session)...")
        df = load_player_data(season_to_use)
        _season_data_cache[season_to_use] = df
        print(f"💾 Cached {season_to_use} data for future use")
    else:
        df = _season_data_cache[season_to_use]
        # print(f"✅ Using cached {season_to_use} data")
    
    player_df = df[df['PLAYER \nFULL NAME'] == player_name]
    
    if len(player_df) == 0:
        return None
        
    # Handle special full season cache key
    if max_date is not None and not str(max_date).endswith('_FULL_SEASON'):
        # 🔧 FIX: Use < instead of <= to match individual method behavior  
        player_df = player_df[player_df['DATE'] < pd.to_datetime(max_date)]
    # If max_date is None or ends with '_FULL_SEASON', use all games (no date filtering)
    
    # Check if we have sufficient games after date filtering
    games_played = len(player_df)
    min_games_threshold = 5  # Minimum games needed
    
    # If insufficient games, try fallback to previous season (avoid infinite recursion)
    # Don't fallback if this is already a fallback call (indicated by _FULL_SEASON suffix)
    if (games_played < min_games_threshold and max_date is not None and 
        not str(max_date).endswith('_FULL_SEASON')):  # Only fallback if we have a date filter and not already a fallback
        # Calculate previous season correctly
        try:
            year_parts = season_to_use.split('-')
            current_start_year = int(year_parts[0])
            current_end_year = int(year_parts[1])
            prev_season = f"{current_start_year-1}-{current_end_year-1}"
            
            # Avoid infinite recursion - only go back to certain known seasons
            if prev_season in ["2022-2023", "2021-2022", "2020-2021", "2019-2020"]:
                print(f"🔄 {player_name}: Only {games_played} games in {season_to_use}, falling back to {prev_season}")
                
                # Try to get previous season data WITHOUT recursion (use full season stats)
                # Use special cache key for full season to avoid confusion with partial season
                fallback_stats = calculate_player_stats(player_name, f"{prev_season}_FULL_SEASON", prev_season)
                if fallback_stats and fallback_stats.get('GP', 0) >= min_games_threshold:
                    # Add metadata to indicate fallback was used
                    fallback_stats['FALLBACK_FROM'] = season_to_use
                    fallback_stats['ORIGINAL_GAMES'] = games_played
                    return fallback_stats
                else:
                    print(f"⚠️ {player_name}: No sufficient fallback data in {prev_season} either")
            else:
                print(f"⚠️ {player_name}: Preventing infinite recursion, skipping fallback to {prev_season}")
        except Exception as e:
            import traceback
            print(f"⚠️ {player_name}: Fallback calculation failed: {e}")
            print(f"🔎 Full traceback: {traceback.format_exc()[:200]}...")  # First 200 chars of traceback
    
    numeric_cols = ['MIN', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA', 'OR', 'DR', 'TOT', 'A', 'PF', 'ST', 'TO', 'BL', 'PTS']
    
    # Only use columns that actually exist in the dataframe
    available_cols = [col for col in numeric_cols if col in player_df.columns]
    if len(available_cols) != len(numeric_cols):
        missing_cols = set(numeric_cols) - set(available_cols)
        print(f"⚠️ {player_name}: Missing columns {missing_cols} in {season_to_use} data")
    
    stats_sum = player_df[available_cols].sum().to_dict()
    
    # Fill in missing columns with 0
    for col in numeric_cols:
        if col not in stats_sum:
            stats_sum[col] = 0
    stats_sum['PLAYER_NAME'] = player_name
    stats_sum['GP'] = games_played
    # Calculate average for percentage-based stats (usage rate)
    usage_col = 'USAGE \nRATE (%)'
    if usage_col in player_df.columns:
        stats_sum['USAGE_RATE'] = player_df[usage_col].mean()
    else:
        stats_sum['USAGE_RATE'] = None
    
    advanced_stats = calculate_advanced_stats(stats_sum)
    
    # Save to cache for future use
    save_to_cache(player_name, max_date, season_to_use, advanced_stats)
    
    return advanced_stats

def get_distinct_players():
    """Get all distinct player names from the loaded dataset."""
    from config.settings import config
    season_year = config.season_year
    
    global _season_data_cache
    if season_year not in _season_data_cache:
        print(f"📊 Loading {season_year} data to get distinct players...")
        df = load_player_data(season_year)
        _season_data_cache[season_year] = df
    else:
        df = _season_data_cache[season_year]
    
    return df['PLAYER \nFULL NAME'].unique()

def get_current_season():
    """Get the currently configured season year."""
    return SEASON_YEAR

def set_season_year(season_year):
    """Change the season year and load the data into cache if not already loaded."""
    global SEASON_YEAR, _season_data_cache
    SEASON_YEAR = season_year
    
    if season_year not in _season_data_cache:
        df = load_player_data(season_year)
        _season_data_cache[season_year] = df
        print(f"Loaded and cached data for season: {season_year}")
    else:
        print(f"Season {season_year} already cached")

# 🚀 NEW: Batch cache builder
def build_cache_for_date_range(start_date, end_date, date_interval_days=7):
    """
    Build cache for all players across a date range efficiently.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD) 
        date_interval_days: Days between each cache point
    """
    print(f"🚀 Building cache from {start_date} to {end_date} (every {date_interval_days} days)")
    
    # Generate target dates
    date_range = pd.date_range(start=start_date, end=end_date, freq=f'{date_interval_days}D')
    target_dates = [date.strftime('%Y-%m-%d') for date in date_range]
    
    print(f"📅 Target dates ({len(target_dates)}): {target_dates[:3]} ... {target_dates[-3:]}")
    
    # Get current season data
    from config.settings import config
    current_season = config.season_year
    
    global _season_data_cache
    if current_season not in _season_data_cache:
        df = load_player_data(current_season)
        _season_data_cache[current_season] = df
    else:
        df = _season_data_cache[current_season]
    
    # 🚀 VECTORIZED: Calculate all stats in batch
    all_results = build_all_player_stats_vectorized(df, target_dates)
    
    # 💾 Save all to cache
    batch_save_to_cache(all_results, SEASON_YEAR)
    
    print(f"🎉 Cache building complete! {len(all_results)} total cache files created.")
    
    return len(all_results)

if __name__ == "__main__":
    # Example: Build cache for October 2023
    cache_count = build_cache_for_date_range("2023-10-01", "2023-10-31", date_interval_days=3)
    print(f"Built {cache_count} cache entries!")
