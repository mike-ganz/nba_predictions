"""
League-Relative Feature Normalization

Calculates season-to-date league averages and normalizes team features
to be relative to contemporary league performance (no look-ahead bias).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import json
import math

# Cache for league averages by (season, date)
_league_avg_cache: Dict[Tuple[str, str], Dict[str, float]] = {}

# Cache for loaded DataFrames (much faster than re-reading Excel files!)
_df_cache: Dict[str, pd.DataFrame] = {}

# PERFORMANCE: Cache for datetime conversions (avoid repeated parsing)
_datetime_cache: Dict[str, pd.Timestamp] = {}


def _load_team_boxscores_for_season(season: str) -> pd.DataFrame:
    """Load team boxscore data for a given season (with caching)"""
    # Check cache first
    if season in _df_cache:
        return _df_cache[season]
    
    season_map = {
        '2021-2022': '2021-2022_NBA_Box_Score_Team-Stats.xlsx',
        '2022-2023': '2022-2023_NBA_Box_Score_Team-Stats.xlsx',
        '2023-2024': '2023-2024_NBA_Box_Score_Team-Stats.xlsx',
        '2024-2025': '2024-2025_NBA_Box_Score_Team-Stats.xlsx',
    }
    
    df = pd.DataFrame()
    
    # Check historical seasons first
    if season in season_map:
        # Files are in historical directory
        file_path = Path('data') / 'team_boxscores' / 'historical' / season_map[season]
        
        if file_path.exists():
            print(f"  Loading {season} data from Excel... ", end='', flush=True)
            df = pd.read_excel(file_path)
            df['DATE'] = pd.to_datetime(df['DATE'])
            print(f"OK ({len(df)} games)")
    
    # For current season (2025-2026), look in current/ directory
    elif season == '2025-2026':
        current_dir = Path('data') / 'team_boxscores' / 'current'
        if current_dir.exists():
            # Find most recent file by sorting filenames (date prefix like 10-31-2025)
            xlsx_files = list(current_dir.glob('*.xlsx'))
            if xlsx_files:
                # Sort by filename (descending) to get most recent date
                xlsx_files_sorted = sorted(xlsx_files, key=lambda x: x.name, reverse=True)
                file_path = xlsx_files_sorted[0]
                print(f"  Loading {season} data from {file_path.name}... ", end='', flush=True)
                df = pd.read_excel(file_path)
                df['DATE'] = pd.to_datetime(df['DATE'])
                print(f"OK ({len(df)} games)")
    
    # Cache the loaded DataFrame
    if not df.empty:
        _df_cache[season] = df
    
    return df


def calculate_league_averages(season: str, target_date: str, min_games: int = 20) -> Optional[Dict[str, float]]:
    """
    Calculate league-wide averages for a given season up to (but not including) target_date.
    
    Args:
        season: Season string (e.g., '2023-2024')
        target_date: Date string (e.g., '2023-12-15')
        min_games: Minimum number of games required to calculate averages
        
    Returns:
        Dictionary of league averages, or None if insufficient data
    """
    # Check cache first
    cache_key = (season, target_date)
    if cache_key in _league_avg_cache:
        return _league_avg_cache[cache_key]
    
    # Load season data
    df = _load_team_boxscores_for_season(season)
    
    if df.empty:
        return None
    
    # PERFORMANCE FIX #2: Cache datetime conversions (avoid repeated parsing)
    if target_date not in _datetime_cache:
        _datetime_cache[target_date] = pd.to_datetime(target_date)
    target_dt = _datetime_cache[target_date]
    
    # PERFORMANCE FIX #1: Remove .copy() - we're only reading, not modifying!
    df_before = df[df['DATE'] < target_dt]
    
    # Check if we have enough data
    if len(df_before) < min_games:
        # Fall back to prior season's final averages
        prior_season_map = {
            '2022-2023': '2021-2022',
            '2023-2024': '2022-2023',
            '2024-2025': '2023-2024',
            '2025-2026': '2024-2025',
        }
        
        if season in prior_season_map:
            prior_season = prior_season_map[season]
            prior_df = _load_team_boxscores_for_season(prior_season)
            
            if not prior_df.empty:
                # Use entire prior season as baseline
                df_before = prior_df
            else:
                # Use hardcoded league averages
                return {
                    'OEFF': 110.0, 'OEFF_std': 5.0,
                    'DEFF': 110.0, 'DEFF_std': 5.0,
                    'PACE': 100.0, 'PACE_std': 2.0,
                    '3PAr': 0.40, '3PAr_mean': 0.40, '3PAr_std': 0.05,
                    'FTr': 0.25, 'FTr_mean': 0.25, 'FTr_std': 0.05,
                    'ORr': 0.25, 'ORr_mean': 0.25, 'ORr_std': 0.05,
                    'DRr': 0.75, 'DRr_mean': 0.75, 'DRr_std': 0.05,
                    'ASTr': 0.55, 'ASTr_mean': 0.55, 'ASTr_std': 0.05,
                    'TOr': 0.13, 'TOr_mean': 0.13, 'TOr_std': 0.02,
                }
    
    # Calculate stats for derived metrics
    # Handle potential division by zero with safe replacement
    fga_safe = df_before['FGA'].replace(0, np.nan)
    fg_safe = df_before['FG'].replace(0, np.nan)
    
    # Calculate per-game rate series for std dev calculation
    s_3par = df_before['3PA'] / fga_safe
    s_ftr = df_before['FTA'] / fga_safe
    
    # Rebound rates require total rebounds
    total_reb = df_before['OR'] + df_before['DR']
    reb_safe = total_reb.replace(0, np.nan)
    s_orr = df_before['OR'] / reb_safe
    s_drr = df_before['DR'] / reb_safe
    
    s_astr = df_before['A'] / fg_safe
    
    # Turnover rate: TO / (FGA + 0.44*FTA + TO)
    poss_est = df_before['FGA'] + 0.44 * df_before['FTA'] + df_before['TO']
    poss_safe = poss_est.replace(0, np.nan)
    s_tor = df_before['TO'] / poss_safe

    # Calculate averages and standard deviations
    averages = {
        # Raw means (team averages)
        'OEFF': df_before['OEFF'].mean(),
        'OEFF_std': df_before['OEFF'].std(),
        
        'DEFF': df_before['DEFF'].mean(),
        'DEFF_std': df_before['DEFF'].std(),
        
        'PACE': df_before['PACE'].mean(),
        'PACE_std': df_before['PACE'].std(),
        
        # Derived metrics (Aggregate ratios for Center mode, Per-game Mean/Std for Z-Score)
        # Note: Center mode uses Sum/Sum to match historical behavior
        '3PAr': df_before['3PA'].sum() / df_before['FGA'].sum() if df_before['FGA'].sum() > 0 else 0.40,
        '3PAr_mean': s_3par.mean(),
        '3PAr_std': s_3par.std(),
        
        'FTr': df_before['FTA'].sum() / df_before['FGA'].sum() if df_before['FGA'].sum() > 0 else 0.25,
        'FTr_mean': s_ftr.mean(),
        'FTr_std': s_ftr.std(),
        
        'ORr': df_before['OR'].sum() / total_reb.sum() if total_reb.sum() > 0 else 0.25,
        'ORr_mean': s_orr.mean(),
        'ORr_std': s_orr.std(),
        
        'DRr': df_before['DR'].sum() / total_reb.sum() if total_reb.sum() > 0 else 0.75,
        'DRr_mean': s_drr.mean(),
        'DRr_std': s_drr.std(),
        
        'ASTr': df_before['A'].sum() / df_before['FG'].sum() if df_before['FG'].sum() > 0 else 0.55,
        'ASTr_mean': s_astr.mean(),
        'ASTr_std': s_astr.std(),
        
        'TOr': df_before['TO'].sum() / poss_est.sum() if poss_est.sum() > 0 else 0.13,
        'TOr_mean': s_tor.mean(),
        'TOr_std': s_tor.std(),
    }
    
    # Cache the result
    _league_avg_cache[cache_key] = averages
    
    return averages


# Cache for league context by (season, date)
_league_context_cache: Dict[Tuple[str, str], Dict[str, float]] = {}


def calculate_league_context(season: str, target_date: str, min_games: int = 50) -> Dict[str, float]:
    """
    Calculate league-wide context features (volatility/variance metrics) for a given date.
    
    These features describe the "state" of the league - how spread out teams are -
    which helps the model adjust its confidence based on the current environment.
    
    Args:
        season: Season string (e.g., '2023-2024')
        target_date: Date string (e.g., '2023-12-15')
        min_games: Minimum games required; if fewer, use prior season's final context
        
    Returns:
        Dictionary with context features:
        - ctx_std_oeff: Std dev of team offensive ratings
        - ctx_std_deff: Std dev of team defensive ratings
        - ctx_std_pace: Std dev of team pace
        - ctx_std_orb: Std dev of team offensive rebound rates
    """
    # Check cache first
    cache_key = (season, target_date)
    if cache_key in _league_context_cache:
        return _league_context_cache[cache_key]
    
    # Load season data
    df = _load_team_boxscores_for_season(season)
    
    if df.empty:
        # Return default context (historical averages)
        return _get_default_context()
    
    # Filter to games before target date
    if target_date not in _datetime_cache:
        _datetime_cache[target_date] = pd.to_datetime(target_date)
    target_dt = _datetime_cache[target_date]
    
    df_before = df[df['DATE'] < target_dt]
    
    # Check if we have enough data
    if len(df_before) < min_games:
        # Fall back to prior season's final context
        prior_season_map = {
            '2022-2023': '2021-2022',
            '2023-2024': '2022-2023',
            '2024-2025': '2023-2024',
            '2025-2026': '2024-2025',
        }
        
        if season in prior_season_map:
            prior_season = prior_season_map[season]
            prior_df = _load_team_boxscores_for_season(prior_season)
            
            if not prior_df.empty:
                # Use entire prior season
                df_before = prior_df
            else:
                return _get_default_context()
        else:
            return _get_default_context()
    
    # Calculate TEAM-LEVEL statistics (not game-level)
    # Group by team and calculate season-to-date averages per team
    team_stats = df_before.groupby('TEAM').agg({
        'OEFF': 'mean',
        'DEFF': 'mean',
        'PACE': 'mean',
        'OR': 'sum',
        'DR': 'sum',
    })
    
    # Calculate offensive rebound rate per team
    team_stats['ORB_rate'] = team_stats['OR'] / (team_stats['OR'] + team_stats['DR'])
    
    # Calculate standard deviations ACROSS TEAMS (this is the "spread" of the league)
    context = {
        'ctx_std_oeff': team_stats['OEFF'].std(),
        'ctx_std_deff': team_stats['DEFF'].std(),
        'ctx_std_pace': team_stats['PACE'].std(),
        'ctx_std_orb': team_stats['ORB_rate'].std(),
    }
    
    # Handle NaN (can happen with very few teams)
    for key in context:
        if pd.isna(context[key]):
            context[key] = _get_default_context()[key]
    
    # Cache the result
    _league_context_cache[cache_key] = context
    
    return context


def _get_default_context() -> Dict[str, float]:
    """Return default/fallback context values based on historical averages."""
    return {
        'ctx_std_oeff': 3.5,   # Historical avg std dev of team offensive ratings
        'ctx_std_deff': 3.5,   # Historical avg std dev of team defensive ratings
        'ctx_std_pace': 2.0,   # Historical avg std dev of team pace
        'ctx_std_orb': 0.03,   # Historical avg std dev of team ORB rate
    }


def normalize_team_features(raw_features: Dict[str, float], season: str, game_date: str, method: str = 'center') -> Dict[str, float]:
    """
    Normalize team features to be league-relative.
    
    Args:
        raw_features: Dictionary of raw team features (from get_team_features)
        season: Season string
        game_date: Game date string
        method: Normalization method ('center' or 'zscore'). 
               'center' = raw - avg (default)
               'zscore' = (raw - avg) / std
        
    Returns:
        Dictionary with both raw and normalized features
    """
    league_avg = calculate_league_averages(season, game_date)
    
    if league_avg is None:
        # No normalization possible, return raw features
        return raw_features
    
    # Create normalized features
    normalized = raw_features.copy()
    
    # Normalize core stats to be league-relative
    mapping = {
        'off_rating': 'OEFF',
        'def_rating': 'DEFF',
        'pace': 'PACE',
        'three_pt_rate': '3PAr',
        'free_throw_rate': 'FTr',
        'off_reb_rate': 'ORr',
        'def_reb_rate': 'DRr',
        'assist_rate': 'ASTr',
        'turnover_rate': 'TOr',
    }
    
    for feature_key, league_key in mapping.items():
        if feature_key in raw_features:
            raw_value = raw_features[feature_key]
            
            # Determine league stats based on method
            if method == 'zscore':
                # Use per-game mean and std for Z-score
                mu = league_avg.get(f"{league_key}_mean", league_avg.get(league_key))
                sigma = league_avg.get(f"{league_key}_std", 1.0)
                
                if sigma is None or sigma == 0:
                    sigma = 1.0
                    
            else:
                # Default 'center': use aggregate average
                mu = league_avg.get(league_key)
                sigma = 1.0 # Not used for centering
            
            # Only add normalized value if valid
            if raw_value is not None and mu is not None:
                if not (isinstance(raw_value, float) and math.isnan(raw_value)):
                    if not (isinstance(mu, float) and math.isnan(mu)):
                        
                        if method == 'zscore':
                             normalized[feature_key + '_norm'] = (raw_value - mu) / sigma
                        else:
                             normalized[feature_key + '_norm'] = raw_value - mu
    
    return normalized


def normalize_game_jsonl(input_path: str, output_path: str, method: str = 'center', include_context: bool = False):
    """
    Read games from input JSONL, add normalized features, write to output JSONL.
    OPTIMIZED: Batch process games by date to minimize league average calculations.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        method: Normalization method ('center' or 'zscore')
        include_context: If True, inject league context features into each game
    """
    context_str = " +context" if include_context else ""
    print(f"Normalizing features ({method}{context_str}): {input_path} -> {output_path}")
    
    # PERFORMANCE FIX #3: Batch process games by (season, date)
    # Read all games first
    print("  Reading games...")
    games = []
    with open(input_path, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            games.append(json.loads(line))
    
    print(f"  Read {len(games)} games, grouping by date...")
    
    # Group games by (season, date) - typically 5-15 games per night
    from collections import defaultdict
    games_by_date = defaultdict(list)
    for game in games:
        key = (game['season'], game['date'])
        games_by_date[key].append(game)
    
    print(f"  Found {len(games_by_date)} unique dates to process")
    print("  Computing league averages and normalizing...")
    
    # Process all games for each date at once
    normalized_games = []
    processed_count = 0
    
    for (season, game_date), date_games in games_by_date.items():
        # Calculate league average ONCE per date (not once per game!)
        league_avg = calculate_league_averages(season, game_date)
        
        # Calculate league context ONCE per date (if requested)
        league_context = None
        if include_context:
            league_context = calculate_league_context(season, game_date)
        
        # Apply to all games on this date
        for game in date_games:
            # Normalize home team features
            home_features = game['teams']['H']
            home_normalized = normalize_team_features(home_features, season, game_date, method=method)
            game['teams']['H'] = home_normalized
            
            # Normalize away team features
            away_features = game['teams']['A']
            away_normalized = normalize_team_features(away_features, season, game_date, method=method)
            game['teams']['A'] = away_normalized
            
            # Inject league context (same for all games on this date)
            if include_context and league_context:
                game['league_context'] = league_context
            
            normalized_games.append(game)
            processed_count += 1
            
            if processed_count % 500 == 0:
                print(f"    Processed {processed_count}/{len(games)} games...")
    
    # Write all normalized games
    print("  Writing normalized games...")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for game in normalized_games:
            f_out.write(json.dumps(game) + '\n')
    
    print(f"  Complete! Processed {len(normalized_games)} games total.")


def test_normalization():
    """Test the normalization on a few sample dates"""
    test_cases = [
        ('2023-2024', '2023-10-25'),  # Very early season
        ('2023-2024', '2023-12-15'),  # Mid-season
        ('2023-2024', '2024-04-15'),  # Late season
        ('2024-2025', '2024-11-15'),  # Current season
    ]
    
    print("\n" + "="*80)
    print("LEAGUE AVERAGE CALCULATION TEST")
    print("="*80)
    
    for season, date in test_cases:
        print(f"\n{season} as of {date}:")
        averages = calculate_league_averages(season, date)
        
        if averages:
            print(f"  Off Rating: {averages['OEFF']:.2f}")
            print(f"  Def Rating: {averages['DEFF']:.2f}")
            print(f"  Pace:       {averages['PACE']:.2f}")
            print(f"  3PT Rate:   {averages['3PAr']:.3f}")
        else:
            print("  [No data available]")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    # Run test
    test_normalization()
    
    # Example usage
    print("\nTo normalize your JSONL files, run:")
    print("  from league_normalizer import normalize_game_jsonl")
    print("  normalize_game_jsonl('data/games_train.jsonl', 'data/games_train_normalized.jsonl')")

