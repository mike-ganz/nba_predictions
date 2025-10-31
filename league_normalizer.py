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

# Cache for league averages by (season, date)
_league_avg_cache: Dict[Tuple[str, str], Dict[str, float]] = {}


def _load_team_boxscores_for_season(season: str) -> pd.DataFrame:
    """Load team boxscore data for a given season"""
    season_map = {
        '2021-2022': '2021-2022_NBA_Box_Score_Team-Stats.xlsx',
        '2022-2023': '2022-2023_NBA_Box_Score_Team-Stats.xlsx',
        '2023-2024': '2023-2024_NBA_Box_Score_Team-Stats.xlsx',
        '2024-2025': '2024-2025_NBA_Box_Score_Team-Stats.xlsx',
    }
    
    # Check historical seasons first
    if season in season_map:
        # Files are in historical directory
        file_path = Path('data') / 'team_boxscores' / 'historical' / season_map[season]
        
        if file_path.exists():
            df = pd.read_excel(file_path)
            df['DATE'] = pd.to_datetime(df['DATE'])
            return df
    
    # For current season (2025-2026), look in current/ directory
    if season == '2025-2026':
        current_dir = Path('data') / 'team_boxscores' / 'current'
        if current_dir.exists():
            # Find most recent file (should be only one, but pick latest if multiple)
            xlsx_files = list(current_dir.glob('*.xlsx'))
            if xlsx_files:
                # Use the first/only file
                file_path = xlsx_files[0]
                df = pd.read_excel(file_path)
                df['DATE'] = pd.to_datetime(df['DATE'])
                return df
    
    return pd.DataFrame()


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
    
    # Filter to games before target date
    target_dt = pd.to_datetime(target_date)
    df_before = df[df['DATE'] < target_dt].copy()
    
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
                    'OEFF': 110.0,
                    'DEFF': 110.0,
                    'PACE': 100.0,
                    '3PAr': 0.40,
                    'FTr': 0.25,
                    'ORr': 0.25,
                    'DRr': 0.75,
                    'ASTr': 0.55,
                    'TOr': 0.13,
                }
    
    # Calculate averages
    averages = {
        'OEFF': df_before['OEFF'].mean(),
        'DEFF': df_before['DEFF'].mean(),
        'PACE': df_before['PACE'].mean(),
        '3PAr': df_before['3PA'].sum() / df_before['FGA'].sum() if df_before['FGA'].sum() > 0 else 0.40,
        'FTr': df_before['FTA'].sum() / df_before['FGA'].sum() if df_before['FGA'].sum() > 0 else 0.25,
        'ORr': df_before['OR'].sum() / (df_before['OR'].sum() + df_before['DR'].sum()) if (df_before['OR'].sum() + df_before['DR'].sum()) > 0 else 0.25,
        'DRr': df_before['DR'].sum() / (df_before['OR'].sum() + df_before['DR'].sum()) if (df_before['OR'].sum() + df_before['DR'].sum()) > 0 else 0.75,
        'ASTr': df_before['A'].sum() / df_before['FG'].sum() if df_before['FG'].sum() > 0 else 0.55,
        'TOr': df_before['TO'].sum() / (df_before['FGA'].sum() + 0.44 * df_before['FTA'].sum() + df_before['TO'].sum()) if (df_before['FGA'].sum() + 0.44 * df_before['FTA'].sum() + df_before['TO'].sum()) > 0 else 0.13,
    }
    
    # Cache the result
    _league_avg_cache[cache_key] = averages
    
    return averages


def normalize_team_features(raw_features: Dict[str, float], season: str, game_date: str) -> Dict[str, float]:
    """
    Normalize team features to be league-relative.
    
    Args:
        raw_features: Dictionary of raw team features (from get_team_features)
        season: Season string
        game_date: Game date string
        
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
        if feature_key in raw_features and league_key in league_avg:
            raw_value = raw_features[feature_key]
            league_value = league_avg[league_key]
            
            # Only add normalized value if both raw and league values are valid
            import math
            if raw_value is not None and league_value is not None:
                if not (isinstance(raw_value, float) and math.isnan(raw_value)):
                    if not (isinstance(league_value, float) and math.isnan(league_value)):
                        normalized[feature_key + '_norm'] = raw_value - league_value
    
    return normalized


def normalize_game_jsonl(input_path: str, output_path: str):
    """
    Read games from input JSONL, add normalized features, write to output JSONL.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
    """
    print(f"Normalizing features: {input_path} -> {output_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for i, line in enumerate(f_in, 1):
            game = json.loads(line)
            season = game['season']
            game_date = game['date']
            
            # Normalize home team features
            home_features = game['teams']['H']
            home_normalized = normalize_team_features(home_features, season, game_date)
            game['teams']['H'] = home_normalized
            
            # Normalize away team features
            away_features = game['teams']['A']
            away_normalized = normalize_team_features(away_features, season, game_date)
            game['teams']['A'] = away_normalized
            
            # Write normalized game
            f_out.write(json.dumps(game) + '\n')
            
            if i % 500 == 0:
                print(f"  Processed {i} games...")
    
    print(f"  Complete! Processed {i} games total.")


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

