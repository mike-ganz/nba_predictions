
"""Load and aggregate player availability data for game records."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from transform_player_stats import load_player_data

# Global cache for RAW player games per season (season -> DataFrame with all games)
_season_player_games: Dict[str, pd.DataFrame] = {}

# Global cache for prior season stats (season -> player -> stats dict)
_prior_season_cache: Dict[str, Dict[str, Dict]] = {}


def get_team_roster_for_game(
    team_name: str,
    game_date: str,
    season: str,
    player_boxscore_df: pd.DataFrame
) -> List[str]:
    """
    Get the rotation players (top 8-10 by minutes) for a team on a specific game.
    
    Args:
        team_name: Team name (e.g., "Milwaukee", "Brooklyn")
        game_date: Game date in YYYY-MM-DD format
        season: Season year (e.g., "2021-2022")
        player_boxscore_df: DataFrame with player boxscore data
        
    Returns:
        List of player names sorted by minutes played (descending)
    """
    # Filter to this team and this specific game
    game_players = player_boxscore_df[
        (player_boxscore_df['OWN \nTEAM'].str.contains(team_name, case=False, na=False)) &
        (pd.to_datetime(player_boxscore_df['DATE']) == pd.to_datetime(game_date))
    ].copy()
    
    if len(game_players) == 0:
        logging.warning(f"No players found for {team_name} on {game_date}")
        return []
    
    # Sort by minutes played (descending) and take top 10
    game_players = game_players.sort_values('MIN', ascending=False)
    top_players = game_players.head(10)
    
    return top_players['PLAYER \nFULL NAME'].tolist()


def _compute_simple_season_stats(player_df: pd.DataFrame) -> Optional[dict]:
    """
    Compute simple full-season averages for a player (FAST, no rolling calculation).
    Returns a single stats dict.
    """
    if len(player_df) == 0:
        return None
    
    total_games = len(player_df)
    if total_games < 5:
        return None
    
    # Simple aggregation (vectorized, FAST)
    total_min = player_df['MIN'].sum()
    total_pts = player_df['PTS'].sum()
    total_fga = player_df['FGA'].sum()
    total_fta = player_df['FTA'].sum()
    
    mpg = total_min / total_games
    ppg = total_pts / total_games
    fga_pg = total_fga / total_games
    fta_pg = total_fta / total_games
    
    # True Shooting % = PTS / (2 * (FGA + 0.44 * FTA))
    ts_denominator = 2 * (fga_pg + 0.44 * fta_pg)
    ts_pct = ppg / ts_denominator if ts_denominator > 0 else 0.53
    
    # Extract usage rate from source data (check both versions - newline and stripped)
    if 'USAGE \nRATE (%)' in player_df.columns:
        usage_rate = player_df['USAGE \nRATE (%)'].mean()
    elif 'USAGE RATE (%)' in player_df.columns:
        usage_rate = player_df['USAGE RATE (%)'].mean()
    else:
        usage_rate = 20.0
        logging.warning(f"No usage rate column found in player data. Available columns: {list(player_df.columns)[:10]}")
    
    return {
        'GP': total_games,
        'MPG': mpg,
        'TS%': ts_pct,
        'USAGE_RATE': usage_rate  # Now actually computed from source data
    }


def precompute_season_baselines(season: str, player_boxscore_df: pd.DataFrame) -> None:
    """
    Store raw player games in memory for fast date-filtered lookups.
    No lookahead bias - we filter by date at lookup time.
    
    Args:
        season: Season year (e.g., "2021-2022")
        player_boxscore_df: Full season player boxscore data
    """
    if season in _season_player_games:
        logging.info(f"Season {season} already cached")
        return
    
    logging.info(f"Caching raw player games for {season}...")
    
    # Store the entire DataFrame in memory (sorted by date for fast filtering)
    _season_player_games[season] = player_boxscore_df.sort_values('DATE').copy()
    
    logging.info(f"  Cached {len(player_boxscore_df)} player-games for {season}")
    
    # Also cache prior season stats (full season, used as fallback)
    year_parts = season.split('-')
    if len(year_parts) == 2:
        prev_season = f"{int(year_parts[0])-1}-{int(year_parts[1])-1}"
        
        if prev_season not in _prior_season_cache:
            logging.info(f"Computing prior season {prev_season} stats...")
            
            try:
                prev_season_df = load_player_data(prev_season)
                if prev_season_df is not None:
                    _prior_season_cache[prev_season] = {}
                    
                    # Group by player (vectorized, FAST)
                    grouped = prev_season_df.groupby('PLAYER \nFULL NAME')
                    
                    for player_name, player_games in grouped:
                        if len(player_games) < 10:
                            continue
                        
                        stats = _compute_simple_season_stats(player_games)
                        if stats:
                            _prior_season_cache[prev_season][player_name] = stats
                    
                    logging.info(f"  Computed {len(_prior_season_cache[prev_season])} players from {prev_season}")
            except Exception as e:
                logging.warning(f"Could not load prior season {prev_season}: {e}")
    
    logging.info(f"Caching complete for {season}")


def get_team_players(
    team_name: str,
    game_date: str,
    season: str,
    player_boxscore_df: pd.DataFrame
) -> List[Dict]:
    """
    Get all PlayerAvailability records for a team's rotation (VECTORIZED).
    
    Args:
        team_name: Team name (e.g., "Milwaukee")
        game_date: Game date in YYYY-MM-DD format
        season: Season year (e.g., "2021-2022")
        player_boxscore_df: DataFrame with player boxscore data
        
    Returns:
        List of PlayerAvailability dicts
    """
    MIN_GAMES = 10
    
    # Get THIS game's players for this team (sorted by minutes, top 10)
    game_data = player_boxscore_df[
        (player_boxscore_df['OWN \nTEAM'].str.contains(team_name, case=False, na=False)) &
        (pd.to_datetime(player_boxscore_df['DATE']) == pd.to_datetime(game_date))
    ].sort_values('MIN', ascending=False).head(10)
    
    if len(game_data) == 0:
        logging.warning(f"No players found for {team_name} on {game_date}")
        return []
    
    # VECTORIZED: Get baseline stats for ALL players at once
    players = []
    
    # Get cached season data
    if season not in _season_player_games:
        logging.warning(f"Season {season} not cached, using defaults")
        season_df = player_boxscore_df
    else:
        season_df = _season_player_games[season]
    
    # For each player in this game, compute baseline
    for _, row in game_data.iterrows():
        player_name = row['PLAYER \nFULL NAME']
        actual_minutes = row['MIN']
        
        # Get player's games BEFORE this date (vectorized filter)
        player_games_before = season_df[
            (season_df['PLAYER \nFULL NAME'] == player_name) &
            (pd.to_datetime(season_df['DATE']) < pd.to_datetime(game_date))
        ]
        
        baseline_stats = None
        if len(player_games_before) >= 5:
            baseline_stats = _compute_simple_season_stats(player_games_before)
        
        # Fallback to prior season if insufficient current season games
        if baseline_stats is None or baseline_stats.get('GP', 0) < MIN_GAMES:
            year_parts = season.split('-')
            if len(year_parts) == 2:
                prev_season = f"{int(year_parts[0])-1}-{int(year_parts[1])-1}"
                if prev_season in _prior_season_cache and player_name in _prior_season_cache[prev_season]:
                    baseline_stats = _prior_season_cache[prev_season][player_name]
        
        # Use defaults if still no data
        if baseline_stats is None or baseline_stats.get('GP', 0) < 5:
            if actual_minutes > 15:
                baseline_minutes = max(20.0, actual_minutes * 0.8)
                baseline_ts = 0.53
                baseline_usage = 20.0
            else:
                baseline_minutes = 15.0
                baseline_ts = 0.51
                baseline_usage = 18.0
        else:
            baseline_minutes = baseline_stats.get('MPG', 20.0)
            baseline_ts = baseline_stats.get('TS%', 0.53)
            # _compute_simple_season_stats returns 'USAGE_RATE' key
            baseline_usage = baseline_stats.get('USAGE_RATE', 20.0)
        
        # Convert usage to decimal
        baseline_usage_decimal = baseline_usage / 100.0 if baseline_usage > 1 else baseline_usage
        
        player_id = player_name.lower().replace(" ", "_").replace("'", "").replace(".", "")
        
        # FIX DATA LEAKAGE: Only use projected_minutes if player was OUT (DNP)
        # If player played ANY minutes, assume we expected their baseline
        # If player played 0 minutes, assume we had injury report (set to 0)
        if actual_minutes == 0:
            projected_minutes_value = 0.0
        else:
            projected_minutes_value = None  # Will default to baseline_minutes
        
        players.append({
            "player_id": player_id,
            "player_name": player_name,
            "baseline_minutes": round(baseline_minutes, 1),
            "projected_minutes": projected_minutes_value,
            "baseline_ts_pct": round(baseline_ts, 3),
            "baseline_usage_rate": round(baseline_usage_decimal, 3)
        })
    
    # Sort by baseline_minutes (descending)
    players.sort(key=lambda p: p['baseline_minutes'], reverse=True)
    
    return players[:10]


__all__ = [
    'get_team_players',
    'get_team_roster_for_game',
    'precompute_season_baselines',
]

