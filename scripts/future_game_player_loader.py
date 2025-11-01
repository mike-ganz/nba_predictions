"""Load player availability data for future games based on season roster."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

import pandas as pd

from scripts.player_data_loader import _compute_simple_season_stats, _season_player_games


def get_last_game_date_for_team(
    team_name: str,
    season: str,
    player_boxscore_df: pd.DataFrame
) -> Optional[str]:
    """
    Find the most recent game date for a team in the player boxscore data.
    
    Args:
        team_name: Team name (e.g., "Milwaukee", "Brooklyn")
        season: Season year (e.g., "2024-2025")
        player_boxscore_df: DataFrame with player boxscore data
        
    Returns:
        Most recent game date as YYYY-MM-DD string, or None if no games found
    """
    team_games = player_boxscore_df[
        player_boxscore_df['OWN \nTEAM'].str.contains(team_name, case=False, na=False)
    ]
    
    if len(team_games) == 0:
        logging.warning(f"No games found for {team_name} in player boxscores")
        return None
    
    last_date = pd.to_datetime(team_games['DATE']).max()
    return last_date.strftime("%Y-%m-%d")


def get_season_roster_players(
    team_name: str,
    season: str,
    player_boxscore_df: pd.DataFrame,
    injured_players: Set[str]
) -> List[Dict]:
    """
    Get PlayerAvailability records for all players who have played for the team this season.
    
    Logic:
    1. Gets all games for the team in the current season
    2. Finds all unique players who have logged minutes
    3. Computes season-to-date baseline stats for each player
    4. Filters out injured players
    5. Returns top players by total minutes played this season
    
    Args:
        team_name: Team name (e.g., "Milwaukee")
        season: Season year (e.g., "2024-2025")
        player_boxscore_df: DataFrame with player boxscore data
        injured_players: Set of player names to exclude (case-insensitive)
        
    Returns:
        List of PlayerAvailability dicts (without injured players), sorted by total minutes
    """
    MIN_GAMES = 5  # Minimum games played to be included
    
    # Get all games for the team this season
    team_games = player_boxscore_df[
        player_boxscore_df['OWN \nTEAM'].str.contains(team_name, case=False, na=False)
    ].copy()
    
    if len(team_games) == 0:
        logging.warning(f"No games found for {team_name} in {season}")
        return []
    
    team_games['DATE'] = pd.to_datetime(team_games['DATE'])
    
    # Get unique players and their total stats
    player_stats = []
    
    for player_name in team_games['PLAYER \nFULL NAME'].unique():
        player_games = team_games[team_games['PLAYER \nFULL NAME'] == player_name]
        
        # Skip players with too few games
        if len(player_games) < MIN_GAMES:
            continue
        
        # Calculate total minutes (to rank by playing time)
        total_minutes = player_games['MIN'].sum()
        games_played = len(player_games)
        avg_minutes = total_minutes / games_played
        
        # Mark if player is injured (but don't skip them yet!)
        is_injured = player_name.lower() in injured_players
        if is_injured:
            logging.info(f"Injured player will be included with 0 projected minutes: {player_name}")
        
        player_stats.append({
            'name': player_name,
            'games': games_played,
            'total_minutes': total_minutes,
            'avg_minutes': avg_minutes,
            'player_games': player_games,
            'is_injured': is_injured
        })
    
    # Sort by total minutes and take top 10
    player_stats.sort(key=lambda x: x['total_minutes'], reverse=True)
    top_players = player_stats[:10]
    
    logging.info(f"Found {len(top_players)} players for {team_name} (from {len(player_stats)} total with {MIN_GAMES}+ games)")
    
    # Get cached season data for stat computation
    global _season_player_games
    if season not in _season_player_games:
        logging.warning(f"Season {season} not cached, using raw data")
        season_df = player_boxscore_df
    else:
        season_df = _season_player_games[season]
    
    # Now compute baseline stats for each player
    players = []
    
    # For each player in our roster, compute baseline stats
    for player_info in top_players:
        player_name = player_info['name']
        
        # Get all season games for this player (for computing stats)
        player_season_games = season_df[
            (season_df['PLAYER \nFULL NAME'] == player_name) &
            (season_df['OWN \nTEAM'].str.contains(team_name, case=False, na=False))
        ]
        
        # Compute baseline stats
        baseline_stats = _compute_simple_season_stats(player_season_games)
        
        if baseline_stats is None or baseline_stats['GP'] < MIN_GAMES:
            logging.debug(f"Insufficient data for {player_name}, using season averages")
            baseline_minutes = player_info['avg_minutes']
            baseline_ts = 0.53
            baseline_usage = 20.0
        else:
            baseline_minutes = baseline_stats['MPG']
            baseline_ts = baseline_stats['TS%']
            baseline_usage = baseline_stats['USAGE_RATE']
        
        # Clamp values
        baseline_minutes = max(0.0, min(48.0, baseline_minutes))
        baseline_ts = max(0.3, min(0.8, baseline_ts))
        baseline_usage = max(5.0, min(40.0, baseline_usage))
        
        # Convert usage to decimal
        baseline_usage_decimal = baseline_usage / 100.0 if baseline_usage > 1 else baseline_usage
        
        player_id = player_name.lower().replace(" ", "_").replace("'", "").replace(".", "")
        
        # Set projected_minutes based on injury status
        # - Injured players: projected_minutes = 0 (so their minutes are counted as "missing")
        # - Healthy players: projected_minutes = None (defaults to baseline_minutes)
        projected_minutes = 0.0 if player_info['is_injured'] else None
        
        players.append({
            "player_id": player_id,
            "player_name": player_name,
            "baseline_minutes": round(baseline_minutes, 1),
            "projected_minutes": projected_minutes,
            "baseline_ts_pct": round(baseline_ts, 3),
            "baseline_usage_rate": round(baseline_usage_decimal, 3)
        })
    
    # Sort by baseline_minutes (descending)
    players.sort(key=lambda p: p['baseline_minutes'], reverse=True)
    
    return players[:10]


__all__ = [
    'get_season_roster_players',
    'get_last_game_date_for_team',
]
