"""Load and aggregate player availability data for game records."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from transform_player_stats import calculate_player_stats, load_player_data

# Global cache for player baselines per season
_player_baseline_cache: Dict[str, Dict[str, Dict]] = {}

# Global cache for prior season stats (load once per season)
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


def _get_prior_season_stat(player_name: str, season: str, stat_name: str, default: float) -> float:
    """Get a stat from prior season cache, with fallback to default."""
    if season not in _prior_season_cache:
        return default
    if player_name not in _prior_season_cache[season]:
        return default
    return _prior_season_cache[season][player_name].get(stat_name, default)


def _load_prior_season_cache(season: str) -> None:
    """Mark that we've attempted to load prior season (lazy load on demand)."""
    if season not in _prior_season_cache:
        _prior_season_cache[season] = {}  # Empty dict means "attempted but load lazily"


def get_player_availability(
    player_name: str,
    game_date: str,
    season: str,
    actual_minutes: Optional[float] = None,
    use_cache: bool = True
) -> Optional[Dict]:
    """
    Get PlayerAvailability data for a single player with fallback logic.
    
    Args:
        player_name: Full player name
        game_date: Game date in YYYY-MM-DD format  
        season: Season year (e.g., "2021-2022")
        actual_minutes: Actual minutes played in THIS game (for historical data)
        use_cache: Whether to use the global baseline cache
        
    Returns:
        Dict with player_id, player_name, baseline_minutes, projected_minutes,
        baseline_ts_pct, baseline_usage_rate
    """
    MIN_GAMES = 10  # Require 10 games for baseline
    
    # Check cache first
    cache_key = f"{season}_{player_name}_{game_date}"
    if use_cache and season in _player_baseline_cache and cache_key in _player_baseline_cache[season]:
        cached = _player_baseline_cache[season][cache_key]
        # Update projected_minutes with actual if provided
        if actual_minutes is not None:
            cached = cached.copy()
            cached['projected_minutes'] = round(actual_minutes, 1)
        return cached
    
    # Try current season first (games BEFORE this date)
    baseline_stats = calculate_player_stats(player_name, max_date=game_date, current_season=season)
    
    # If insufficient games in current season, fallback to prior season
    if baseline_stats is None or baseline_stats.get('GP', 0) < MIN_GAMES:
        # Try prior season (uses disk cache so reasonably fast)
        year_parts = season.split('-')
        if len(year_parts) == 2:
            prev_season = f"{int(year_parts[0])-1}-{int(year_parts[1])-1}"
            try:
                # Get full season stats from prior year (no max_date, leverages disk cache)
                baseline_stats = calculate_player_stats(player_name, max_date=None, current_season=prev_season)
            except Exception:
                baseline_stats = None
    
    # If still no data, use reasonable defaults
    if baseline_stats is None or baseline_stats.get('GP', 0) < 5:
        # Use position-based defaults or actual minutes from THIS game
        logging.debug(f"No historical data for {player_name}, using defaults")
        if actual_minutes and actual_minutes > 15:
            # They're playing significant minutes, use moderate defaults
            baseline_minutes = max(20.0, actual_minutes * 0.8)  # Assume they usually play 80% of current
            baseline_ts = 0.53  # League average
            baseline_usage = 20.0  # Moderate usage
        else:
            # Bench player or no data
            baseline_minutes = 15.0
            baseline_ts = 0.51
            baseline_usage = 18.0
    else:
        baseline_minutes = baseline_stats.get('MPG', 20.0)
        baseline_ts = baseline_stats.get('TS%', 0.53)
        baseline_usage = baseline_stats.get('USAGE_RATE', 20.0)
    
    # Convert usage from percentage (e.g., 28.5) to decimal (e.g., 0.285)
    baseline_usage_decimal = baseline_usage / 100.0 if baseline_usage else 0.20
    
    # For historical data, use actual minutes as projected
    projected_minutes = actual_minutes if actual_minutes is not None else baseline_minutes
    
    # Create a player_id (lowercase, underscores)
    player_id = player_name.lower().replace(" ", "_").replace("'", "").replace(".", "")
    
    result = {
        "player_id": player_id,
        "player_name": player_name,
        "baseline_minutes": round(baseline_minutes, 1),
        "projected_minutes": round(projected_minutes, 1) if projected_minutes else None,
        "baseline_ts_pct": round(baseline_ts, 3) if baseline_ts else 0.530,
        "baseline_usage_rate": round(baseline_usage_decimal, 3) if baseline_usage_decimal else 0.200
    }
    
    # Cache the result
    if use_cache:
        if season not in _player_baseline_cache:
            _player_baseline_cache[season] = {}
        _player_baseline_cache[season][cache_key] = result.copy()
    
    return result


def get_team_players(
    team_name: str,
    game_date: str,
    season: str,
    player_boxscore_df: pd.DataFrame
) -> List[Dict]:
    """
    Get all PlayerAvailability records for a team's rotation.
    
    Args:
        team_name: Team name (e.g., "Milwaukee")
        game_date: Game date in YYYY-MM-DD format
        season: Season year (e.g., "2021-2022")
        player_boxscore_df: DataFrame with player boxscore data
        
    Returns:
        List of PlayerAvailability dicts
    """
    # Get the roster for this specific game
    roster = get_team_roster_for_game(team_name, game_date, season, player_boxscore_df)
    
    if not roster:
        return []
    
    # Get actual minutes played for each player in THIS game
    game_data = player_boxscore_df[
        (player_boxscore_df['OWN \nTEAM'].str.contains(team_name, case=False, na=False)) &
        (pd.to_datetime(player_boxscore_df['DATE']) == pd.to_datetime(game_date))
    ]
    
    player_minutes = {}
    for _, row in game_data.iterrows():
        name = row['PLAYER \nFULL NAME']
        minutes = row['MIN']
        player_minutes[name] = minutes
    
    # Build availability records
    players = []
    for idx, player_name in enumerate(roster):
        if idx == 0:
            logging.debug(f"Processing player {idx+1}/{len(roster)} for {team_name} on {game_date}")
        actual_minutes = player_minutes.get(player_name, 0.0)
        availability = get_player_availability(
            player_name, 
            game_date, 
            season,
            actual_minutes=actual_minutes
        )
        if availability:
            players.append(availability)
    
    # Sort by baseline_minutes (descending) - stars first
    players.sort(key=lambda p: p['baseline_minutes'], reverse=True)
    
    # Return top 8-10 players (minimum 5 required by schema)
    return players[:10]


__all__ = [
    'get_team_players',
    'get_player_availability',
    'get_team_roster_for_game',
]

