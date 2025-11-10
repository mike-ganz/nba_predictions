
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
    
    # PERFORMANCE FIX: Convert DATE to datetime ONCE here, not every lookup!
    df_cached = player_boxscore_df.copy()
    df_cached['DATE'] = pd.to_datetime(df_cached['DATE'])
    
    # PERFORMANCE FIX: Normalize team names to avoid repeated str.contains()
    # Extract city name for faster exact matching
    if 'OWN \nTEAM' in df_cached.columns:
        # Parse team names like "Milwaukee Bucks" -> "Milwaukee"
        df_cached['TEAM_CITY'] = df_cached['OWN \nTEAM'].str.extract(r'^([A-Za-z\s]+?)(?:\s+[A-Z]|$)', expand=False).str.strip()
    
    # Store sorted by date, team, player for better cache locality
    _season_player_games[season] = df_cached.sort_values(['DATE', 'TEAM_CITY', 'PLAYER \nFULL NAME'])
    
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


def _get_baseline_roster(
    team_name: str,
    game_date: str,
    season: str,
    player_boxscore_df: pd.DataFrame,
    lookback_games: int = 10,
    min_games_for_roster: int = 3
) -> Dict[str, Dict]:
    """
    Get baseline roster by analyzing recent games before the target date.
    
    Args:
        team_name: Team name (e.g., "Milwaukee")
        game_date: Game date in YYYY-MM-DD format
        season: Season year (e.g., "2021-2022")
        player_boxscore_df: DataFrame with player boxscore data
        lookback_games: Number of games to look back
        min_games_for_roster: Minimum games a player must appear in to be in baseline roster
        
    Returns:
        Dict mapping player_name -> {baseline_minutes, baseline_ts, baseline_usage, games_played}
    """
    game_date_dt = pd.to_datetime(game_date)
    
    # Use cached DataFrame if available
    if season in _season_player_games:
        season_df = _season_player_games[season]
    else:
        season_df = player_boxscore_df
        if 'DATE' in season_df.columns and season_df['DATE'].dtype != 'datetime64[ns]':
            season_df = season_df.copy()
            season_df['DATE'] = pd.to_datetime(season_df['DATE'])
    
    # Get team's games before this date
    if 'TEAM_CITY' in season_df.columns:
        team_games = season_df[
            ((season_df['TEAM_CITY'] == team_name) | 
             (season_df['TEAM_CITY'].str.contains(team_name, case=False, na=False))) &
            (season_df['DATE'] < game_date_dt)
        ]
    else:
        team_games = season_df[
            (season_df['OWN \nTEAM'].str.contains(team_name, case=False, na=False)) &
            (season_df['DATE'] < game_date_dt)
        ]
    
    if len(team_games) == 0:
        logging.warning(f"No prior games found for {team_name} before {game_date}")
        return {}
    
    # Get unique game dates and take the last N games
    unique_dates = sorted(team_games['DATE'].unique())[-lookback_games:]
    
    # Filter to only those recent games
    recent_games = team_games[team_games['DATE'].isin(unique_dates)]
    
    # Group by player and compute baseline stats
    baseline_roster = {}
    grouped = recent_games.groupby('PLAYER \nFULL NAME')
    
    for player_name, player_games in grouped:
        games_played = len(player_games['DATE'].unique())
        
        # Only include if player appeared in enough games
        if games_played < min_games_for_roster:
            continue
        
        # Compute baseline stats
        total_min = player_games['MIN'].sum()
        total_pts = player_games['PTS'].sum()
        total_fga = player_games['FGA'].sum()
        total_fta = player_games['FTA'].sum()
        
        baseline_minutes = total_min / games_played
        ppg = total_pts / games_played
        fga_pg = total_fga / games_played
        fta_pg = total_fta / games_played
        
        # True Shooting %
        ts_denominator = 2 * (fga_pg + 0.44 * fta_pg)
        baseline_ts = ppg / ts_denominator if ts_denominator > 0 else 0.53
        
        # Usage rate
        if 'USAGE \nRATE (%)' in player_games.columns:
            baseline_usage = player_games['USAGE \nRATE (%)'].mean()
        elif 'USAGE RATE (%)' in player_games.columns:
            baseline_usage = player_games['USAGE RATE (%)'].mean()
        else:
            baseline_usage = 20.0
        
        baseline_roster[player_name] = {
            'baseline_minutes': baseline_minutes,
            'baseline_ts': baseline_ts,
            'baseline_usage': baseline_usage,
            'games_played': games_played
        }
    
    return baseline_roster


def get_team_players_with_injuries(
    team_name: str,
    game_date: str,
    season: str,
    player_boxscore_df: pd.DataFrame,
    lookback_games: int = 10,
    injury_threshold_mins: float = 10.0,
    min_games_for_roster: int = 3
) -> List[Dict]:
    """
    Get PlayerAvailability records with injury reconstruction from recent games.
    
    This function reconstructs the full roster by analyzing recent games, then marks
    players as injured if they're missing from the current game's boxscore.
    
    Logic:
    - Player in boxscore with >0 mins: healthy (projected_minutes=None)
    - Player in boxscore with 0 mins: OUT (projected_minutes=0.0)
    - Player missing, baseline ≥ injury_threshold_mins: injured (projected_minutes=0.0)
    - Player missing, baseline < injury_threshold_mins: DNP-CD (omit from roster)
    
    Args:
        team_name: Team name (e.g., "Milwaukee")
        game_date: Game date in YYYY-MM-DD format
        season: Season year (e.g., "2021-2022")
        player_boxscore_df: DataFrame with player boxscore data
        lookback_games: Number of games to look back for baseline roster
        injury_threshold_mins: Minimum baseline minutes to be considered "injured" when missing
        min_games_for_roster: Minimum games to be in baseline roster
        
    Returns:
        List of PlayerAvailability dicts, sorted by baseline_minutes descending
    """
    game_date_dt = pd.to_datetime(game_date)
    
    # Step 1: Get baseline roster from recent games
    baseline_roster = _get_baseline_roster(
        team_name, game_date, season, player_boxscore_df,
        lookback_games, min_games_for_roster
    )
    
    if not baseline_roster:
        logging.warning(f"No baseline roster for {team_name} on {game_date}, falling back to legacy")
        return get_team_players_legacy(team_name, game_date, season, player_boxscore_df)
    
    # Step 2: Get current game's boxscore
    if season in _season_player_games:
        cached_df = _season_player_games[season]
    else:
        cached_df = player_boxscore_df
        if 'DATE' in cached_df.columns and cached_df['DATE'].dtype != 'datetime64[ns]':
            cached_df = cached_df.copy()
            cached_df['DATE'] = pd.to_datetime(cached_df['DATE'])
    
    # Get this game's players
    if 'TEAM_CITY' in cached_df.columns:
        game_data = cached_df[
            ((cached_df['TEAM_CITY'] == team_name) | 
             (cached_df['TEAM_CITY'].str.contains(team_name, case=False, na=False))) &
            (cached_df['DATE'] == game_date_dt)
        ]
    else:
        game_data = cached_df[
            (cached_df['OWN \nTEAM'].str.contains(team_name, case=False, na=False)) &
            (cached_df['DATE'] == game_date_dt)
        ]
    
    # Create a mapping of player -> actual_minutes for this game
    game_player_minutes = {}
    if len(game_data) > 0:
        for _, row in game_data.iterrows():
            player_name = row['PLAYER \nFULL NAME']
            actual_minutes = row['MIN']
            game_player_minutes[player_name] = actual_minutes
    
    # Step 3: Build the roster with injury status
    players = []
    
    for player_name, baseline_stats in baseline_roster.items():
        baseline_minutes = baseline_stats['baseline_minutes']
        baseline_ts = baseline_stats['baseline_ts']
        baseline_usage = baseline_stats['baseline_usage']
        
        # Determine injury/availability status
        if player_name in game_player_minutes:
            # Player is in the boxscore
            actual_minutes = game_player_minutes[player_name]
            if actual_minutes == 0:
                # Player was with team but didn't play: marked as OUT
                projected_minutes_value = 0.0
            else:
                # Player played: healthy
                projected_minutes_value = None
        else:
            # Player missing from boxscore
            if baseline_minutes >= injury_threshold_mins:
                # Significant player missing: assume injured
                projected_minutes_value = 0.0
            else:
                # Bench player missing: DNP-Coach's Decision, skip
                continue
        
        # Convert usage to decimal
        baseline_usage_decimal = baseline_usage / 100.0 if baseline_usage > 1 else baseline_usage
        
        player_id = player_name.lower().replace(" ", "_").replace("'", "").replace(".", "")
        
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


def get_team_players(
    team_name: str,
    game_date: str,
    season: str,
    player_boxscore_df: pd.DataFrame
) -> List[Dict]:
    """
    Get all PlayerAvailability records for a team's rotation with injury reconstruction.
    
    This is the main entry point that uses the new injury reconstruction logic.
    
    Args:
        team_name: Team name (e.g., "Milwaukee")
        game_date: Game date in YYYY-MM-DD format
        season: Season year (e.g., "2021-2022")
        player_boxscore_df: DataFrame with player boxscore data
        
    Returns:
        List of PlayerAvailability dicts
    """
    return get_team_players_with_injuries(
        team_name, game_date, season, player_boxscore_df
    )


def get_team_players_legacy(
    team_name: str,
    game_date: str,
    season: str,
    player_boxscore_df: pd.DataFrame
) -> List[Dict]:
    """
    LEGACY: Get all PlayerAvailability records for a team's rotation (VECTORIZED).
    
    This function only includes players who appear in the boxscore for the current game.
    Use get_team_players_with_injuries() for consistent injury handling.
    
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
    # PERFORMANCE FIX: Use cached DataFrame with pre-converted dates and normalized teams
    game_date_dt = pd.to_datetime(game_date)
    
    # Use cached DataFrame (already has datetime dates)
    cached_df = player_boxscore_df
    
    # PERFORMANCE: Try exact match on TEAM_CITY first (much faster)
    if 'TEAM_CITY' in cached_df.columns:
        # Try exact city match first
        game_data = cached_df[
            (cached_df['TEAM_CITY'] == team_name) &
            (cached_df['DATE'] == game_date_dt)
        ]
        
        # Fallback to contains if no exact match
        if len(game_data) == 0:
            game_data = cached_df[
                (cached_df['TEAM_CITY'].str.contains(team_name, case=False, na=False)) &
                (cached_df['DATE'] == game_date_dt)
            ]
    else:
        # Old method if TEAM_CITY not available
        game_data = cached_df[
            (cached_df['OWN \nTEAM'].str.contains(team_name, case=False, na=False)) &
            (cached_df['DATE'] == game_date_dt)
        ]
    
    game_data = game_data.sort_values('MIN', ascending=False).head(10)
    
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
    
    # PERFORMANCE FIX: Pre-filter season data ONCE for all players (not per-player)
    games_before_date = season_df[season_df['DATE'] < game_date_dt]
    
    # Extract player info as arrays for faster access
    player_names = game_data['PLAYER \nFULL NAME'].values
    actual_minutes_arr = game_data['MIN'].values
    
    # For each player in this game, compute baseline
    for idx, player_name in enumerate(player_names):
        actual_minutes = actual_minutes_arr[idx]
        
        # Get player's games BEFORE this date (already filtered above!)
        player_games_before = games_before_date[
            games_before_date['PLAYER \nFULL NAME'] == player_name
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
    'get_team_players_with_injuries',
    'get_team_players_legacy',
    'get_team_roster_for_game',
    'precompute_season_baselines',
]

