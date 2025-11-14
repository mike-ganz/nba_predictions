"""Load player availability data for future games based on season roster."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

import pandas as pd

from scripts.player_data_loader import _compute_simple_season_stats, _season_player_games, _get_baseline_roster


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
    game_date: str,
    season: str,
    player_boxscore_df: pd.DataFrame,
    injured_players: Set[str]
) -> List[Dict]:
    """
    Build a day-of roster using last-10-games baselines (training-consistent) and injury overrides.
    
    Logic:
    1) Compute baseline roster over the team's last 10 games BEFORE game_date (no lookahead)
    2) For each player in that baseline roster:
       - baseline_minutes, baseline_ts, baseline_usage come from the 10-game window
       - projected_minutes = 0.0 if player is listed OUT/DOUBTFUL in injuries
       - otherwise projected_minutes = None (defaults to baseline minutes)
    3) Return top 10 players by baseline_minutes
    """
    baseline_roster = _get_baseline_roster(
        team_name=team_name,
        game_date=game_date,
        season=season,
        player_boxscore_df=player_boxscore_df,
        lookback_games=10,
        min_games_for_roster=3,
    )

    if not baseline_roster:
        logging.warning(f"No baseline roster for {team_name} on {game_date} (season {season})")
        return []

    players: List[Dict] = []

    # Sort by baseline minutes (descending) and take top 10
    sorted_items = sorted(
        baseline_roster.items(),
        key=lambda kv: kv[1].get('baseline_minutes', 0.0),
        reverse=True
    )[:10]

    for player_name, stats in sorted_items:
        baseline_minutes = stats.get('baseline_minutes', 20.0)
        baseline_ts = stats.get('baseline_ts', 0.53)
        baseline_usage = stats.get('baseline_usage', 20.0)

        # Clamp values
        baseline_minutes = max(0.0, min(48.0, baseline_minutes))
        baseline_ts = max(0.3, min(0.8, baseline_ts))
        baseline_usage = max(5.0, min(40.0, baseline_usage))

        # Convert usage to decimal if value appears to be a percent
        baseline_usage_decimal = baseline_usage / 100.0 if baseline_usage > 1 else baseline_usage

        player_id = player_name.lower().replace(" ", "_").replace("'", "").replace(".", "")

        # Injury override: set to 0 only when explicitly OUT/DOUBTFUL
        projected_minutes = 0.0 if player_name.lower() in injured_players else None

        players.append({
            "player_id": player_id,
            "player_name": player_name,
            "baseline_minutes": round(baseline_minutes, 1),
            "projected_minutes": projected_minutes,
            "baseline_ts_pct": round(baseline_ts, 3),
            "baseline_usage_rate": round(baseline_usage_decimal, 3)
        })

    return players


__all__ = [
    'get_season_roster_players',
    'get_last_game_date_for_team',
]
