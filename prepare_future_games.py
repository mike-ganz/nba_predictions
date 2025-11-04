"""Prepare game records for future NBA games.

This script reads from:
- Schedule file (Excel): Lists upcoming games with teams and dates
- Player boxscores: Used to identify last game's roster
- Injury data (JSON): Players to exclude from rosters  
- Market data (JSON): Spreads, totals, moneylines for each game

Outputs game records in JSONL format compatible with the prediction pipeline.

Usage:
    python prepare_future_games.py --schedule data/schedules/current/2025-2026_NBA_Regular_Season_Schedule_Updated.xlsx \
                                     --player-boxscores-dir data/player_boxscores/historical \
                                     --injuries data/injuries/current_injuries.json \
                                     --market data/market/current_spreads.json \
                                     --output data/games_future_2025-2026.jsonl \
                                     --season 2025-2026
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from prepare_data import TEAM_NAME_TO_ABBR, get_team_features, LEAGUE_AVERAGES, _clamp
from scripts.future_game_player_loader import get_season_roster_players
from scripts.player_data_loader import precompute_season_baselines
from transform_player_stats import load_player_data


def normalize_team_name(team_name: str) -> str:
    """Normalize team name to match the format used in TEAM_NAME_TO_ABBR."""
    # Remove common suffixes and clean up
    team_name = team_name.strip()
    
    # Handle special cases
    if "Trail Blazers" in team_name:
        return "Portland"
    if "Clippers" in team_name:
        return "LA Clippers"
    if "Lakers" in team_name:
        return "LA Lakers"
    
    # Try to match against known team names
    for known_name in TEAM_NAME_TO_ABBR.keys():
        if known_name.lower() in team_name.lower() or team_name.lower() in known_name.lower():
            return known_name
    
    # Return as-is if no match
    return team_name


def load_schedule(schedule_path: Path) -> pd.DataFrame:
    """
    Load and parse the NBA schedule Excel file.
    
    Args:
        schedule_path: Path to schedule Excel file
        
    Returns:
        DataFrame with columns: date, away_team, home_team, away_rest_days, home_rest_days
    """
    logging.info(f"Loading schedule from {schedule_path}")
    
    # Read with header at row 3 (0-indexed)
    df = pd.read_excel(schedule_path, header=3)
    
    # Rename columns for easier access
    df = df.rename(columns={
        df.columns[0]: 'competition',
        df.columns[1]: 'date',
        df.columns[2]: 'game_time_eastern',
        df.columns[3]: 'game_time_local',
        df.columns[4]: 'away_rest_days',
        df.columns[5]: 'away_team',
        df.columns[8]: 'home_team',
        df.columns[9]: 'home_rest_days',
        df.columns[10]: 'game_label',
        df.columns[11]: 'arena',
        df.columns[12]: 'city',
    })
    
    # Filter to valid games
    df = df[df['competition'].notna() & df['date'].notna()].copy()
    df = df[df['away_team'].notna() & df['home_team'].notna()].copy()
    
    # Parse dates
    df['date'] = pd.to_datetime(df['date'])
    
    # Parse rest days (convert "3+" to 3)
    def parse_rest_days(val):
        if pd.isna(val):
            return None
        s = str(val).replace('+', '').strip()
        try:
            return int(s)
        except ValueError:
            return None
    
    df['away_rest_days'] = df['away_rest_days'].apply(parse_rest_days)
    df['home_rest_days'] = df['home_rest_days'].apply(parse_rest_days)
    
    # Normalize team names
    df['away_team'] = df['away_team'].apply(normalize_team_name)
    df['home_team'] = df['home_team'].apply(normalize_team_name)
    
    logging.info(f"Loaded {len(df)} games from schedule")
    
    return df[['date', 'away_team', 'home_team', 'away_rest_days', 'home_rest_days']]


def load_injuries(injuries_path: Path) -> Set[str]:
    """
    Load injury data and return set of player names to exclude.
    
    Args:
        injuries_path: Path to injuries JSON file
        
    Returns:
        Set of player names (lowercase) who are out or doubtful
    """
    if not injuries_path.exists():
        logging.warning(f"Injury file not found: {injuries_path}, proceeding without injury data")
        return set()
    
    logging.info(f"Loading injuries from {injuries_path}")
    
    with open(injuries_path, 'r') as f:
        data = json.load(f)
    
    injured = set()
    for injury in data.get('injuries', []):
        status = injury.get('status', '').lower()
        if status in ['out', 'doubtful']:
            player_name = injury.get('player_name', '')
            if player_name:
                injured.add(player_name.lower())
    
    logging.info(f"Found {len(injured)} players out or doubtful")
    return injured


def load_market_data(market_path: Path) -> Dict[Tuple[str, str, str], Dict]:
    """
    Load market data and return dictionary keyed by (date, away_team, home_team).
    
    Args:
        market_path: Path to market JSON file
        
    Returns:
        Dict mapping (date_str, away_team, home_team) to market info
    """
    if not market_path.exists():
        logging.warning(f"Market file not found: {market_path}")
        return {}
    
    logging.info(f"Loading market data from {market_path}")
    
    with open(market_path, 'r') as f:
        data = json.load(f)
    
    # Handle both formats: array or object with "games" key
    if isinstance(data, list):
        games = data
    elif isinstance(data, dict) and 'games' in data:
        games = data['games']
    else:
        logging.error(f"Invalid market data format in {market_path}")
        return {}
    
    market_dict = {}
    for game in games:
        date_str = game.get('game_date')
        away_team = normalize_team_name(game.get('away_team', ''))
        home_team = normalize_team_name(game.get('home_team', ''))
        
        key = (date_str, away_team, home_team)
        market_dict[key] = {
            'spread_home': game.get('spread_home'),
            'total': game.get('total'),
            'moneyline_home': game.get('moneyline_home'),
            'moneyline_away': game.get('moneyline_away'),
        }
    
    logging.info(f"Loaded market data for {len(market_dict)} games")
    return market_dict


def build_future_game_record(
    date: pd.Timestamp,
    away_team: str,
    home_team: str,
    season: str,
    away_rest_days: Optional[int],
    home_rest_days: Optional[int],
    market_info: Optional[Dict],
    player_boxscore_df: Optional[pd.DataFrame],
    injured_players: Set[str]
) -> Optional[Dict]:
    """
    Build a game record for a future game.
    
    Args:
        date: Game date
        away_team: Away team name
        home_team: Home team name
        season: Season (e.g., "2025-2026")
        away_rest_days: Rest days for away team (from schedule)
        home_rest_days: Rest days for home team (from schedule)
        market_info: Market data dict with spread, total, moneylines
        player_boxscore_df: Player boxscore data for getting rosters
        injured_players: Set of injured player names to exclude
        
    Returns:
        Game record dict, or None if critical data is missing
    """
    date_str = date.strftime("%Y-%m-%d")
    
    # Get team features (this uses generate_team_stats internally)
    away_features, away_meta = get_team_features(away_team, date_str, season)
    home_features, home_meta = get_team_features(home_team, date_str, season)
    
    # Override rest days if provided in schedule
    if away_rest_days is not None:
        rest_bucket = 3 if away_rest_days >= 3 else int(max(0, away_rest_days))
        away_features['rest_days'] = _clamp(float(rest_bucket), 0.0, 10.0)
        away_features['b2b'] = (away_rest_days == 1)  # Back-to-back: played yesterday
        away_features['three_in_four'] = (away_rest_days == 1)  # Proxy for compressed schedule
        away_meta['rest_days_raw'] = away_rest_days
    if home_rest_days is not None:
        rest_bucket = 3 if home_rest_days >= 3 else int(max(0, home_rest_days))
        home_features['rest_days'] = _clamp(float(rest_bucket), 0.0, 10.0)
        home_features['b2b'] = (home_rest_days == 1)  # Back-to-back: played yesterday
        home_features['three_in_four'] = (home_rest_days == 1)  # Proxy for compressed schedule
        home_meta['rest_days_raw'] = home_rest_days
    
    # Get market data (required)
    if market_info is None:
        logging.warning(f"No market data for {away_team} @ {home_team} on {date_str}")
        return None
    
    spread_home = market_info.get('spread_home')
    total = market_info.get('total')
    moneyline_home = market_info.get('moneyline_home')
    moneyline_away = market_info.get('moneyline_away')
    
    if None in {spread_home, total, moneyline_home, moneyline_away}:
        logging.warning(f"Incomplete market data for {away_team} @ {home_team} on {date_str}")
        return None
    
    # Generate game_id
    away_abbr = TEAM_NAME_TO_ABBR.get(away_team, away_team[:3].upper())
    home_abbr = TEAM_NAME_TO_ABBR.get(home_team, home_team[:3].upper())
    game_id = f"{date_str}-{away_abbr}-{home_abbr}"
    
    # Build player availability data if player boxscore data is provided
    players_data = None
    injury_warnings = []
    if player_boxscore_df is not None:
        try:
            away_players = get_season_roster_players(away_team, season, player_boxscore_df, injured_players)
            home_players = get_season_roster_players(home_team, season, player_boxscore_df, injured_players)
            
            # Always include players (even with injuries)
            players_data = {
                "A": away_players,
                "H": home_players,
            }
            
            # Warn if teams have few players due to injuries
            if len(away_players) < 5:
                warning = f"WARNING: {away_team} has only {len(away_players)} players available (multiple injuries)"
                logging.warning(warning)
                injury_warnings.append(warning)
            
            if len(home_players) < 5:
                warning = f"WARNING: {home_team} has only {len(home_players)} players available (multiple injuries)"
                logging.warning(warning)
                injury_warnings.append(warning)
                
        except Exception as e:
            logging.warning(f"Failed to load player data for {game_id}: {e}")
    
    game_record = {
        "game_id": game_id,
        "season": season,
        "date": date_str,
        "teams": {
            "A": away_features,
            "H": home_features,
        },
        "market": {
            "spread_home": float(spread_home),
            "total": float(total),
            "moneyline_home": int(moneyline_home),
            "moneyline_away": int(moneyline_away),
        },
        "metadata": {
            "source": "future_game_prediction",
            "team_stats": {
                "away": away_meta,
                "home": home_meta,
            },
            "injury_warnings": injury_warnings if injury_warnings else None,
        },
    }
    
    # Add players field if we have player data
    if players_data is not None:
        game_record["players"] = players_data
    
    # Note: No "outcome" field since this is a future game
    
    return game_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare future game data for predictions")
    parser.add_argument("--schedule", type=str, required=True, 
                       help="Path to schedule Excel file")
    parser.add_argument("--player-boxscores-dir", type=str, 
                       default="data/player_boxscores/historical",
                       help="Directory containing player boxscore Excel files")
    parser.add_argument("--injuries", type=str, 
                       default="data/injuries/current_injuries.json",
                       help="Path to injuries JSON file")
    parser.add_argument("--market", type=str,
                       default="data/market/current_spreads.json",
                       help="Path to market data JSON file")
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSONL file path")
    parser.add_argument("--season", type=str, required=True,
                       help="Season (e.g., '2025-2026')")
    parser.add_argument("--include-players", action="store_true",
                       help="Include player availability data in the output")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of games to process (for testing)")
    parser.add_argument("--start-date", type=str, default=None,
                       help="Only include games on or after this date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None,
                       help="Only include games before or on this date (YYYY-MM-DD)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")
    
    # Load schedule
    schedule_path = Path(args.schedule)
    if not schedule_path.exists():
        parser.error(f"Schedule file not found: {schedule_path}")
    
    schedule_df = load_schedule(schedule_path)
    
    # Filter by date range if specified
    if args.start_date:
        start_date = pd.to_datetime(args.start_date)
        schedule_df = schedule_df[schedule_df['date'] >= start_date]
        logging.info(f"Filtered to games on or after {args.start_date}: {len(schedule_df)} games")
    
    if args.end_date:
        end_date = pd.to_datetime(args.end_date)
        schedule_df = schedule_df[schedule_df['date'] <= end_date]
        logging.info(f"Filtered to games on or before {args.end_date}: {len(schedule_df)} games")
    
    # Limit games if specified
    if args.limit:
        schedule_df = schedule_df.head(args.limit)
        logging.info(f"Limited to first {args.limit} games")
    
    # Load injuries
    injuries_path = Path(args.injuries)
    injured_players = load_injuries(injuries_path)
    
    # Load market data
    market_path = Path(args.market)
    market_dict = load_market_data(market_path)
    
    # Load player boxscore data if requested
    player_df = None
    if args.include_players:
        player_dir = Path(args.player_boxscores_dir)
        if player_dir.exists():
            try:
                # First try to load current season data (2025-2026)
                player_df = load_player_data(args.season)
                if player_df is not None:
                    logging.info(f"Loaded player data for {args.season}: {len(player_df)} player-games")
                    # Precompute season baselines for speed
                    precompute_season_baselines(args.season, player_df)
                else:
                    # Fall back to prior season if current not available
                    season_parts = args.season.split('-')
                    prior_season = f"{int(season_parts[0])-1}-{int(season_parts[1])-1}"
                    player_df = load_player_data(prior_season)
                    if player_df is not None:
                        logging.info(f"Loaded player data for {prior_season}: {len(player_df)} player-games")
                        # Precompute season baselines for speed
                        precompute_season_baselines(prior_season, player_df)
                    else:
                        logging.warning(f"Could not load player data for {prior_season}")
            except Exception as e:
                logging.warning(f"Failed to load player data: {e}")
        else:
            logging.warning(f"Player boxscores directory not found: {player_dir}")
    
    # Process games
    all_games: List[Dict] = []
    skipped = 0
    
    for idx, row in schedule_df.iterrows():
        date = row['date']
        away_team = row['away_team']
        home_team = row['home_team']
        away_rest = row['away_rest_days']
        home_rest = row['home_rest_days']
        
        date_str = date.strftime("%Y-%m-%d")
        
        # Look up market data
        market_key = (date_str, away_team, home_team)
        market_info = market_dict.get(market_key)
        
        if market_info is None:
            logging.debug(f"No market data for {away_team} @ {home_team} on {date_str}, skipping")
            skipped += 1
            continue
        
        # Build game record
        game_record = build_future_game_record(
            date=date,
            away_team=away_team,
            home_team=home_team,
            season=args.season,
            away_rest_days=away_rest,
            home_rest_days=home_rest,
            market_info=market_info,
            player_boxscore_df=player_df,
            injured_players=injured_players
        )
        
        if game_record:
            all_games.append(game_record)
        else:
            skipped += 1
    
    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        for record in all_games:
            f.write(json.dumps(record) + "\n")
    
    games_with_players = sum(1 for g in all_games if "players" in g)
    logging.info("")
    logging.info("="*70)
    logging.info(f"Wrote {len(all_games)} game records to {output_path}")
    if args.include_players:
        logging.info(f"  {games_with_players} games with player data")
    logging.info(f"  {skipped} games skipped (missing market data)")
    logging.info("="*70)


if __name__ == "__main__":
    main()

