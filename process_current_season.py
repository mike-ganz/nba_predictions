"""
Process current season (2025-2026) data and generate predictions.

This script:
1. Finds the most recent team/player boxscore files in current/ directories
2. Processes them into JSONL format using the existing pipeline
3. Applies league normalization
4. Generates predictions using the trained model
"""
import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from prepare_data import process_file, TEAM_NAME_TO_ABBR, load_market_data
from league_normalizer import normalize_game_jsonl

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def find_most_recent_file(directory: Path, pattern: str = "*.xlsx") -> Optional[Path]:
    """
    Find the most recent file in a directory based on date prefix.
    
    Expects files named like: MM-DD-YYYY-*.xlsx
    
    Args:
        directory: Directory to search
        pattern: Glob pattern for files
        
    Returns:
        Path to the most recent file, or None if no files found
    """
    files = list(directory.glob(pattern))
    if not files:
        return None
    
    # Extract dates from filenames
    dated_files = []
    for f in files:
        # Look for date pattern like 10-30-2025
        match = re.match(r'(\d{1,2})-(\d{1,2})-(\d{4})', f.name)
        if match:
            month, day, year = match.groups()
            try:
                date = datetime(int(year), int(month), int(day))
                dated_files.append((date, f))
            except ValueError:
                logging.warning(f"Invalid date in filename: {f.name}")
                continue
    
    if not dated_files:
        # Fallback: just return the first file
        logging.warning(f"No dated files found in {directory}, using first file")
        return files[0]
    
    # Return the file with the most recent date
    dated_files.sort(reverse=True)
    most_recent = dated_files[0][1]
    logging.info(f"Selected most recent file: {most_recent.name}")
    return most_recent


def main():
    parser = argparse.ArgumentParser(description="Process current season data")
    parser.add_argument(
        '--team-boxscores-dir',
        type=str,
        default='data/team_boxscores/current',
        help='Directory containing current season team boxscores'
    )
    parser.add_argument(
        '--player-boxscores-dir',
        type=str,
        default='data/player_boxscores/current',
        help='Directory containing current season player boxscores'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/games_2025_2026_current.jsonl',
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--season',
        type=str,
        default='2025-2026',
        help='Season identifier'
    )
    parser.add_argument(
        '--include-players',
        action='store_true',
        default=True,
        help='Include player availability data'
    )
    parser.add_argument(
        '--market-data',
        type=str,
        default='data/market/current_spreads.json',
        help='Path to market data JSON file (current_spreads.json)'
    )
    args = parser.parse_args()
    
    # Load market data from current_spreads.json
    market_data_dict = {}
    market_path = Path(args.market_data)
    if market_path.exists():
        logging.info(f"Loading market data from: {market_path}")
        market_data_dict = load_market_data(market_path)
        if market_data_dict:
            logging.info(f"Loaded market data for {len(market_data_dict)} games")
        else:
            logging.warning("No market data loaded, will fallback to boxscore data")
    else:
        logging.warning(f"Market data file not found: {market_path}, will use boxscore data only")
    
    # Find most recent files
    team_dir = Path(args.team_boxscores_dir)
    player_dir = Path(args.player_boxscores_dir)
    
    if not team_dir.exists():
        logging.error(f"Team boxscores directory not found: {team_dir}")
        return
    
    team_file = find_most_recent_file(team_dir)
    if not team_file:
        logging.error(f"No team boxscore files found in {team_dir}")
        return
    
    logging.info(f"Processing team boxscores from: {team_file}")
    
    # Load player data if requested
    player_df = None
    if args.include_players and player_dir.exists():
        player_file = find_most_recent_file(player_dir)
        if player_file:
            logging.info(f"Loading player data from: {player_file}")
            try:
                # Load player data directly
                player_df = pd.read_excel(player_file)
                # Ensure DATE column is datetime
                if 'DATE' in player_df.columns:
                    player_df['DATE'] = pd.to_datetime(player_df['DATE'])
                logging.info(f"Loaded {len(player_df)} player-game records")
                
                # Pre-compute baselines for this season
                from scripts.player_data_loader import precompute_season_baselines
                precompute_season_baselines(args.season, player_df)
            except Exception as e:
                logging.warning(f"Failed to load player data: {e}")
                player_df = None
        else:
            logging.warning(f"No player boxscore files found in {player_dir}")
    
    # Process the team boxscore file
    logging.info(f"Processing {args.season} season data...")
    games = process_file(team_file, args.season, player_df, market_data_dict)
    
    # Write to JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        for record in games:
            f.write(json.dumps(record) + "\n")
    
    games_with_players = sum(1 for g in games if "players" in g)
    logging.info(f"✅ Wrote {len(games)} game records to {output_path}")
    logging.info(f"   {games_with_players} games include player data")
    
    # Now apply normalization
    logging.info(f"Applying league normalization...")
    output_norm_path = output_path.with_name(output_path.stem + "_norm.jsonl")
    
    try:
        normalize_game_jsonl(str(output_path), str(output_norm_path))
        logging.info(f"✅ Wrote normalized data to {output_norm_path}")
    except Exception as e:
        logging.error(f"Normalization failed: {e}")
        raise
    
    logging.info("=" * 60)
    logging.info("Current season data processing complete!")
    logging.info(f"  Raw data: {output_path}")
    logging.info(f"  Normalized data: {output_norm_path}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()

