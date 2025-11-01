#!/usr/bin/env python
"""Convenience wrapper to prepare today's games.

This script automatically uses today's date and calls prepare_future_games.py
with appropriate arguments for a typical daily prediction workflow.

Usage:
    # Today's games
    python prepare_todays_games.py
    
    # Specific date
    python prepare_todays_games.py --date 2025-11-15
    
    # Without player data (faster)
    python prepare_todays_games.py --no-players
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="Prepare game data for today's (or specified date's) NBA games"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date in YYYY-MM-DD format (defaults to today)"
    )
    parser.add_argument(
        "--season",
        type=str,
        default="2025-2026",
        help="Season (default: 2025-2026)"
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="data/schedules/current/2025-2026_NBA_Regular_Season_Schedule_Updated.xlsx",
        help="Path to schedule file"
    )
    parser.add_argument(
        "--injuries",
        type=str,
        default="data/injuries/current_injuries.json",
        help="Path to injuries JSON file"
    )
    parser.add_argument(
        "--market",
        type=str,
        default="data/market/current_spreads.json",
        help="Path to market data JSON file"
    )
    parser.add_argument(
        "--no-players",
        action="store_true",
        help="Exclude player data (faster processing)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()
    
    # Use today's date if not specified
    if args.date:
        date_str = args.date
        print(f"Using specified date: {date_str}")
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")
        print(f"Using today's date: {date_str}")
    
    # Build output filename
    output_file = f"data/games_future_{date_str}.jsonl"
    
    print("")
    print("=" * 70)
    print(f"Preparing game data for {date_str}")
    print("=" * 70)
    print("")
    
    # Build command
    cmd = [
        sys.executable,
        "prepare_future_games.py",
        "--schedule", args.schedule,
        "--injuries", args.injuries,
        "--market", args.market,
        "--output", output_file,
        "--season", args.season,
        "--start-date", date_str,
        "--end-date", date_str,
    ]
    
    # Add optional flags
    if not args.no_players:
        cmd.append("--include-players")
    
    if args.debug:
        cmd.append("--debug")
    
    # Run the pipeline
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        # Normalize the data for model compatibility
        print("")
        print("Normalizing features for model...")
        from league_normalizer import normalize_game_jsonl
        
        normalized_output = output_file.replace('.jsonl', '_norm.jsonl')
        normalize_game_jsonl(output_file, normalized_output)
        print(f"Normalized data saved to: {normalized_output}")
        output_file = normalized_output  # Update for message below
        print("")
        print("=" * 70)
        print(f"SUCCESS: Game data prepared for {date_str}")
        print("=" * 70)
        print("")
        print(f"Output file: {output_file}")
        print("Next steps:")
        print("  1. Review the prepared game data")
        print("  2. Generate predictions:")
        print("")
        print(f"     python predict_margin.py \\")
        print(f"       --data {output_file} \\")
        print(f"       --model artifacts/margin_normalized \\")
        print(f"       --output predictions/predictions_{date_str}.csv")
        print("")
        print("Note: Using normalized features (league-relative) as required by the model")
        print("")
    else:
        print("")
        print("ERROR: Failed to prepare game data", file=sys.stderr)
        print("")
        print("Common issues:")
        print(f"  - Missing market data for games on {date_str}")
        print("  - Update data/market/current_spreads.json with today's lines")
        print("")
        sys.exit(1)


if __name__ == "__main__":
    main()

