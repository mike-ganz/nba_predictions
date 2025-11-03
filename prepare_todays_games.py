#!/usr/bin/env python
"""Convenience wrapper to prepare today's games.

This script automatically uses today's date and calls prepare_future_games.py
with appropriate arguments for a typical daily prediction workflow.

NEW: Can optionally scrape odds and injuries automatically before preparing games!

Usage:
    # Today's games (manual scraping)
    python prepare_todays_games.py
    
    # Today's games with automatic scraping (RECOMMENDED)
    python prepare_todays_games.py --scrape-all
    
    # Specific date with scraping
    python prepare_todays_games.py --date 2025-11-15 --scrape-all
    
    # Scrape only odds or only injuries
    python prepare_todays_games.py --scrape-odds
    python prepare_todays_games.py --scrape-injuries
"""

import argparse
import subprocess
import sys
import logging
from datetime import datetime
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def scrape_odds(output_file: str, date: str = None, show_browser: bool = False) -> bool:
    """
    Scrape odds using Selenium.
    
    Args:
        output_file: Path to output JSON file
        date: Optional date filter
        show_browser: Show browser window for debugging
        
    Returns:
        True if successful, False otherwise
    """
    print("")
    print("=" * 70)
    print("SCRAPING ODDS (Selenium)")
    print("=" * 70)
    print("")
    
    cmd = [sys.executable, "scrape_odds_selenium.py", "--output", output_file, "--mode", "append"]
    
    if date:
        cmd.extend(["--date", date])
    
    if show_browser:
        cmd.append("--show-browser")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("")
        print("[OK] Odds scraped successfully")
        print("")
        return True
    else:
        print("")
        print("[ERROR] Failed to scrape odds", file=sys.stderr)
        print("")
        return False


def scrape_injuries(output_file: str) -> bool:
    """
    Scrape injuries from ESPN.
    
    Args:
        output_file: Path to output JSON file
        
    Returns:
        True if successful, False otherwise
    """
    print("")
    print("=" * 70)
    print("SCRAPING INJURIES (ESPN)")
    print("=" * 70)
    print("")
    
    cmd = [sys.executable, "scrape_injuries.py", "--output", output_file]
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("")
        print("[OK] Injuries scraped successfully")
        print("")
        return True
    else:
        print("")
        print("[ERROR] Failed to scrape injuries", file=sys.stderr)
        print("")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Prepare game data for today's (or specified date's) NBA games",
        epilog="TIP: Use --scrape-all to automatically fetch fresh odds and injury data!"
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
    
    # NEW: Scraping options
    parser.add_argument(
        "--scrape-all",
        action="store_true",
        help="Automatically scrape odds and injuries before preparing games (RECOMMENDED)"
    )
    parser.add_argument(
        "--scrape-odds",
        action="store_true",
        help="Scrape odds before preparing games"
    )
    parser.add_argument(
        "--scrape-injuries",
        action="store_true",
        help="Scrape injuries before preparing games"
    )
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Show browser window when scraping odds (for debugging)"
    )
    
    args = parser.parse_args()
    
    # Use today's date if not specified
    if args.date:
        date_str = args.date
        print(f"Using specified date: {date_str}")
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")
        print(f"Using today's date: {date_str}")
    
    # Handle scraping flags
    scrape_odds_flag = args.scrape_all or args.scrape_odds
    scrape_injuries_flag = args.scrape_all or args.scrape_injuries
    
    # Scrape odds if requested
    if scrape_odds_flag:
        success = scrape_odds(args.market, date_str, args.show_browser)
        if not success:
            print("[WARNING] Odds scraping failed, will try to use existing market data")
            print("")
    
    # Scrape injuries if requested
    if scrape_injuries_flag:
        success = scrape_injuries(args.injuries)
        if not success:
            print("[WARNING] Injury scraping failed, will try to use existing injury data")
            print("")
    
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
        
        if scrape_odds_flag or scrape_injuries_flag:
            print("")
            print("Data sources:")
            if scrape_odds_flag:
                print(f"  [LIVE] Odds scraped from evanalytics.com")
            if scrape_injuries_flag:
                print(f"  [LIVE] Injuries scraped from ESPN")
        
        print("")
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

