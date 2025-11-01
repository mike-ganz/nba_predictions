"""
NBA Odds Scraper for Oddschecker.com

Scrapes NBA game odds (spread, total, moneyline) from oddschecker.com and outputs
to the format expected by prepare_future_games.py:
{
    "home_team": "Team Name",
    "away_team": "Team Name",
    "game_date": "YYYY-MM-DD",
    "spread_home": float,  # negative if home favored
    "total": float,
    "moneyline_home": int,
    "moneyline_away": int
}

Usage:
    python scrape_odds.py [--output OUTPUT_FILE] [--date YYYY-MM-DD]

Requirements:
    - Must be run with browser automation (uses browser snapshot data)
    - Oddschecker.com has Cloudflare protection, so direct requests don't work
"""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
import argparse
import sys

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Team name mapping from oddschecker format to our standard names
TEAM_NAME_MAPPING = {
    'Atlanta Hawks': 'Atlanta Hawks',
    'Boston Celtics': 'Boston Celtics',
    'Brooklyn Nets': 'Brooklyn Nets',
    'Charlotte Hornets': 'Charlotte Hornets',
    'Chicago Bulls': 'Chicago Bulls',
    'Cleveland Cavaliers': 'Cleveland Cavaliers',
    'Dallas Mavericks': 'Dallas Mavericks',
    'Denver Nuggets': 'Denver Nuggets',
    'Detroit Pistons': 'Detroit Pistons',
    'Golden State Warriors': 'Golden State Warriors',
    'Houston Rockets': 'Houston Rockets',
    'Indiana Pacers': 'Indiana Pacers',
    'LA Clippers': 'Los Angeles Clippers',
    'Los Angeles Clippers': 'Los Angeles Clippers',
    'LA Lakers': 'Los Angeles Lakers',
    'Los Angeles Lakers': 'Los Angeles Lakers',
    'Memphis Grizzlies': 'Memphis Grizzlies',
    'Miami Heat': 'Miami Heat',
    'Milwaukee Bucks': 'Milwaukee Bucks',
    'Minnesota Timberwolves': 'Minnesota Timberwolves',
    'New Orleans Pelicans': 'New Orleans Pelicans',
    'New York Knicks': 'New York Knicks',
    'Oklahoma City Thunder': 'Oklahoma City Thunder',
    'Orlando Magic': 'Orlando Magic',
    'Philadelphia 76ers': 'Philadelphia 76ers',
    'Phoenix Suns': 'Phoenix Suns',
    'Portland Trail Blazers': 'Portland Trail Blazers',
    'Sacramento Kings': 'Sacramento Kings',
    'San Antonio Spurs': 'San Antonio Spurs',
    'Toronto Raptors': 'Toronto Raptors',
    'Utah Jazz': 'Utah Jazz',
    'Washington Wizards': 'Washington Wizards',
}


def normalize_team_name(team: str) -> str:
    """Normalize team name to match our standard format."""
    team = team.strip()
    
    # Handle "@ Team Name" format
    if team.startswith('@'):
        team = team[1:].strip()
    
    # Look up in mapping
    if team in TEAM_NAME_MAPPING:
        return TEAM_NAME_MAPPING[team]
    
    # If not found, return as-is and log warning
    logging.warning(f"Unknown team name format: '{team}' - using as-is")
    return team


def parse_odds_american(odds_str: str) -> int:
    """Parse American odds format (e.g., '-110', '+200') to integer."""
    odds_str = odds_str.strip()
    if odds_str.startswith('+') or odds_str.startswith('-'):
        return int(odds_str)
    return int(f"+{odds_str}")  # Assume positive if no sign


def parse_spread(spread_str: str) -> float:
    """Parse spread string (e.g., '-6.5', '+11') to float."""
    spread_str = spread_str.strip()
    return float(spread_str)


def parse_total(total_str: str) -> float:
    """Parse total string (e.g., 'O 233', '233.5') to float."""
    # Remove O/U prefix if present
    total_str = total_str.strip().upper()
    if total_str.startswith('O ') or total_str.startswith('U '):
        total_str = total_str[2:]
    return float(total_str)


def parse_game_date(date_label: str, game_time: str) -> str:
    """
    Parse date from label like 'Today' or 'Tomorrow' and time like '5:00 PM EDT'.
    Returns date in YYYY-MM-DD format.
    """
    today = datetime.now().date()
    
    if date_label.lower() == 'today':
        game_date = today
    elif date_label.lower() == 'tomorrow':
        game_date = today + timedelta(days=1)
    else:
        # Try to parse as a date string
        try:
            game_date = datetime.strptime(date_label, '%B %d').replace(year=today.year).date()
            # If date is in the past, assume next year
            if game_date < today:
                game_date = game_date.replace(year=today.year + 1)
        except ValueError:
            logging.warning(f"Could not parse date label: '{date_label}' - using today")
            game_date = today
    
    return game_date.strftime('%Y-%m-%d')


def scrape_from_snapshot_file(snapshot_file: str, target_date: Optional[str] = None) -> List[Dict]:
    """
    Parse games from a browser snapshot file.
    
    The snapshot file should contain the accessibility tree from oddschecker.com/us/basketball/nba
    """
    logging.info(f"Reading snapshot file: {snapshot_file}")
    
    with open(snapshot_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    games = []
    current_game = {}
    current_date_label = None
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for date labels (Today, Tomorrow, etc.)
        if 'paragraph' in line and ('Today' in line or 'Tomorrow' in line or re.search(r'\w+ \d+', line)):
            date_match = re.search(r'paragraph.*?: (.+)', line)
            if date_match:
                current_date_label = date_match.group(1).strip()
                logging.debug(f"Found date label: {current_date_label}")
        
        # Look for game time and teams
        if 'PM EDT' in line or 'AM EDT' in line or 'PM EST' in line or 'AM EST' in line:
            time_match = re.search(r'generic.*?: (\d+:\d+ [AP]M [A-Z]+)', line)
            if time_match:
                game_time = time_match.group(1)
                logging.debug(f"Found game time: {game_time}")
                
                # Next lines should have the teams link and then paragraph elements with team names
                i += 1
                if i < len(lines):
                    teams_link_line = lines[i].strip()
                    
                    # Verify this is a teams link
                    if 'link' in teams_link_line and '@' in teams_link_line:
                        # Look ahead for paragraph elements with team names
                        # Structure:
                        #   - link "..."
                        #     - /url: ...
                        #     - generic:
                        #       - img "Away Team logo"
                        #       - paragraph: Away Team Name
                        #     - generic:
                        #       - img "Home Team logo"  
                        #       - paragraph: "@ Home Team Name"
                        
                        away_team = None
                        home_team = None
                        
                        # Scan next 20 lines for team paragraph elements
                        for j in range(i, min(i + 20, len(lines))):
                            para_line = lines[j].strip()
                            
                            # Look for paragraph with team name (not starting with @)
                            para_match = re.search(r'paragraph \[ref=\w+\]: (?!")([^"]+)$', para_line)
                            if para_match and not away_team:
                                away_team = para_match.group(1).strip()
                                logging.debug(f"Found away team: {away_team}")
                                continue
                            
                            # Look for paragraph with "@ Team Name"
                            para_match_home = re.search(r'paragraph \[ref=\w+\]: "@ (.+)"', para_line)
                            if para_match_home:
                                home_team = para_match_home.group(1).strip()
                                logging.debug(f"Found home team: {home_team}")
                                break
                        
                        if away_team and home_team:
                            current_game = {
                                'away_team': normalize_team_name(away_team),
                                'home_team': normalize_team_name(home_team),
                                'game_date': parse_game_date(current_date_label, game_time) if current_date_label else None,
                                'game_time': game_time,
                            }
                            
                            logging.debug(f"Found game: {current_game['away_team']} @ {current_game['home_team']}")
                        else:
                            logging.warning(f"Could not extract teams from game at {game_time}")
        
        # Look for spread odds
        if current_game and 'spread_home' not in current_game:
            # Spread buttons come in pairs: away spread, home spread
            # Format: button "+6.5 -106" or button "-6.5 -105"
            spread_match = re.search(r'button "([+-]\d+\.?\d*)\s+([+-]\d+)"', line)
            if spread_match:
                spread_value = spread_match.group(1)
                spread_odds = spread_match.group(2)
                
                # First spread button is away team
                if 'away_spread' not in current_game:
                    current_game['away_spread'] = parse_spread(spread_value)
                    current_game['away_spread_odds'] = parse_odds_american(spread_odds)
                else:
                    # Second spread button is home team
                    current_game['home_spread'] = parse_spread(spread_value)
                    current_game['home_spread_odds'] = parse_odds_american(spread_odds)
                    
                    # Verify spread logic: home spread should be negative of away spread
                    expected_home = -current_game['away_spread']
                    if abs(current_game['home_spread'] - expected_home) > 0.1:
                        logging.warning(
                            f"Spread mismatch for {current_game.get('away_team')} @ {current_game.get('home_team')}: "
                            f"away={current_game['away_spread']}, home={current_game['home_spread']}"
                        )
                    
                    # Store home spread (negative if favored)
                    current_game['spread_home'] = current_game['home_spread']
        
        # Look for totals
        if current_game and 'total' not in current_game:
            # Total buttons: "O 233 -110" and "U 233 -110"
            total_match = re.search(r'button "[OU] ([\d.]+)\s+([+-]\d+)"', line)
            if total_match and 'O ' in line:  # Only process Over line
                total_value = total_match.group(1)
                current_game['total'] = parse_total(total_value)
        
        # Look for moneylines
        if current_game and 'moneyline_home' not in current_game:
            # Moneyline buttons: "+210" or "-250"
            ml_match = re.search(r'button "([+-]\d+)"', line)
            if ml_match:
                ml_value = ml_match.group(1)
                
                # First moneyline is away, second is home
                if 'moneyline_away' not in current_game:
                    current_game['moneyline_away'] = parse_odds_american(ml_value)
                else:
                    current_game['moneyline_home'] = parse_odds_american(ml_value)
                    
                    # Game is complete, add to list
                    if all(k in current_game for k in ['away_team', 'home_team', 'spread_home', 'total', 'moneyline_away', 'moneyline_home']):
                        # Filter by date if specified
                        if target_date is None or current_game.get('game_date') == target_date:
                            games.append({
                                'away_team': current_game['away_team'],
                                'home_team': current_game['home_team'],
                                'game_date': current_game['game_date'],
                                'spread_home': current_game['spread_home'],
                                'total': current_game['total'],
                                'moneyline_home': current_game['moneyline_home'],
                                'moneyline_away': current_game['moneyline_away'],
                            })
                            logging.info(f"Scraped: {current_game['away_team']} @ {current_game['home_team']}")
                        
                        current_game = {}  # Reset for next game
        
        i += 1
    
    return games


def merge_games(existing_games: List[Dict], new_games: List[Dict]) -> List[Dict]:
    """
    Merge new games with existing games. Updates existing games if found,
    adds new games otherwise. Match is based on away_team + home_team + game_date.
    """
    # Create lookup for existing games
    game_key = lambda g: (g['away_team'], g['home_team'], g['game_date'])
    existing_dict = {game_key(g): g for g in existing_games}
    
    # Update/add new games
    updated_count = 0
    added_count = 0
    
    for new_game in new_games:
        key = game_key(new_game)
        if key in existing_dict:
            # Update existing game
            existing_dict[key].update(new_game)
            updated_count += 1
            logging.info(f"Updated: {new_game['away_team']} @ {new_game['home_team']}")
        else:
            # Add new game
            existing_dict[key] = new_game
            added_count += 1
            logging.info(f"Added: {new_game['away_team']} @ {new_game['home_team']}")
    
    logging.info(f"Merge summary: {updated_count} updated, {added_count} added")
    
    # Return sorted by date then home team
    return sorted(existing_dict.values(), key=lambda g: (g['game_date'], g['home_team']))


def main():
    parser = argparse.ArgumentParser(description='Scrape NBA odds from oddschecker.com')
    parser.add_argument('--snapshot', type=str, help='Path to browser snapshot file (for testing)')
    parser.add_argument('--output', type=str, default='data/market/current_spreads.json',
                        help='Output JSON file path')
    parser.add_argument('--date', type=str, help='Filter games for specific date (YYYY-MM-DD)')
    parser.add_argument('--mode', type=str, choices=['append', 'replace'], default='append',
                        help='append: merge with existing games (default), replace: overwrite file')
    
    args = parser.parse_args()
    
    if args.snapshot:
        # Use provided snapshot file
        games = scrape_from_snapshot_file(args.snapshot, args.date)
    else:
        print("ERROR: This scraper requires browser automation")
        print("")
        print("Since oddschecker.com has Cloudflare protection, we can't use direct HTTP requests.")
        print("Instead, you need to:")
        print("  1. Open the NBA odds page in your browser")
        print("  2. Take a snapshot using browser tools")
        print("  3. Pass the snapshot file to this script")
        print("")
        print("Example:")
        print("  python scrape_odds.py --snapshot C:\\path\\to\\snapshot.log")
        print("")
        print("Or see ODDS_SCRAPER.md for automated workflow")
        sys.exit(1)
    
    if not games:
        logging.error("No games found in snapshot!")
        sys.exit(1)
    
    # Output to file
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing games if in append mode
    existing_games = []
    if args.mode == 'append' and output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                
                # Handle both formats: array or object with "games" key
                if isinstance(existing_data, list):
                    existing_games = existing_data
                elif isinstance(existing_data, dict) and 'games' in existing_data:
                    existing_games = existing_data['games']
                
                logging.info(f"Loaded {len(existing_games)} existing games")
        except Exception as e:
            logging.warning(f"Could not load existing games: {e}")
    
    # Merge if in append mode
    if args.mode == 'append' and existing_games:
        all_games = merge_games(existing_games, games)
    else:
        all_games = games
    
    # Prepare output structure (matching the existing format)
    output_data = {
        "games": all_games,
        "last_updated": datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
        "source": "oddschecker.com",
        "source_url": "https://www.oddschecker.com/us/basketball/nba"
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    logging.info(f"Saved {len(all_games)} games to {output_file}")
    
    # Print summary
    print("")
    print("=" * 70)
    print("ODDS SCRAPING COMPLETE")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Newly scraped: {len(games)} games")
    print(f"Total in file: {len(all_games)} games")
    print(f"Output file: {output_file}")
    print("")
    
    # Show newly scraped games
    print("Newly scraped games:")
    for game in games:
        print(f"{game['game_date']}: {game['away_team']} @ {game['home_team']}")
        print(f"  Spread: {game['spread_home']:+.1f} (home)")
        print(f"  Total: {game['total']}")
        print(f"  ML: {game['moneyline_away']:+d} / {game['moneyline_home']:+d}")
        print("")


if __name__ == "__main__":
    main()

