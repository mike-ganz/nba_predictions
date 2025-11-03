"""
NBA Odds Scraper with Selenium - Fully Automated

Scrapes NBA game odds (spread, total, moneyline) from evanalytics.com using Selenium.

Usage:
    python scrape_odds_selenium.py [--output OUTPUT_FILE] [--date YYYY-MM-DD] [--headless]

Requirements:
    pip install selenium webdriver-manager
"""

import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
import argparse
import sys

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Team name mapping (same as scrape_odds.py)
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


def parse_game_date_text(date_text: str) -> str:
    """
    Parse date from text like 'Sunday, November 02, 2025'.
    Returns date in YYYY-MM-DD format.
    """
    today = datetime.now().date()
    
    # Try to parse as full date string "Sunday, November 02, 2025"
    try:
        match = re.search(r'(\w+),\s+(\w+)\s+(\d+),\s+(\d{4})', date_text)
        if match:
            month_name = match.group(2)
            day = int(match.group(3))
            year = int(match.group(4))
            game_date = datetime.strptime(f"{month_name} {day} {year}", '%B %d %Y').date()
            return game_date.strftime('%Y-%m-%d')
    except (ValueError, AttributeError):
        pass
    
    # Fallback: try without year
    try:
        match = re.search(r'(\w+),?\s+(\w+)\s+(\d+)', date_text)
        if match:
            month_name = match.group(2)
            day = int(match.group(3))
            game_date = datetime.strptime(f"{month_name} {day} {today.year}", '%B %d %Y').date()
            # If date is in the past, assume next year
            if game_date < today:
                game_date = game_date.replace(year=today.year + 1)
            return game_date.strftime('%Y-%m-%d')
    except (ValueError, AttributeError):
        pass
    
    logging.warning(f"Could not parse date text: '{date_text}' - using today")
    return today.strftime('%Y-%m-%d')


def scrape_odds_with_selenium(headless: bool = True, target_date: Optional[str] = None) -> List[Dict]:
    """
    Scrape odds using Selenium from evanalytics.com.
    
    Args:
        headless: Run browser in headless mode (no GUI)
        target_date: Filter games for specific date (YYYY-MM-DD)
        
    Returns:
        List of game dictionaries with odds data
    """
    logging.info("Initializing Chrome WebDriver...")
    
    # Set up Chrome options
    chrome_options = Options()
    if headless:
        chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--window-size=1920,1080')
    
    # Add user agent
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    # Initialize driver
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
    except Exception as e:
        logging.error(f"Failed to initialize Chrome WebDriver: {e}")
        logging.error("Make sure Chrome is installed and you have internet connection for driver download")
        sys.exit(1)
    
    games = []
    
    try:
        url = "https://evanalytics.com/nba/odds"
        logging.info(f"Navigating to {url}")
        driver.get(url)
        
        # Wait for page to load
        logging.info("Waiting for page to load...")
        time.sleep(2)
        
        # Wait for table to appear
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
            logging.info("Page loaded successfully")
        except Exception as e:
            logging.error(f"Timeout waiting for page to load: {e}")
            driver.save_screenshot("error_screenshot.png")
            logging.error("Screenshot saved to error_screenshot.png")
            return games
        
        # Wait for odds data to load (JavaScript rendered)
        logging.info("Waiting for odds data to load...")
        time.sleep(5)  # Give extra time for odds data to populate
        
        # Click on "Game Line" tab to show only full game lines (not quarters/halves)
        try:
            game_line_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Game Line') or contains(., '1. Game Line')]")
            game_line_button.click()
            time.sleep(1)
            logging.info("Clicked 'Game Line' tab")
        except:
            logging.warning("Could not find/click 'Game Line' tab, proceeding anyway")
        
        # Find all table rows
        rows = driver.find_elements(By.TAG_NAME, "tr")
        logging.info(f"Found {len(rows)} table rows")
        
        current_date = None
        processed_games = set()
        
        for i, row in enumerate(rows):
            try:
                # Check if this row is a date header
                row_text = row.text.strip()
                
                # Date headers contain full dates like "Sunday, November 02, 2025"
                if re.search(r'\w+,\s+\w+\s+\d+,\s+\d{4}', row_text):
                    current_date = parse_game_date_text(row_text)
                    logging.info(f"Found date header: {row_text} -> {current_date}")
                    continue
                
                # Skip if we don't have a date yet
                if not current_date:
                    continue
                
                # Filter by target date if specified
                if target_date and current_date != target_date:
                    continue
                
                # Get all cells in the row
                cells = row.find_elements(By.TAG_NAME, "td")
                
                # We need at least 7 cells: Time, Team, ?, Spread, Totals, Moneyline, Win Prob
                if len(cells) < 7:
                    continue
                
                # Extract time cell - only process rows with valid game times
                time_cell = cells[0].text.strip()
                if not re.search(r'\d+:\d+\s+(AM|PM)', time_cell):
                    continue
                
                # Extract team cell - must contain both teams
                team_cell = cells[1]
                team_imgs = team_cell.find_elements(By.TAG_NAME, "img")
                
                # A full game line row should have 2 team logos (away and home)
                if len(team_imgs) < 2:
                    continue
                
                # Get team names from img alt tags
                away_team = team_imgs[0].get_attribute("alt").replace(" logo", "").strip()
                home_team = team_imgs[1].get_attribute("alt").replace(" logo", "").strip()
                
                if not away_team or not home_team:
                    continue
                
                # Check for duplicates
                game_key = (away_team, home_team, current_date)
                if game_key in processed_games:
                    continue
                
                logging.debug(f"Processing: {away_team} @ {home_team}")
                
                # Extract odds cells (indices based on evanalytics.com table structure)
                # cells[2] appears to be team ref/abbreviation
                spread_cell = cells[3].text.strip()  # Was cells[2]
                totals_cell = cells[4].text.strip()   # Was cells[3]
                moneyline_cell = cells[5].text.strip() # Was cells[4]
                
                # Debug: log cell contents for first game
                if len(games) == 0:
                    logging.info(f"First game cell contents:")
                    logging.info(f"  spread_cell: '{spread_cell}'")
                    logging.info(f"  totals_cell: '{totals_cell}'")
                    logging.info(f"  moneyline_cell: '{moneyline_cell}'")
                
                # Parse spread (format: "+13.0-110\n-13.0-110" - each line has spread+odds)
                # Extract just the spread values (before the odds)
                spread_lines = [line.strip() for line in spread_cell.split('\n') if line.strip()]
                if len(spread_lines) >= 2:
                    # Extract spread from first line (away): e.g., "+13.0" from "+13.0-110"
                    away_match = re.match(r'([+-]\d+\.?\d*)', spread_lines[0])
                    # Extract spread from second line (home): e.g., "-13.0" from "-13.0-110"
                    home_match = re.match(r'([+-]\d+\.?\d*)', spread_lines[1])
                    
                    if away_match and home_match:
                        away_spread = float(away_match.group(1))
                        home_spread = float(home_match.group(1))
                    else:
                        logging.warning(f"Could not parse spread for {away_team} @ {home_team}: '{spread_cell}'")
                        continue
                else:
                    logging.warning(f"Could not parse spread for {away_team} @ {home_team}: '{spread_cell}'")
                    continue
                
                # Parse totals (format: "o228.5 -110 u228.5 -110")
                total_match = re.search(r'[ou](\d+\.?\d*)', totals_cell.lower())
                if total_match:
                    total = float(total_match.group(1))
                else:
                    logging.warning(f"Could not parse total for {away_team} @ {home_team}: '{totals_cell}'")
                    continue
                
                # Parse moneyline (format: "+520 -680")
                ml_match = re.findall(r'([+-]\d+)', moneyline_cell)
                if len(ml_match) >= 2:
                    moneyline_away = int(ml_match[0])
                    moneyline_home = int(ml_match[1])
                else:
                    logging.warning(f"Could not parse moneyline for {away_team} @ {home_team}: '{moneyline_cell}'")
                    continue
                
                # Create game entry
                game = {
                    'away_team': normalize_team_name(away_team),
                    'home_team': normalize_team_name(home_team),
                    'game_date': current_date,
                    'spread_home': home_spread,
                    'total': total,
                    'moneyline_home': moneyline_home,
                    'moneyline_away': moneyline_away,
                }
                
                games.append(game)
                processed_games.add(game_key)
                logging.info(f"Scraped: {game['away_team']} @ {game['home_team']} ({current_date})")
                
            except Exception as e:
                logging.debug(f"Error processing row: {e}")
                continue
    
    finally:
        driver.quit()
        logging.info("Browser closed")
    
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
    
    if updated_count > 0 or added_count > 0:
        logging.info(f"Merge summary: {updated_count} updated, {added_count} added")
    
    # Return sorted by date then home team
    return sorted(existing_dict.values(), key=lambda g: (g['game_date'], g['home_team']))


def main():
    parser = argparse.ArgumentParser(description='Scrape NBA odds from evanalytics.com using Selenium')
    parser.add_argument('--output', type=str, default='data/market/current_spreads.json',
                        help='Output JSON file path')
    parser.add_argument('--date', type=str, help='Filter games for specific date (YYYY-MM-DD)')
    parser.add_argument('--mode', type=str, choices=['append', 'replace'], default='append',
                        help='append: merge with existing games (default), replace: overwrite file')
    parser.add_argument('--headless', action='store_true', default=True,
                        help='Run browser in headless mode (default: True)')
    parser.add_argument('--show-browser', action='store_true',
                        help='Show browser window (disable headless mode)')
    
    args = parser.parse_args()
    
    # Scrape odds
    headless = args.headless and not args.show_browser
    games = scrape_odds_with_selenium(headless=headless, target_date=args.date)
    
    if not games:
        logging.error("No games found!")
        logging.error("This could mean:")
        logging.error("  1. No games scheduled for the specified date")
        logging.error("  2. The website structure has changed")
        logging.error("  3. The page failed to load properly (try --show-browser to debug)")
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
        "source": "evanalytics.com",
        "source_url": "https://evanalytics.com/nba/odds"
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
    if games:
        print("Newly scraped games:")
        for game in games:
            print(f"{game['game_date']}: {game['away_team']} @ {game['home_team']}")
            print(f"  Spread: {game['spread_home']:+.1f} (home)")
            print(f"  Total: {game['total']}")
            print(f"  ML: {game['moneyline_away']:+d} / {game['moneyline_home']:+d}")
            print("")
    
    print("=" * 70)


if __name__ == "__main__":
    main()

