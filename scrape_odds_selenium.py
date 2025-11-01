"""
NBA Odds Scraper with Selenium - Fully Automated

Scrapes NBA game odds (spread, total, moneyline) from oddschecker.com using Selenium
to bypass Cloudflare protection.

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
    Parse date from text like 'Today' or 'Tomorrow'.
    Returns date in YYYY-MM-DD format.
    """
    today = datetime.now().date()
    
    if 'today' in date_text.lower():
        game_date = today
    elif 'tomorrow' in date_text.lower():
        game_date = today + timedelta(days=1)
    else:
        # Try to parse as a date string
        try:
            # Try format like "Thursday, December 25"
            match = re.search(r'(\w+),?\s+(\w+)\s+(\d+)', date_text)
            if match:
                month_name = match.group(2)
                day = int(match.group(3))
                game_date = datetime.strptime(f"{month_name} {day} {today.year}", '%B %d %Y').date()
                # If date is in the past, assume next year
                if game_date < today:
                    game_date = game_date.replace(year=today.year + 1)
            else:
                logging.warning(f"Could not parse date text: '{date_text}' - using today")
                game_date = today
        except ValueError:
            logging.warning(f"Could not parse date text: '{date_text}' - using today")
            game_date = today
    
    return game_date.strftime('%Y-%m-%d')


def scrape_odds_with_selenium(headless: bool = True, target_date: Optional[str] = None) -> List[Dict]:
    """
    Scrape odds using Selenium to automate browser interaction.
    
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
    
    # Add user agent to avoid detection
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
        url = "https://www.oddschecker.com/us/basketball/nba"
        logging.info(f"Navigating to {url}")
        driver.get(url)
        
        # Wait for page to load - look for game elements
        logging.info("Waiting for page to load...")
        time.sleep(5)  # Give Cloudflare time to pass
        
        # Try to find game container sections
        try:
            # Wait for game elements to appear
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "article"))
            )
            logging.info("Page loaded successfully")
        except Exception as e:
            logging.error(f"Timeout waiting for page to load: {e}")
            logging.error("The page structure may have changed or Cloudflare is blocking")
            driver.save_screenshot("error_screenshot.png")
            logging.error("Screenshot saved to error_screenshot.png")
            return games
        
        # Find all game sections
        game_sections = driver.find_elements(By.XPATH, "//article//div[contains(@class, 'flex-col')]")
        
        logging.info(f"Found {len(game_sections)} potential game sections")
        
        current_date_label = None
        
        for section in game_sections:
            try:
                section_html = section.get_attribute('innerHTML')
                
                # Check if this section has a date label
                date_elements = section.find_elements(By.TAG_NAME, "p")
                for elem in date_elements:
                    text = elem.text.strip()
                    if text in ['Today', 'Tomorrow'] or re.search(r'\w+,\s+\w+\s+\d+', text):
                        current_date_label = text
                        logging.debug(f"Found date label: {current_date_label}")
                        break
                
                # Look for team names in links
                game_links = section.find_elements(By.TAG_NAME, "a")
                
                for link in game_links:
                    link_text = link.text.strip()
                    
                    # Check if this is a game link (contains @ symbol)
                    if '@' in link_text:
                        # Extract teams from link text
                        # Format: "Away Team @ Home Team" or "Away Team logo @ Home Team"
                        parts = link_text.split('@')
                        if len(parts) == 2:
                            away_team = parts[0].strip()
                            home_team = parts[1].strip()
                            
                            # Clean up team names (remove "logo" text)
                            away_team = re.sub(r'\s+logo\s*$', '', away_team, flags=re.IGNORECASE).strip()
                            home_team = re.sub(r'^logo\s+', '', home_team, flags=re.IGNORECASE).strip()
                            
                            if not away_team or not home_team:
                                continue
                            
                            # Get game date
                            game_date = parse_game_date_text(current_date_label) if current_date_label else datetime.now().strftime('%Y-%m-%d')
                            
                            # Filter by date if specified
                            if target_date and game_date != target_date:
                                continue
                            
                            logging.debug(f"Found game: {away_team} @ {home_team} on {game_date}")
                            
                            # Now find odds for this game (in the same section or nearby)
                            # Look for buttons with odds data
                            parent_section = link.find_element(By.XPATH, "./ancestor::div[contains(@class, 'flex-col')]")
                            
                            # Find spread, total, and moneyline buttons
                            odds_buttons = parent_section.find_elements(By.TAG_NAME, "button")
                            
                            spread_away = None
                            spread_home = None
                            total = None
                            moneyline_away = None
                            moneyline_home = None
                            
                            for button in odds_buttons:
                                button_text = button.text.strip()
                                
                                # Spread pattern: "+6.5 -106" or "-6.5 -105"
                                spread_match = re.match(r'([+-]\d+\.?\d*)\s+([+-]\d+)', button_text)
                                if spread_match:
                                    if spread_away is None:
                                        spread_away = parse_spread(spread_match.group(1))
                                    elif spread_home is None:
                                        spread_home = parse_spread(spread_match.group(1))
                                    continue
                                
                                # Total pattern: "O 233 -110" or "U 233 -110"
                                total_match = re.match(r'[OU]\s+([\d.]+)\s+([+-]\d+)', button_text)
                                if total_match and total is None and button_text.startswith('O'):
                                    total = parse_total(total_match.group(1))
                                    continue
                                
                                # Moneyline pattern: "+210" or "-250"
                                ml_match = re.match(r'^([+-]\d+)$', button_text)
                                if ml_match:
                                    if moneyline_away is None:
                                        moneyline_away = parse_odds_american(ml_match.group(1))
                                    elif moneyline_home is None:
                                        moneyline_home = parse_odds_american(ml_match.group(1))
                                    continue
                            
                            # Validate we have all required data
                            if all(x is not None for x in [spread_home, total, moneyline_away, moneyline_home]):
                                game = {
                                    'away_team': normalize_team_name(away_team),
                                    'home_team': normalize_team_name(home_team),
                                    'game_date': game_date,
                                    'spread_home': spread_home,
                                    'total': total,
                                    'moneyline_home': moneyline_home,
                                    'moneyline_away': moneyline_away,
                                }
                                games.append(game)
                                logging.info(f"Scraped: {game['away_team']} @ {game['home_team']}")
                            else:
                                logging.warning(f"Incomplete odds data for {away_team} @ {home_team}")
                                logging.debug(f"  spread_home={spread_home}, total={total}, ml_away={moneyline_away}, ml_home={moneyline_home}")
            
            except Exception as e:
                logging.debug(f"Error processing section: {e}")
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
    parser = argparse.ArgumentParser(description='Scrape NBA odds from oddschecker.com using Selenium')
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
        logging.error("  2. Cloudflare is still blocking (try --show-browser to debug)")
        logging.error("  3. The website structure has changed")
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

