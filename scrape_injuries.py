"""Scrape NBA injury data from ESPN and save to JSON format.

Usage:
    python scrape_injuries.py
    python scrape_injuries.py --output data/injuries/current_injuries.json
"""

import argparse
import json
import re
from datetime import datetime
from typing import List, Dict

import requests
from bs4 import BeautifulSoup


# Team name normalization mapping (ESPN -> Our format)
TEAM_NAME_MAPPING = {
    "Atlanta Hawks": "Atlanta",
    "Boston Celtics": "Boston",
    "Brooklyn Nets": "Brooklyn",
    "Charlotte Hornets": "Charlotte",
    "Chicago Bulls": "Chicago",
    "Cleveland Cavaliers": "Cleveland",
    "Dallas Mavericks": "Dallas",
    "Denver Nuggets": "Denver",
    "Detroit Pistons": "Detroit",
    "Golden State Warriors": "Golden State",
    "Houston Rockets": "Houston",
    "Indiana Pacers": "Indiana",
    "LA Clippers": "LA Clippers",
    "Los Angeles Lakers": "LA Lakers",
    "Memphis Grizzlies": "Memphis",
    "Miami Heat": "Miami",
    "Milwaukee Bucks": "Milwaukee",
    "Minnesota Timberwolves": "Minnesota",
    "New Orleans Pelicans": "New Orleans",
    "New York Knicks": "New York",
    "Oklahoma City Thunder": "Oklahoma City",
    "Orlando Magic": "Orlando",
    "Philadelphia 76ers": "Philadelphia",
    "Phoenix Suns": "Phoenix",
    "Portland Trail Blazers": "Portland",
    "Sacramento Kings": "Sacramento",
    "San Antonio Spurs": "San Antonio",
    "Toronto Raptors": "Toronto",
    "Utah Jazz": "Utah",
    "Washington Wizards": "Washington",
}


def scrape_espn_injuries() -> List[Dict]:
    """
    Scrape injury data from ESPN NBA injuries page.
    
    Returns:
        List of injury dictionaries with player_name, team, status, injury, comment
    """
    url = "https://www.espn.com/nba/injuries"
    
    print(f"Fetching injury data from {url}...")
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    response.raise_for_status()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    injuries = []
    
    # Find all team sections - ESPN wraps each team in a div
    # Look for all tables which contain injury data
    all_tables = soup.find_all('table')
    
    # We need to find team names by looking backwards from tables
    # Each table is preceded by a div containing the team name/image
    for table in all_tables:
        # Find the parent container
        parent = table.find_parent()
        
        # Look for team name in preceding siblings or parent
        team_name = None
        search_element = table
        for _ in range(5):  # Search up to 5 levels up
            if search_element is None:
                break
            search_element = search_element.find_previous()
            if search_element and search_element.name in ['div', 'span', 'h2', 'h3']:
                # Look for team name in text or img alt
                text = search_element.get_text().strip()
                if text in TEAM_NAME_MAPPING:
                    team_name = TEAM_NAME_MAPPING[text]
                    break
                
                # Check for img with alt text
                img = search_element.find('img', alt=True)
                if img and img['alt'] in TEAM_NAME_MAPPING:
                    team_name = TEAM_NAME_MAPPING[img['alt']]
                    break
        
        if not team_name:
            continue  # Skip tables without identifiable teams
        
        print(f"Found team: {team_name}")
        
        # Parse the table
        tbody = table.find('tbody')
        if not tbody:
            continue
        
        rows = tbody.find_all('tr')
        for row in rows:
            cells = row.find_all('td')
            if len(cells) < 4:
                continue
            
            # Extract data - ESPN structure: NAME, POS, EST. RETURN, STATUS, COMMENT
            player_link = cells[0].find('a')
            if not player_link:
                continue
            
            player_name = player_link.text.strip()
            position = cells[1].text.strip() if len(cells) > 1 else ""
            est_return = cells[2].text.strip() if len(cells) > 2 else ""
            status = cells[3].text.strip() if len(cells) > 3 else ""
            comment = cells[4].text.strip() if len(cells) > 4 else ""
            
            # Extract injury type from comment
            injury_type = extract_injury_type(comment)
            
            # Normalize status
            status_normalized = normalize_status(status)
            
            injury_record = {
                "player_name": player_name,
                "team": team_name,
                "status": status_normalized,
                "injury": injury_type,
                "position": position,
                "est_return": est_return,
                "comment": comment[:200]  # Truncate long comments
            }
            
            injuries.append(injury_record)
            print(f"  - {player_name} ({status_normalized}): {injury_type}")
    
    return injuries


def extract_injury_type(comment: str) -> str:
    """Extract injury type from comment text."""
    if not comment:
        return "unknown"
    
    # Look for injury type in parentheses
    match = re.search(r'\(([^)]+)\)', comment)
    if match:
        injury = match.group(1).lower()
        # Clean up common patterns
        injury = injury.replace('left ', '').replace('right ', '')
        return injury
    
    # Look for common injury keywords
    injury_keywords = [
        'knee', 'ankle', 'hamstring', 'shoulder', 'back', 'foot', 'wrist',
        'hip', 'groin', 'calf', 'achilles', 'hand', 'thumb', 'finger',
        'elbow', 'toe', 'neck', 'quad', 'illness', 'personal', 'rest'
    ]
    
    comment_lower = comment.lower()
    for keyword in injury_keywords:
        if keyword in comment_lower:
            return keyword
    
    return "unknown"


def normalize_status(status: str) -> str:
    """Normalize status to our standard values."""
    status_lower = status.lower().strip()
    
    if status_lower in ['out', 'o']:
        return 'out'
    elif status_lower in ['day-to-day', 'dtd', 'questionable', 'q']:
        return 'questionable'
    elif status_lower in ['doubtful', 'd']:
        return 'doubtful'
    elif status_lower in ['probable', 'p']:
        return 'probable'
    else:
        return status_lower


def main():
    parser = argparse.ArgumentParser(description="Scrape ESPN NBA injury data")
    parser.add_argument(
        "--output",
        type=str,
        default="data/injuries/current_injuries.json",
        help="Output JSON file path"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("ESPN NBA INJURY SCRAPER")
    print("=" * 70)
    print()
    
    try:
        injuries = scrape_espn_injuries()
        
        if not injuries:
            print("\nWARNING: No injuries found! Check if ESPN page structure changed.")
            print("Creating empty injuries file...")
            injuries = []
        
        # Create output structure
        output_data = {
            "injuries": injuries,
            "last_updated": datetime.now().isoformat(),
            "source": "ESPN",
            "source_url": "https://www.espn.com/nba/injuries"
        }
        
        # Save to JSON
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print()
        print("=" * 70)
        print(f"SUCCESS: Scraped {len(injuries)} injury records")
        print(f"Saved to: {args.output}")
        print("=" * 70)
        print()
        
        # Summary by status
        status_counts = {}
        for injury in injuries:
            status = injury['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print("Summary by status:")
        for status, count in sorted(status_counts.items()):
            print(f"  {status}: {count} players")
        
    except requests.RequestException as e:
        print(f"\nERROR: Failed to fetch data from ESPN: {e}")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

