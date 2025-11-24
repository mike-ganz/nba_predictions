"""
Script to scrape player game log data from BettingPros
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pandas as pd
import time
import re
import numpy as np
import os


def extract_table_data(driver, prefer_game_log=True):
    """
    Extract table data from the current page
    
    Args:
        driver: Selenium webdriver instance
        prefer_game_log (bool): If True, look for the game log table specifically
        
    Returns:
        pandas.DataFrame: Extracted table data
    """
    wait = WebDriverWait(driver, 15)
    
    # Find all tables on the page
    print("Looking for tables on the page...")
    tables = driver.find_elements(By.TAG_NAME, "table")
    print(f"Found {len(tables)} tables")
    
    table = None
    
    if prefer_game_log and len(tables) > 1:
        # Try to identify the game log table by checking for expected columns
        print("Multiple tables found, looking for game log table...")
        for idx, tbl in enumerate(tables):
            try:
                # Check headers of this table
                header_elements = tbl.find_elements(By.CSS_SELECTOR, "thead tr th")
                if not header_elements:
                    header_elements = tbl.find_elements(By.CSS_SELECTOR, "tr:first-child th")
                
                headers_text = [h.text.strip() for h in header_elements]
                print(f"Table {idx} headers: {headers_text[:5]}...")  # Print first 5
                
                # Game log table should have Date and Matchup columns
                if any('Date' in h for h in headers_text) or any('Matchup' in h for h in headers_text):
                    print(f"Found game log table (table {idx})")
                    table = tbl
                    break
            except Exception as e:
                print(f"Error checking table {idx}: {e}")
                continue
    
    # If no game log table found, use the first table
    if not table and tables:
        table = tables[0]
        print(f"Using first table")
    
    if not table:
        raise Exception("No tables found on page")
    
    # Extract table data
    print("Extracting table data...")
    headers = []
    rows_data = []
    
    # Get headers
    try:
        header_elements = table.find_elements(By.CSS_SELECTOR, "thead tr th")
        if not header_elements:
            header_elements = table.find_elements(By.CSS_SELECTOR, "tr:first-child th")
        if not header_elements:
            header_elements = table.find_elements(By.CSS_SELECTOR, "tr:first-child td")
        
        headers = [header.text.strip() for header in header_elements]
        print(f"Headers found: {headers}")
    except Exception as e:
        print(f"Error extracting headers: {e}")
    
    # Get data rows
    try:
        tbody = table.find_element(By.TAG_NAME, "tbody")
        row_elements = tbody.find_elements(By.TAG_NAME, "tr")
    except NoSuchElementException:
        row_elements = table.find_elements(By.TAG_NAME, "tr")[1:]
    
    print(f"Found {len(row_elements)} data rows")
    
    for row in row_elements:
        try:
            cells = row.find_elements(By.TAG_NAME, "td")
            if not cells:
                cells = row.find_elements(By.TAG_NAME, "th")
            
            if cells:
                row_data = [cell.text.strip() for cell in cells]
                rows_data.append(row_data)
        except Exception as e:
            print(f"Error extracting row: {e}")
            continue
    
    # Create DataFrame
    if headers and rows_data:
        df = pd.DataFrame(rows_data, columns=headers)
    elif rows_data:
        df = pd.DataFrame(rows_data)
    else:
        df = pd.DataFrame()
    
    print(f"Successfully extracted {len(df)} rows")
    return df


def clean_game_log_data(df):
    """
    Clean and split combined columns into granular data
    
    Args:
        df (pandas.DataFrame): Raw scraped data
        
    Returns:
        pandas.DataFrame: Cleaned data with split columns
    """
    df_clean = df.copy()
    
    # 1. Clean ASSISTS column: "O 13" or "U 6" -> assists (numeric) + over_under_result
    if 'ASSISTS' in df_clean.columns:
        def parse_assists(val):
            if pd.isna(val) or str(val).strip() == '':
                return pd.Series({'assists': np.nan, 'over_under_result': None})
            
            val_str = str(val).strip()
            match = re.match(r'([OU])\s*(\d+)', val_str)
            if match:
                return pd.Series({
                    'over_under_result': match.group(1),
                    'assists': int(match.group(2))
                })
            
            # Try just extracting number if no O/U prefix
            match_num = re.search(r'(\d+)', val_str)
            if match_num:
                return pd.Series({
                    'over_under_result': None,
                    'assists': int(match_num.group(1))
                })
            
            return pd.Series({'assists': np.nan, 'over_under_result': None})
        
        assists_split = df_clean['ASSISTS'].apply(parse_assists)
        df_clean['assists'] = assists_split['assists']
        df_clean['over_under_result'] = assists_split['over_under_result']
        df_clean = df_clean.drop('ASSISTS', axis=1)
    
    # 2. Clean Score column: "L 130-127" -> win_loss + team_score + opponent_score
    if 'Score' in df_clean.columns:
        def parse_score(val):
            if pd.isna(val) or str(val).strip() == '':
                return pd.Series({'win_loss': None, 'team_score': np.nan, 'opponent_score': np.nan})
            
            val_str = str(val).strip()
            match = re.match(r'([WL])\s*(\d+)-(\d+)', val_str)
            if match:
                return pd.Series({
                    'win_loss': match.group(1),
                    'team_score': int(match.group(2)),
                    'opponent_score': int(match.group(3))
                })
            
            return pd.Series({'win_loss': None, 'team_score': np.nan, 'opponent_score': np.nan})
        
        score_split = df_clean['Score'].apply(parse_score)
        df_clean['win_loss'] = score_split['win_loss']
        df_clean['team_score'] = score_split['team_score']
        df_clean['opponent_score'] = score_split['opponent_score']
        df_clean = df_clean.drop('Score', axis=1)
    
    # 3. Clean FGM-FGA (%): "13-27 (48%)" -> fgm, fga, fg_pct
    if 'FGM-FGA (%)' in df_clean.columns:
        def parse_shooting(val):
            if pd.isna(val) or str(val).strip() == '' or str(val).strip() == '-':
                return pd.Series({'made': np.nan, 'attempted': np.nan, 'pct': np.nan})
            
            val_str = str(val).strip()
            match = re.match(r'(\d+)-(\d+)\s*\((\d+(?:\.\d+)?)%\)', val_str)
            if match:
                return pd.Series({
                    'made': int(match.group(1)),
                    'attempted': int(match.group(2)),
                    'pct': float(match.group(3))
                })
            
            return pd.Series({'made': np.nan, 'attempted': np.nan, 'pct': np.nan})
        
        fg_split = df_clean['FGM-FGA (%)'].apply(parse_shooting)
        df_clean['fgm'] = fg_split['made']
        df_clean['fga'] = fg_split['attempted']
        df_clean['fg_pct'] = fg_split['pct']
        df_clean = df_clean.drop('FGM-FGA (%)', axis=1)
    
    # 4. Clean FTM-FTA (%): "9-12 (75%)" -> ftm, fta, ft_pct
    if 'FTM-FTA (%)' in df_clean.columns:
        def parse_shooting(val):
            if pd.isna(val) or str(val).strip() == '' or str(val).strip() == '-':
                return pd.Series({'made': np.nan, 'attempted': np.nan, 'pct': np.nan})
            
            val_str = str(val).strip()
            match = re.match(r'(\d+)-(\d+)\s*\((\d+(?:\.\d+)?)%\)', val_str)
            if match:
                return pd.Series({
                    'made': int(match.group(1)),
                    'attempted': int(match.group(2)),
                    'pct': float(match.group(3))
                })
            
            return pd.Series({'made': np.nan, 'attempted': np.nan, 'pct': np.nan})
        
        ft_split = df_clean['FTM-FTA (%)'].apply(parse_shooting)
        df_clean['ftm'] = ft_split['made']
        df_clean['fta'] = ft_split['attempted']
        df_clean['ft_pct'] = ft_split['pct']
        df_clean = df_clean.drop('FTM-FTA (%)', axis=1)
    
    # 5. Clean 3PM-3PA (%): "1-7 (14%)" -> threes_made, threes_attempted, three_pct
    if '3PM-3PA (%)' in df_clean.columns:
        def parse_shooting(val):
            if pd.isna(val) or str(val).strip() == '' or str(val).strip() == '-':
                return pd.Series({'made': np.nan, 'attempted': np.nan, 'pct': np.nan})
            
            val_str = str(val).strip()
            match = re.match(r'(\d+)-(\d+)\s*\((\d+(?:\.\d+)?)%\)', val_str)
            if match:
                return pd.Series({
                    'made': int(match.group(1)),
                    'attempted': int(match.group(2)),
                    'pct': float(match.group(3))
                })
            
            return pd.Series({'made': np.nan, 'attempted': np.nan, 'pct': np.nan})
        
        three_split = df_clean['3PM-3PA (%)'].apply(parse_shooting)
        df_clean['threes_made'] = three_split['made']
        df_clean['threes_attempted'] = three_split['attempted']
        df_clean['three_pct'] = three_split['pct']
        df_clean = df_clean.drop('3PM-3PA (%)', axis=1)
    
    # 6. Clean Spread Result: "CHI +13.5" or "DEN -4.5" -> spread_team, spread_value
    if 'Spread Result' in df_clean.columns:
        def parse_spread(val):
            if pd.isna(val) or str(val).strip() == '':
                return pd.Series({'spread_team': None, 'spread_value': np.nan})
            
            val_str = str(val).strip()
            match = re.match(r'([A-Z]+)\s*([\+\-]?\d+(?:\.\d+)?)', val_str)
            if match:
                spread_val = float(match.group(2))
                return pd.Series({
                    'spread_team': match.group(1),
                    'spread_value': spread_val
                })
            
            return pd.Series({'spread_team': None, 'spread_value': np.nan})
        
        spread_split = df_clean['Spread Result'].apply(parse_spread)
        df_clean['spread_team'] = spread_split['spread_team']
        df_clean['spread_value'] = spread_split['spread_value']
        df_clean = df_clean.drop('Spread Result', axis=1)
    
    # 7. Split Matchup column: "@MIN" -> opponent="MIN", home_away="Away"
    if 'Matchup' in df_clean.columns:
        def parse_matchup(val):
            if pd.isna(val) or str(val).strip() == '':
                return pd.Series({'opponent': None, 'home_away': None})
            
            val_str = str(val).strip()
            if val_str.startswith('@'):
                # Away game
                return pd.Series({
                    'opponent': val_str[1:],  # Remove @ symbol
                    'home_away': 'Away'
                })
            else:
                # Home game
                return pd.Series({
                    'opponent': val_str,
                    'home_away': 'Home'
                })
        
        matchup_split = df_clean['Matchup'].apply(parse_matchup)
        df_clean['opponent'] = matchup_split['opponent']
        df_clean['home_away'] = matchup_split['home_away']
        df_clean = df_clean.drop('Matchup', axis=1)
    
    # Convert numeric columns to appropriate types
    numeric_cols = ['Minutes', 'Prop Line', 'PTS', 'REBS', '3s Scored', 'STEALS', 'BLOCKS']
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Rename columns for consistency (lowercase with underscores)
    column_rename = {
        'Date': 'date',
        'Minutes': 'minutes',
        'Prop Line': 'prop_line',
        'PTS': 'points',
        'REBS': 'rebounds',
        '3s Scored': 'threes_scored',
        'STEALS': 'steals',
        'BLOCKS': 'blocks',
        'Season': 'season'
    }
    df_clean = df_clean.rename(columns=column_rename)
    
    # Filter out obviously errant rows
    initial_rows = len(df_clean)
    
    # Remove rows where critical fields are missing
    critical_fields = ['date', 'matchup', 'assists', 'points']
    for field in critical_fields:
        if field in df_clean.columns:
            df_clean = df_clean[df_clean[field].notna() & (df_clean[field] != '')]
    
    # Remove rows where ALL key statistical fields are zero or missing
    stat_fields = ['assists', 'points', 'rebounds', 'minutes']
    stat_fields_present = [f for f in stat_fields if f in df_clean.columns]
    
    if stat_fields_present:
        # Check if all stat fields are either 0 or NaN
        all_zero_or_missing = pd.Series([True] * len(df_clean), index=df_clean.index)
        for field in stat_fields_present:
            # A field is "bad" if it's NaN or 0
            field_bad = (df_clean[field].isna()) | (df_clean[field] == 0)
            all_zero_or_missing = all_zero_or_missing & field_bad
        
        # Remove rows where all stats are zero/missing
        df_clean = df_clean[~all_zero_or_missing]
    
    rows_removed = initial_rows - len(df_clean)
    if rows_removed > 0:
        print(f"  Removed {rows_removed} errant row(s) with missing/zero data")
    
    # Reorder columns for better readability
    priority_cols = [
        'date', 'season', 'opponent', 'home_away', 'win_loss', 'team_score', 'opponent_score',
        'minutes', 'prop_line', 'assists', 'over_under_result',
        'points', 'rebounds', 'threes_scored', 'steals', 'blocks',
        'fgm', 'fga', 'fg_pct', 'ftm', 'fta', 'ft_pct',
        'threes_made', 'threes_attempted', 'three_pct',
        'spread_team', 'spread_value'
    ]
    
    # Keep only columns that exist
    final_cols = [col for col in priority_cols if col in df_clean.columns]
    other_cols = [col for col in df_clean.columns if col not in final_cols]
    df_clean = df_clean[final_cols + other_cols]
    
    return df_clean


def scrape_player_game_log(url, headless=True, scrape_previous_season=True, timeout=60):
    """
    Scrape player game log data from BettingPros for current and previous season
    
    Args:
        url (str): The BettingPros URL for the player prop
        headless (bool): Whether to run browser in headless mode
        scrape_previous_season (bool): Whether to also scrape 2024 season data
        timeout (int): Maximum time in seconds to wait for page loads (default: 60)
        
    Returns:
        dict: Dictionary with 'current_season' and 'previous_season' DataFrames
    """
    # Set up Chrome options
    chrome_options = Options()
    if headless:
        chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36')
    
    # Initialize the driver
    driver = webdriver.Chrome(options=chrome_options)
    driver.set_page_load_timeout(timeout)
    
    try:
        print(f"Navigating to {url}...")
        driver.get(url)
        
        # Wait for the page to load
        time.sleep(3)
        
        # Scrape current season
        print("\n" + "="*80)
        print("SCRAPING CURRENT SEASON (25-26)")
        print("="*80)
        current_season_df = extract_table_data(driver)
        current_season_df['Season'] = '25-26'
        
        # Scrape 2024 season if requested
        previous_season_df = pd.DataFrame()
        if scrape_previous_season:
            try:
                print("\n" + "="*80)
                print("SWITCHING TO 2024 SEASON")
                print("="*80)
                
                # Find the season selector dropdown/button
                # Common selectors for season dropdown
                season_selectors = [
                    "select",
                    ".season-select",
                    ".season-dropdown",
                    "[class*='season']",
                    "button[class*='season']",
                    "div[class*='season']",
                    "button[class*='dropdown']",
                    ".dropdown-toggle"
                ]
                
                season_selector = None
                wait = WebDriverWait(driver, 10)
                
                # Try to find the season selector
                for selector in season_selectors:
                    try:
                        elements = driver.find_elements(By.CSS_SELECTOR, selector)
                        for elem in elements:
                            text = elem.text.lower()
                            if 'season' in text or '2025' in text or 'dropdown' in elem.get_attribute('class').lower():
                                season_selector = elem
                                print(f"Found season selector: {selector} with text: {elem.text}")
                                break
                        if season_selector:
                            break
                    except:
                        continue
                
                # If not found by common selectors, look for any button/select near the table
                if not season_selector:
                    print("Trying to find season selector by looking for buttons/selects...")
                    all_buttons = driver.find_elements(By.TAG_NAME, "button")
                    all_selects = driver.find_elements(By.TAG_NAME, "select")
                    
                    for elem in all_buttons + all_selects:
                        try:
                            elem_text = elem.text.lower()
                            if 'season' in elem_text or '2025' in elem_text:
                                season_selector = elem
                                print(f"Found season selector button/select with text: {elem.text}")
                                break
                        except:
                            continue
                
                if season_selector:
                    # Click the dropdown
                    print("Clicking season selector...")
                    driver.execute_script("arguments[0].scrollIntoView(true);", season_selector)
                    time.sleep(1)
                    
                    # Try clicking with JavaScript if regular click fails
                    try:
                        season_selector.click()
                    except:
                        driver.execute_script("arguments[0].click();", season_selector)
                    
                    time.sleep(2)
                    
                    # Take a screenshot to see the dropdown state
                    try:
                        driver.save_screenshot("dropdown_opened.png")
                        print("Screenshot saved: dropdown_opened.png")
                    except:
                        pass
                    
                    # Find and click the 2024 Season option
                    print("Looking for 2024 Season option...")
                    
                    # Try different ways to find the 2024 option
                    option_found = False
                    
                    # Method 1: Look for select option
                    try:
                        select_elem = driver.find_element(By.TAG_NAME, "select")
                        options = select_elem.find_elements(By.TAG_NAME, "option")
                        for option in options:
                            if "2024" in option.text:
                                print(f"Found option: {option.text}")
                                option.click()
                                option_found = True
                                break
                    except:
                        pass
                    
                    # Method 2: Look for ANY newly visible clickable elements with 2024
                    if not option_found:
                        time.sleep(1)  # Give dropdown time to fully render
                        all_elements = driver.find_elements(By.XPATH, "//*[contains(text(), '2024')]")
                        print(f"Found {len(all_elements)} elements containing '2024'")
                        
                        for elem in all_elements:
                            try:
                                elem_text = elem.text.strip()
                                print(f"Checking element: '{elem_text}', displayed: {elem.is_displayed()}, enabled: {elem.is_enabled()}")
                                
                                if elem.is_displayed() and "2024" in elem_text:
                                    print(f"Attempting to click: {elem_text}")
                                    try:
                                        elem.click()
                                    except:
                                        # Try JavaScript click
                                        driver.execute_script("arguments[0].click();", elem)
                                    option_found = True
                                    break
                            except Exception as e:
                                print(f"Error checking element: {e}")
                                continue
                    
                    # Method 3: Look for li, a, or div elements in dropdown menus
                    if not option_found:
                        print("Trying to find dropdown menu items (li, a, div)...")
                        menu_selectors = [
                            "li", "a", "div[role='menuitem']", 
                            "div[role='option']", ".dropdown-item",
                            "[class*='menu-item']", "[class*='dropdown']"
                        ]
                        
                        for selector in menu_selectors:
                            try:
                                menu_items = driver.find_elements(By.CSS_SELECTOR, selector)
                                for item in menu_items:
                                    if item.is_displayed() and "2024" in item.text:
                                        print(f"Found menu item with 2024: {item.text}")
                                        try:
                                            item.click()
                                        except:
                                            driver.execute_script("arguments[0].click();", item)
                                        option_found = True
                                        break
                                if option_found:
                                    break
                            except:
                                continue
                    
                    if option_found:
                        print("Selected 2024 Season, waiting for table to reload...")
                        time.sleep(4)
                        
                        # Scroll to ensure the game log table is in view
                        try:
                            game_log_heading = driver.find_element(By.XPATH, "//*[contains(text(), 'Player Game Log')]")
                            driver.execute_script("arguments[0].scrollIntoView(true);", game_log_heading)
                            time.sleep(1)
                        except:
                            pass
                        
                        # Wait for the table to be stale (indicating it's refreshing)
                        # Then wait for the new table to load
                        try:
                            wait = WebDriverWait(driver, 10)
                            # Wait for table to be present and stable
                            wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
                            time.sleep(2)
                        except:
                            pass
                        
                        # Take screenshot of the 2024 season view
                        try:
                            driver.save_screenshot("2024_season_loaded.png")
                            print("Screenshot saved: 2024_season_loaded.png")
                        except:
                            pass
                        
                        # Scrape the 2024 season table
                        print("\n" + "="*80)
                        print("SCRAPING 2024 SEASON (24-25)")
                        print("="*80)
                        previous_season_df = extract_table_data(driver)
                        
                        # Validate we got the right table (should have Date, Matchup columns)
                        if 'Date' in previous_season_df.columns or 'Matchup' in previous_season_df.columns:
                            previous_season_df['Season'] = '24-25'
                            print("Successfully scraped 2024 season game log")
                        else:
                            print(f"Warning: 2024 season table has unexpected structure")
                            print(f"Columns: {list(previous_season_df.columns)}")
                            print("This might not be the game log table")
                            previous_season_df['Season'] = '24-25'
                    else:
                        print("Could not find 2024 Season option")
                        print("Check dropdown_opened.png to see the dropdown state")
                else:
                    print("Could not find season selector dropdown")
                    
            except Exception as e:
                print(f"Error scraping 2024 season: {e}")
                import traceback
                traceback.print_exc()
        
        return {
            'current_season': current_season_df,
            'previous_season': previous_season_df
        }
        
    except Exception as e:
        print(f"Error during scraping: {e}")
        # Take a screenshot for debugging
        try:
            driver.save_screenshot("error_screenshot.png")
            print("Error screenshot saved as error_screenshot.png")
        except:
            pass
        raise
        
    finally:
        driver.quit()
        print("Browser closed")


def player_name_to_url_slug(player_name):
    """
    Convert player name to URL-friendly slug
    
    Args:
        player_name (str): Player name like "Nikola Jokic" or "LeBron James"
        
    Returns:
        str: URL slug like "nikola-jokic" or "lebron-james"
    """
    return player_name.lower().replace(' ', '-').replace('.', '')


def scrape_multiple_players(player_names, stat='assists', headless=False, scrape_previous_season=True, player_timeout=90):
    """
    Scrape multiple players' prop data and combine into one dataset
    
    Args:
        player_names (list): List of player names (e.g., ["Nikola Jokic", "LeBron James"])
        stat (str): The stat to scrape (default: 'assists')
        headless (bool): Whether to run browser in headless mode
        scrape_previous_season (bool): Whether to also scrape previous season
        player_timeout (int): Maximum seconds to spend on each player (default: 90)
        
    Returns:
        pandas.DataFrame: Combined data for all players
    """
    all_players_data = []
    failed_players = []
    
    for player_name in player_names:
        print("\n" + "="*80)
        print(f"STARTING SCRAPE FOR: {player_name.upper()}")
        print("="*80)
        
        # Convert player name to URL slug
        player_slug = player_name_to_url_slug(player_name)
        url = f"https://www.bettingpros.com/nba/props/{player_slug}/{stat}/"
        
        print(f"URL: {url}")
        
        try:
            # Scrape this player's data with timeout
            import signal
            
            # Create a wrapper that will timeout
            results = scrape_player_game_log(url, headless=headless, scrape_previous_season=scrape_previous_season, timeout=player_timeout)
            current_df = results['current_season']
            previous_df = results['previous_season']
            
            # Clean the data
            print(f"\nCleaning data for {player_name}...")
            current_df_clean = clean_game_log_data(current_df)
            print(f"  ✓ Cleaned current season: {len(current_df_clean)} games")
            
            if not previous_df.empty:
                previous_df_clean = clean_game_log_data(previous_df)
                print(f"  ✓ Cleaned previous season: {len(previous_df_clean)} games")
            else:
                previous_df_clean = pd.DataFrame()
            
            # Combine seasons for this player
            if not previous_df_clean.empty:
                player_combined = pd.concat([current_df_clean, previous_df_clean], ignore_index=True)
            else:
                player_combined = current_df_clean.copy()
            
            # Add player name column
            player_combined['player_name'] = player_name
            
            print(f"  ✓ Total games for {player_name}: {len(player_combined)}")
            
            all_players_data.append(player_combined)
            
        except TimeoutException as e:
            print(f"  ✗ TIMEOUT scraping {player_name}: Taking too long, skipping...")
            failed_players.append((player_name, "Timeout"))
            continue
        except Exception as e:
            print(f"  ✗ ERROR scraping {player_name}: {e}")
            failed_players.append((player_name, str(e)))
            import traceback
            traceback.print_exc()
            continue
    
    # Show summary of failed players if any
    if failed_players:
        print("\n" + "="*80)
        print(f"FAILED TO SCRAPE {len(failed_players)} PLAYER(S):")
        print("="*80)
        for player, reason in failed_players:
            print(f"  ✗ {player}: {reason[:100]}")
    
    # Combine all players' data
    if all_players_data:
        combined_all = pd.concat(all_players_data, ignore_index=True)
        
        # Reorder columns to put player_name first
        cols = combined_all.columns.tolist()
        if 'player_name' in cols:
            cols.remove('player_name')
            cols = ['player_name'] + cols
            combined_all = combined_all[cols]
        
        return combined_all
    else:
        return pd.DataFrame()


def scrape_nikola_jokic_assists():
    """
    Scrape Nikola Jokic assists game log for current and 2024 season
    """
    url = "https://www.bettingpros.com/nba/props/nikola-jokic/assists/"
    return scrape_player_game_log(url, headless=False, scrape_previous_season=True)


if __name__ == "__main__":
    # ============================================================================
    # CONFIGURATION: Add your players here
    # ============================================================================
    
    # Denver Nuggets roster (players from 24-25 and 25-26 seasons)
    # MISSING PLAYERS ONLY - completing the roster
    PLAYERS = [
        "Hunter Tyson"
    ]
    
    # Stat to scrape (assists, points, rebounds, etc.)
    STAT = "assists"
    
    # Whether to run browser in headless mode (True = no browser window)
    HEADLESS = False
    
    # Whether to scrape previous season data
    SCRAPE_PREVIOUS_SEASON = True
    
    # Output file name
    OUTPUT_FILE = f"denver_nuggets_{STAT}_complete.csv"
    
    # ============================================================================
    # EXECUTION
    # ============================================================================
    
    print("="*80)
    print(f"NBA PROPS SCRAPER - {STAT.upper()}")
    print("="*80)
    print(f"Players to scrape: {', '.join(PLAYERS)}")
    print(f"Stat: {STAT}")
    print(f"Scrape previous season: {SCRAPE_PREVIOUS_SEASON}")
    print("="*80)
    
    try:
        # Check if existing data exists
        existing_df = pd.DataFrame()
        if os.path.exists(OUTPUT_FILE):
            print(f"\n📁 Found existing data in {OUTPUT_FILE}")
            existing_df = pd.read_csv(OUTPUT_FILE)
            print(f"   Existing: {len(existing_df)} games, {existing_df['player_name'].nunique()} players")
        
        # Scrape all players
        new_df = scrape_multiple_players(
            player_names=PLAYERS,
            stat=STAT,
            headless=HEADLESS,
            scrape_previous_season=SCRAPE_PREVIOUS_SEASON,
            player_timeout=90
        )
        
        if new_df.empty:
            print("\n✗ No new data scraped. Please check for errors above.")
        else:
            # Combine with existing data if present
            if not existing_df.empty:
                print(f"\n🔗 Combining new data with existing data...")
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                print(f"   Added: {len(new_df)} games from {new_df['player_name'].nunique()} players")
            else:
                combined_df = new_df
            
            # Display summary
            print("\n" + "="*80)
            print("FINAL COMBINED DATA SUMMARY:")
            print("="*80)
            print(f"Total games scraped: {len(combined_df)}")
            print(f"Total columns: {len(combined_df.columns)}")
            print(f"\nGames per player:")
            print(combined_df['player_name'].value_counts().to_string())
            print(f"\nGames per season:")
            print(combined_df['season'].value_counts().to_string())
            
            # Save to CSV
            combined_df.to_csv(OUTPUT_FILE, index=False)
            print(f"\n✓ All data saved to {OUTPUT_FILE}")
            
            # Show sample
            print("\n" + "="*80)
            print("SAMPLE OF FINAL DATA:")
            print("="*80)
            display_cols = ['player_name', 'date', 'season', 'opponent', 'home_away', 
                          'assists', 'over_under_result', 'points', 'rebounds']
            display_cols = [col for col in display_cols if col in combined_df.columns]
            print(combined_df[display_cols].head(15).to_string())
            
    except Exception as e:
        print(f"\n✗ Failed to scrape data: {e}")
        import traceback
        traceback.print_exc()

