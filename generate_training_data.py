import pandas as pd
import os

# Import required functions from other modules
from transform_player_stats import calculate_player_stats, get_distinct_players
from generate_team_stats import generate_team_stats, get_available_teams
from pca import get_player_pca_score
from get_team_city import get_team_city
from generate_lineup import get_lineup_by_game_id, load_all_player_boxscores

# Configuration
SEASON_YEAR = "2023-2024"  # Default season year

def load_play_by_play_data(season_year):
    """
    Load play-by-play data for the specified season year.
    
    Args:
        season_year (str): Season year in format "YYYY-YYYY" (e.g., "2023-2024")
    
    Returns:
        pd.DataFrame: Loaded play-by-play data
    """
    # Map season year to file path
    file_mapping = {
        "2022-2023": r"C:\Users\micha\nba_predictions\data\play_by_play\historical\[10-18-2022]-[06-12-2023]-combined-stats.csv",
        "2023-2024": r"C:\Users\micha\nba_predictions\data\play_by_play\historical\[10-24-2023]-[06-17-2024]-combined-stats.csv"
    }
    
    if season_year not in file_mapping:
        raise ValueError(f"Season year {season_year} not supported. Available options: {list(file_mapping.keys())}")
    
    file_path = file_mapping[season_year]
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Play-by-play data file not found: {file_path}")
    
    print(f"Loading play-by-play data for season {season_year}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows of play-by-play data")
    
    return df

def set_season_year(season_year):
    """
    Set the season year for data loading.
    
    Args:
        season_year (str): Season year in format "YYYY-YYYY" (e.g., "2023-2024")
    """
    global SEASON_YEAR
    SEASON_YEAR = season_year
    print(f"Season year set to: {SEASON_YEAR}")

def get_current_season_year():
    """Get the currently configured season year."""
    return SEASON_YEAR

def load_training_data(season_year=None):
    """
    Load all necessary data for training data generation.
    
    Args:
        season_year (str, optional): Season year to load. If None, uses current SEASON_YEAR
    
    Returns:
        pd.DataFrame: Play-by-play data for the specified season
    """
    if season_year is None:
        season_year = SEASON_YEAR
    
    # Load the play-by-play data
    play_by_play_df = load_play_by_play_data(season_year)
    
    return play_by_play_df

# Example usage and testing functions
def test_data_loading():
    """Test function to verify all imports and data loading work correctly."""
    print("Testing data loading functionality...")
    
    # Test loading play-by-play data for both seasons
    for season in ["2022-2023", "2023-2024"]:
        try:
            print(f"\n--- Testing season {season} ---")
            df = load_play_by_play_data(season)
            print(f"Successfully loaded {len(df)} rows for {season}")
            print(f"Columns: {list(df.columns)}")
        except Exception as e:
            print(f"Error loading {season}: {e}")
    
    # Test imported functions
    print("\n--- Testing imported functions ---")
    try:
        players = get_distinct_players()
        print(f"Found {len(players)} distinct players")
        
        teams = get_available_teams()
        print(f"Found {len(teams)} available teams")
        
        print("All imports and basic functionality working correctly!")
        
    except Exception as e:
        print(f"Error testing imported functions: {e}")

def calculate_game_time_remaining(period, remaining_time):
    """
    Calculate total game time remaining based on period and remaining time in current period.
    
    Args:
        period (int): Current period (1-4)
        remaining_time (str): Time remaining in current period (format: "0:MM:SS")
        
    Returns:
        str: Total game time remaining (format: "MM:SS")
    """
    try:
        # Parse remaining_time string (format: "0:MM:SS")
        if pd.isna(remaining_time) or not remaining_time:
            return "00:00"
            
        time_parts = str(remaining_time).split(':')
        if len(time_parts) >= 2:
            # Get minutes and seconds from the current period
            minutes = int(time_parts[-2])  # Second to last part is minutes
            seconds = int(time_parts[-1])   # Last part is seconds
        else:
            return "00:00"
        
        # Calculate total minutes remaining based on period
        # NBA: 4 periods of 12 minutes each = 48 minutes total
        if period == 1:
            total_minutes_remaining = 36 + minutes  # 3 full periods + current period remaining
        elif period == 2:
            total_minutes_remaining = 24 + minutes  # 2 full periods + current period remaining
        elif period == 3:
            total_minutes_remaining = 12 + minutes  # 1 full period + current period remaining
        elif period == 4:
            total_minutes_remaining = minutes       # Only current period remaining
        else:
            # Handle overtime or invalid periods
            total_minutes_remaining = minutes
            
        return f"{total_minutes_remaining:02d}:{seconds:02d}"
        
    except (ValueError, IndexError, TypeError):
        return "00:00"

def create_team_abbreviation_mapping():
    """
    Create mapping from 3-letter team abbreviations to full team names.
    
    Returns:
        dict: Mapping from abbreviation to full team name
    """
    return {
        'ATL': 'Atlanta Hawks',
        'BKN': 'Brooklyn Nets', 
        'BOS': 'Boston Celtics',
        'CHA': 'Charlotte Hornets',
        'CHI': 'Chicago Bulls',
        'CLE': 'Cleveland Cavaliers',
        'DAL': 'Dallas Mavericks',
        'DEN': 'Denver Nuggets',
        'DET': 'Detroit Pistons',
        'GSW': 'Golden State Warriors',
        'HOU': 'Houston Rockets',
        'IND': 'Indiana Pacers',
        'LAC': 'Los Angeles Clippers',
        'LAL': 'Los Angeles Lakers',
        'MEM': 'Memphis Grizzlies',
        'MIA': 'Miami Heat',
        'MIL': 'Milwaukee Bucks',
        'MIN': 'Minnesota Timberwolves',
        'NOP': 'New Orleans Pelicans',
        'NYK': 'New York Knicks',
        'OKC': 'Oklahoma City Thunder',
        'ORL': 'Orlando Magic',
        'PHI': 'Philadelphia 76ers',
        'PHX': 'Phoenix Suns',
        'POR': 'Portland Trail Blazers',
        'SAC': 'Sacramento Kings',
        'SAS': 'San Antonio Spurs',
        'TOR': 'Toronto Raptors',
        'UTA': 'Utah Jazz',
        'WAS': 'Washington Wizards'
    }

def get_team_stats_for_game(game_df, team_mapping, target_date=None):
    """
    Get team stats for both teams in a game.
    Falls back to season averages if date-filtered stats aren't available.
    
    Args:
        game_df (pd.DataFrame): DataFrame for a single game
        team_mapping (dict): Mapping of game_id to home/away teams
        target_date (str, optional): Date for stats calculation
        
    Returns:
        dict: Team stats for home and away teams
    """
    abbrev_mapping = create_team_abbreviation_mapping()
    game_id = game_df.iloc[0]['game_id']
    
    # Get team abbreviations from mapping
    home_abbrev = team_mapping.get(game_id, {}).get('home_team', 'Unknown')
    away_abbrev = team_mapping.get(game_id, {}).get('away_team', 'Unknown')
    
    # Convert to full team names
    home_team_full = abbrev_mapping.get(home_abbrev, home_abbrev)
    away_team_full = abbrev_mapping.get(away_abbrev, away_abbrev)
    
    # Get team stats - try with date first, then fall back to season averages
    home_stats = generate_team_stats(home_team_full, target_date)
    if not home_stats:  # If no stats with date filter, use season averages
        home_stats = generate_team_stats(home_team_full, None) or {}
    
    away_stats = generate_team_stats(away_team_full, target_date)
    if not away_stats:  # If no stats with date filter, use season averages
        away_stats = generate_team_stats(away_team_full, None) or {}
    
    return {
        'home_team_stats': home_stats,
        'away_team_stats': away_stats,
        'home_abbrev': home_abbrev,
        'away_abbrev': away_abbrev
    }

def determine_home_away_teams(df):
    """
    Determine which team is home and which is away for each game by tracking score increments.
    
    Args:
        df (pd.DataFrame): Play-by-play DataFrame with game_id, team, away_score, home_score columns
        
    Returns:
        dict: Dictionary mapping game_id to {'home_team': team_name, 'away_team': team_name}
    """
    game_team_mapping = {}
    
    for game_id in df['game_id'].unique():
        game_df = df[df['game_id'] == game_id].sort_values('play_id').reset_index(drop=True)
        
        home_team = None
        away_team = None
        prev_away_score = 0
        prev_home_score = 0
        
        for i, row in game_df.iterrows():
            current_away_score = row.get('away_score', 0) or 0
            current_home_score = row.get('home_score', 0) or 0
            team = row.get('team', '')
            
            # Check if away score incremented and we haven't identified away team yet
            if current_away_score > prev_away_score and away_team is None:
                away_team = team
                
            # Check if home score incremented and we haven't identified home team yet  
            if current_home_score > prev_home_score and home_team is None:
                home_team = team
                
            # Update previous scores
            prev_away_score = current_away_score
            prev_home_score = current_home_score
            
            # Break early if we've identified both teams
            if home_team and away_team:
                break
        
        # Store the mapping for this game
        game_team_mapping[game_id] = {
            'home_team': home_team or 'Unknown',
            'away_team': away_team or 'Unknown'
        }
    
    return game_team_mapping

def create_llm_training_data(df, n_total=5, filter_nan=True):
    """
    Create LLM training data by concatenating descriptions with team stats, lineups, time, and score context.
    Only concatenates within the same game (respects game_id boundaries).
    Format: "AWAY_TEAM (OEFF: x DEFF: y PACE: z REST_DAYS: w) HOME_TEAM (...) AWAY_LINEUP: [...] HOME_LINEUP: [...] || XX:XX away_team: score home_team: score | description --> ..."
    
    Args:
        df (pd.DataFrame): Play-by-play DataFrame with required columns
        n_total (int): Total number of descriptions to concatenate together (default: 5)
        filter_nan (bool): Whether to filter out rows with NaN descriptions (default: True)
    
    Returns:
        pd.DataFrame: DataFrame with new 'concatenated_description' column
    """
    # Work with a copy to avoid modifying original DataFrame
    result_df = df.copy()
    
    # Filter out NaN descriptions if requested
    if filter_nan:
        result_df = result_df.dropna(subset=['description']).reset_index(drop=True)
    
    # Determine home/away team mapping for all games
    print("Determining home/away team mappings...")
    game_team_mapping = determine_home_away_teams(result_df)
    
    # Get team stats and lineups for each game (cache to avoid repeated calls)
    print("Loading team stats for games...")
    print("Loading player boxscore data for lineups...")
    try:
        boxscore_data = load_all_player_boxscores()
        print(f"Loaded boxscore data with {len(boxscore_data)} player records")
    except Exception as e:
        print(f"Warning: Could not load boxscore data for lineups: {e}")
        boxscore_data = None
    
    game_team_stats = {}
    unique_games = result_df['game_id'].unique()
    
    for game_id in unique_games:
        game_df = result_df[result_df['game_id'] == game_id]
        # Use the game date for team stats context
        game_date = game_df.iloc[0].get('date', None)
        stats = get_team_stats_for_game(game_df, game_team_mapping, target_date=game_date)
        
        # Get lineups for this game
        lineups = {}
        if boxscore_data is not None:
            try:
                lineups = get_lineup_by_game_id(game_id, boxscore_data)
            except Exception as e:
                print(f"Warning: Could not get lineups for game {game_id}: {e}")
                lineups = {}
        
        # Combine stats and lineups
        stats['lineups'] = lineups
        game_team_stats[game_id] = stats
    
    concatenated_descriptions = []
    
    for i in range(len(result_df)):
        current_game_id = result_df.iloc[i]['game_id']
        current_desc = result_df.iloc[i]['description']
        
        # Get team stats and lineups for current game
        team_stats = game_team_stats.get(current_game_id, {})
        away_stats = team_stats.get('away_team_stats', {})
        home_stats = team_stats.get('home_team_stats', {})
        away_abbrev = team_stats.get('away_abbrev', 'Unknown')
        home_abbrev = team_stats.get('home_abbrev', 'Unknown')
        lineups = team_stats.get('lineups', {})
        
        # Collect descriptions with enhanced context from the same game only
        descriptions_to_concat = []
        
        # Go backwards from current position to collect descriptions
        collected_count = 0
        
        for j in range(i, -1, -1):  # Start from current row and go backwards
            row_game_id = result_df.iloc[j]['game_id']
            row_desc = result_df.iloc[j]['description']
            row_away_score = result_df.iloc[j].get('away_score', 0) or 0
            row_home_score = result_df.iloc[j].get('home_score', 0) or 0
            
            # Stop if we've moved to a different game
            if row_game_id != current_game_id:
                break
            
            # Add valid descriptions with score and time context
            if pd.notna(row_desc):
                # Calculate game time remaining
                row_period = result_df.iloc[j].get('period', 4)
                row_remaining_time = result_df.iloc[j].get('remaining_time', '0:00:00')
                game_time_remaining = calculate_game_time_remaining(row_period, row_remaining_time)
                
                # Format: "XX:XX away_team: score home_team: score | description"
                time_score_context = f"{game_time_remaining} {away_abbrev}: {row_away_score} {home_abbrev}: {row_home_score} | {str(row_desc)}"
                descriptions_to_concat.insert(0, time_score_context)  # Insert at beginning to maintain order
                collected_count += 1
                
                # Stop if we've collected the desired total number of descriptions
                if collected_count >= n_total:
                    break
        
        # Create team stats header with lineups (once per concatenated description)
        # Handle None REST_DAYS by converting to 0
        away_rest_days = away_stats.get('REST_DAYS') if away_stats.get('REST_DAYS') is not None else 0
        home_rest_days = home_stats.get('REST_DAYS') if home_stats.get('REST_DAYS') is not None else 0
        
        away_stats_str = f"OEFF: {away_stats.get('OEFF', 'N/A')} DEFF: {away_stats.get('DEFF', 'N/A')} PACE: {away_stats.get('PACE', 'N/A')} REST_DAYS: {away_rest_days}"
        home_stats_str = f"OEFF: {home_stats.get('OEFF', 'N/A')} DEFF: {home_stats.get('DEFF', 'N/A')} PACE: {home_stats.get('PACE', 'N/A')} REST_DAYS: {home_rest_days}"
        
        # Format lineups
        away_lineup = "N/A"
        home_lineup = "N/A"
        
        # The lineups dict has team names as keys, we need to match them to away/home
        # We need to convert abbreviations to full names to match lineup keys
        abbrev_mapping = create_team_abbreviation_mapping()
        away_full_name = abbrev_mapping.get(away_abbrev, away_abbrev)
        home_full_name = abbrev_mapping.get(home_abbrev, home_abbrev)
        
        # Try to find matching lineup by checking if team name contains the full name or vice versa
        for team_name, players in lineups.items():
            # Check for away team match
            if (away_full_name in team_name or team_name in away_full_name or 
                any(part in team_name for part in away_full_name.split())):
                away_lineup = ", ".join(players[:5])  # Limit to first 5 players for brevity
                if len(players) > 5:
                    away_lineup += f" (+{len(players)-5} more)"
            
            # Check for home team match  
            if (home_full_name in team_name or team_name in home_full_name or
                any(part in team_name for part in home_full_name.split())):
                home_lineup = ", ".join(players[:5])  # Limit to first 5 players for brevity
                if len(players) > 5:
                    home_lineup += f" (+{len(players)-5} more)"
        
        team_stats_header = f"{away_abbrev} ({away_stats_str}) {home_abbrev} ({home_stats_str}) AWAY_LINEUP: [{away_lineup}] HOME_LINEUP: [{home_lineup}] || "
        
        # Join descriptions and prepend team stats header
        descriptions_part = " --> ".join(descriptions_to_concat)
        concatenated_desc = team_stats_header + descriptions_part
        concatenated_descriptions.append(concatenated_desc)
    
    # Add the concatenated descriptions as a new column
    result_df['concatenated_description'] = concatenated_descriptions
    
    return result_df

def generate_llm_dataset(season_year=None, n_total=5, sample_size=None, game_id_filter=None):
    """
    Generate a complete LLM training dataset from play-by-play data.
    
    Args:
        season_year (str, optional): Season year to use. If None, uses current SEASON_YEAR
        n_total (int): Total number of descriptions to concatenate together per row
        sample_size (int, optional): If provided, randomly sample this many rows
        game_id_filter (list, optional): If provided, only include these game IDs
    
    Returns:
        pd.DataFrame: DataFrame ready for LLM training with concatenated descriptions
    """
    if season_year is None:
        season_year = SEASON_YEAR
    
    print(f"Generating LLM dataset for season {season_year}...")
    
    # Load the play-by-play data
    df = load_play_by_play_data(season_year)
    
    # Filter by game IDs if specified
    if game_id_filter:
        df = df[df['game_id'].isin(game_id_filter)]
        print(f"Filtered to {len(df)} rows from {len(game_id_filter)} games")
    
    # Sort by game_id and play sequence to ensure proper chronological order
    df = df.sort_values(['game_id', 'play_id']).reset_index(drop=True)
    
    # Create concatenated descriptions
    result_df = create_llm_training_data(df, n_total=n_total)
    
    # Sample if requested
    if sample_size and sample_size < len(result_df):
        result_df = result_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Sampled {sample_size} rows from dataset")
    
    print(f"Generated dataset with {len(result_df)} rows")
    print(f"Each row contains up to {n_total} descriptions concatenated together")
    
    return result_df

def save_llm_dataset(df, filename=None, season_year=None):
    """
    Save the LLM training dataset to a CSV file.
    
    Args:
        df (pd.DataFrame): Dataset to save
        filename (str, optional): Custom filename. If None, auto-generates based on season
        season_year (str, optional): Season year for auto-generated filename
    """
    if filename is None:
        if season_year is None:
            season_year = SEASON_YEAR
        filename = f"llm_training_data_{season_year.replace('-', '_')}.csv"
    
    # Create output directory if it doesn't exist
    output_dir = "data/training"
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved LLM training dataset to: {filepath}")
    return filepath

def preview_llm_data(df, n_samples=3):
    """
    Preview sample concatenated descriptions from the LLM dataset.
    
    Args:
        df (pd.DataFrame): LLM dataset with concatenated_description column
        n_samples (int): Number of samples to show
    """
    print(f"\n=== Preview of LLM Training Data (showing {n_samples} samples) ===\n")
    
    # Show random samples
    sample_df = df.sample(n=min(n_samples, len(df)), random_state=42)
    
    for i, (idx, row) in enumerate(sample_df.iterrows(), 1):
        print(f"Sample {i}:")
        print(f"Game ID: {row['game_id']}")
        print(f"Date: {row['date']}")
        print(f"Original Description: {row['description']}")
        print(f"Concatenated Description: {row['concatenated_description']}")
        print("-" * 80)

if __name__ == "__main__":
    # Run tests when script is executed directly
    test_data_loading()
    
    # Example of generating LLM training data
    print("\n" + "="*50)
    print("TESTING LLM TRAINING DATA GENERATION")
    print("="*50)
    
    try:
        # Load only first 1500 rows for faster testing
        print("Loading first 1500 rows for testing...")
        df_test = load_play_by_play_data("2023-2024")
        df_test = df_test.head(1500)  # Limit to first 1500 rows
        df_test = df_test.sort_values(['game_id', 'play_id']).reset_index(drop=True)
        
        # Generate training data from the limited dataset
        test_df = create_llm_training_data(df_test, n_total=3, filter_nan=True)
        print(f"Generated test dataset with {len(test_df)} rows from first 1500 rows")
        
        # Preview the results
        preview_llm_data(test_df, n_samples=2)
        
        # Save the test data to CSV
        saved_path = save_llm_dataset(test_df, filename="test_llm_data_sample.csv")
        print(f"\nTest data successfully saved!")
        
    except Exception as e:
        print(f"Error testing LLM data generation: {e}")
