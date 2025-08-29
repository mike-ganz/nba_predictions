import pandas as pd
import os
from typing import Optional, Union, List, Dict

def load_all_player_boxscores() -> pd.DataFrame:
    """
    Load and combine all player boxscore Excel files from the historical directory.
    
    Returns:
        pd.DataFrame: Combined dataframe containing all player boxscore data
    """
    # Directory containing the historical player boxscore files
    data_dir = "data/player_boxscores/historical"
    
    # Get all xlsx files in the directory
    xlsx_files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
    
    if not xlsx_files:
        raise FileNotFoundError(f"No Excel files found in {data_dir}")
    
    # List to store individual dataframes
    dataframes = []
    
    # Load each Excel file and add to the list
    for file in xlsx_files:
        file_path = os.path.join(data_dir, file)
        print(f"Loading {file}...")
        
        try:
            df = pd.read_excel(file_path)
            dataframes.append(df)
            print(f"Successfully loaded {file} with {len(df)} rows")
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
    
    if not dataframes:
        raise Exception("No valid Excel files were loaded")
    
    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Combined dataset has {len(combined_df)} total rows")
    
    return combined_df

def get_lineup_by_game_id(game_id: Union[str, int], df: Optional[pd.DataFrame] = None) -> Dict[str, List[str]]:
    """
    Get player boxscore data for a specific game_id organized by team.
    
    Args:
        game_id (Union[str, int]): The game ID to filter by
        df (Optional[pd.DataFrame]): Pre-loaded dataframe. If None, will load all data.
        
    Returns:
        Dict[str, List[str]]: Dictionary with team names as keys and lists of player names as values
                             Example: {"Lakers": ["LeBron James", "Anthony Davis"], "Celtics": ["Jayson Tatum"]}
    """
    # Load data if not provided
    if df is None:
        df = load_all_player_boxscores()
    
    # Ensure game_id column exists
    if 'game_id' not in df.columns:
        # Check for alternative column names
        possible_columns = ['GAME-ID', 'Game_ID', 'GameID', 'GAME_ID', 'ID']
        game_id_column = None
        
        for col in possible_columns:
            if col in df.columns:
                game_id_column = col
                break
        
        if game_id_column is None:
            available_columns = list(df.columns)
            raise ValueError(f"No game_id column found. Available columns: {available_columns}")
        
        print(f"Using column '{game_id_column}' as game_id")
    else:
        game_id_column = 'game_id'
    
    # Filter by game_id
    filtered_df = df[df[game_id_column] == game_id]
    
    if len(filtered_df) == 0:
        print(f"No data found for game_id: {game_id}")
        return {}
    
    print(f"Found {len(filtered_df)} player records for game_id: {game_id}")
    
    # Find team and player name columns
    team_column = None
    player_column = None
    
    # Check for team column variations
    team_columns = ['OWN \nTEAM', 'OWN TEAM', 'TEAM', 'Team', 'team']
    for col in team_columns:
        if col in filtered_df.columns:
            team_column = col
            break
    
    # Check for player name column variations  
    player_columns = ['PLAYER \nFULL NAME', 'PLAYER FULL NAME', 'PLAYER_NAME', 'Player', 'Name']
    for col in player_columns:
        if col in filtered_df.columns:
            player_column = col
            break
    
    if team_column is None:
        raise ValueError(f"No team column found. Available columns: {list(filtered_df.columns)}")
    
    if player_column is None:
        raise ValueError(f"No player name column found. Available columns: {list(filtered_df.columns)}")
    
    print(f"Using '{team_column}' as team column and '{player_column}' as player column")
    
    # Group players by team
    teams_players = {}
    for _, row in filtered_df.iterrows():
        team = row[team_column]
        player = row[player_column]
        
        if team not in teams_players:
            teams_players[team] = []
        
        teams_players[team].append(player)
    
    # Print summary
    for team, players in teams_players.items():
        print(f"{team}: {len(players)} players")
    
    return teams_players

def get_available_game_ids(df: Optional[pd.DataFrame] = None, limit: int = 10) -> List:
    """
    Get a sample of available game IDs from the dataset.
    
    Args:
        df (Optional[pd.DataFrame]): Pre-loaded dataframe. If None, will load all data.
        limit (int): Number of game IDs to return (default: 10)
        
    Returns:
        List: List of available game IDs
    """
    # Load data if not provided
    if df is None:
        df = load_all_player_boxscores()
    
    # Find the game_id column
    game_id_column = 'game_id'
    if 'game_id' not in df.columns:
        possible_columns = ['GAME-ID', 'Game_ID', 'GameID', 'GAME_ID', 'ID']
        for col in possible_columns:
            if col in df.columns:
                game_id_column = col
                break
    
    # Get unique game IDs
    unique_game_ids = df[game_id_column].unique()
    
    print(f"Total unique game IDs available: {len(unique_game_ids)}")
    return unique_game_ids[:limit].tolist()

# Example usage
if __name__ == "__main__":
    # Load all player boxscore data
    print("Loading all player boxscore data...")
    all_data = load_all_player_boxscores()
    
    # Show basic info about the combined dataset
    print(f"\nDataset Info:")
    print(f"Shape: {all_data.shape}")
    print(f"Columns: {list(all_data.columns)}")
    
    # Get a sample of available game IDs
    print(f"\nSample of available game IDs:")
    sample_game_ids = get_available_game_ids(all_data)
    for game_id in sample_game_ids:
        print(f"  {game_id}")
    
    # Example: Get data for the first available game ID
    if sample_game_ids:
        first_game_id = sample_game_ids[0]
        print(f"\nExample: Getting data for game_id {first_game_id}")
        game_data = get_lineup_by_game_id(first_game_id, all_data)
        
        if game_data:
            print(f"\nTeams and players in this game:")
            for team, players in game_data.items():
                print(f"\n{team}:")
                for player in players[:5]:  # Show first 5 players per team
                    print(f"  - {player}")
                if len(players) > 5:
                    print(f"  ... and {len(players) - 5} more players")
        else:
            print("No data found for this game ID")
