"""
Load and transform player statistics from historical data.
"""
from pathlib import Path
from typing import Optional

import pandas as pd


def load_player_data(season: str) -> Optional[pd.DataFrame]:
    """
    Load player boxscore data for a given season.
    
    Args:
        season: Season identifier (e.g., "2021-2022", "2025-2026")
        
    Returns:
        DataFrame with player boxscore data, or None if not found
    """
    # Map season to file paths (try multiple naming patterns)
    season_map = {
        '2020-2021': [
            'data/player_boxscores/historical/2020-2021_NBA_Box_Score_Player-Stats.xlsx',
            'data/player_boxscores/historical/NBA-2020-2021-Player-BoxScore-Dataset.xlsx',
        ],
        '2021-2022': [
            'data/player_boxscores/historical/2021-2022_NBA_Box_Score_Player-Stats.xlsx',
            'data/player_boxscores/historical/NBA-2021-2022-Player-BoxScore-Dataset.xlsx',
        ],
        '2022-2023': [
            'data/player_boxscores/historical/2022-2023_NBA_Box_Score_Player-Stats.xlsx',
            'data/player_boxscores/historical/NBA-2022-2023-Player-BoxScore-Dataset.xlsx',
        ],
        '2023-2024': [
            'data/player_boxscores/historical/2023-2024_NBA_Box_Score_Player-Stats.xlsx',
            'data/player_boxscores/historical/NBA-2023-2024-Player-BoxScore-Dataset.xlsx',
        ],
        '2024-2025': [
            'data/player_boxscores/historical/2024-2025_NBA_Box_Score_Player-Stats.xlsx',
            'data/player_boxscores/historical/NBA-2024-2025-Player-BoxScore-Dataset.xlsx',
        ],
    }
    
    # Check if it's a historical season
    if season in season_map:
        # Try each possible file path
        for file_path_str in season_map[season]:
            file_path = Path(file_path_str)
            if file_path.exists():
                df = pd.read_excel(file_path)
                if 'DATE' in df.columns:
                    df['DATE'] = pd.to_datetime(df['DATE'])
                return df
    
    # For current season (2025-2026), look in current/ directory
    if season == '2025-2026':
        current_dir = Path('data/player_boxscores/current')
        if current_dir.exists():
            # Find most recent file by sorting filenames (date prefix like 10-31-2025)
            xlsx_files = list(current_dir.glob('*.xlsx'))
            if xlsx_files:
                # Sort by filename (descending) to get most recent date
                xlsx_files_sorted = sorted(xlsx_files, key=lambda x: x.name, reverse=True)
                file_path = xlsx_files_sorted[0]
                df = pd.read_excel(file_path)
                if 'DATE' in df.columns:
                    df['DATE'] = pd.to_datetime(df['DATE'])
                return df
    
    # For unknown seasons, return None (caller should handle)
    return None

