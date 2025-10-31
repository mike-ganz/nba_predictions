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
    # Map season to file paths
    season_map = {
        '2021-2022': 'data/player_boxscores/historical/2021-2022_NBA_Box_Score_Player-Stats.xlsx',
        '2022-2023': 'data/player_boxscores/historical/2022-2023_NBA_Box_Score_Player-Stats.xlsx',
        '2023-2024': 'data/player_boxscores/historical/2023-2024_NBA_Box_Score_Player-Stats.xlsx',
        '2024-2025': 'data/player_boxscores/historical/2024-2025_NBA_Box_Score_Player-Stats.xlsx',
    }
    
    # Check if it's a historical season
    if season in season_map:
        file_path = Path(season_map[season])
        if file_path.exists():
            df = pd.read_excel(file_path)
            if 'DATE' in df.columns:
                df['DATE'] = pd.to_datetime(df['DATE'])
            return df
    
    # For current season or unknown, return None (caller should handle)
    return None

