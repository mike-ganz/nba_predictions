"""
NBA Settings Configuration for GCP
"""

import os
from typing import Optional, Dict

# Constants needed by orchestrator
DEFAULT_N_TOTAL_PLAYS = 500

class Settings:
    def __init__(self):
        self._season_year = "2023-2024"
        self.data_dir = "data"
        self.DEFAULT_N_TOTAL_PLAYS = 500
    
    @property
    def season_year(self) -> str:
        return self._season_year
    
    def get_play_by_play_file_path(self, season_year: Optional[str] = None) -> str:
        if season_year is None:
            season_year = self._season_year
        
        file_mapping = {
            "2022-2023": "[10-18-2022]-[06-12-2023]-combined-stats.csv",
            "2023-2024": "[10-24-2023]-[06-17-2024]-combined-stats.csv", 
            "2024-2025": "[10-22-2024]-[06-22-2025]-combined-stats.csv"
        }
        
        if season_year not in file_mapping:
            raise ValueError(f"Season year {season_year} not supported.")
        
        file_path = os.path.join(self.data_dir, "play_by_play", "historical", file_mapping[season_year])
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Play-by-play data file not found: {file_path}")
        
        return file_path

# Global config instance
config = Settings()

def set_season_year(season_year: str) -> None:
    """Set the season year for the configuration."""
    config._season_year = season_year
