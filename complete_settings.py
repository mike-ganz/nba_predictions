"""
Configuration management for NBA predictions training data generation.

This module centralizes all configuration settings and provides a clean API
for managing season years, file paths, and other constants.
"""

import os
from typing import Dict, Optional


class Config:
    """Central configuration management for the NBA training data pipeline."""
    
    def __init__(self):
        self._season_year = "2024-2025"  # Default to current season
        self._root_dir = "/home/micha"  # GCP VM root directory
        
    @property
    def season_year(self) -> str:
        """Get the current season year."""
        return self._season_year
    
    @season_year.setter
    def season_year(self, value: str) -> None:
        """Set the season year with validation."""
        if not self._validate_season_format(value):
            raise ValueError(f"Invalid season format: {value}. Expected 'YYYY-YYYY'")
        self._season_year = value
        print(f"Season year set to: {self._season_year}")
    
    @property
    def data_dir(self) -> str:
        """Get the data directory path."""
        return os.path.join(self._root_dir, "data")
    
    @property
    def play_by_play_dir(self) -> str:
        """Get the play-by-play data directory."""
        return os.path.join(self.data_dir, "play_by_play", "historical")
    
    @property
    def training_dir(self) -> str:
        """Get the training data directory."""
        return os.path.join(self.data_dir, "training")
    
    def get_play_by_play_file_path(self, season_year: Optional[str] = None) -> str:
        """
        Get the file path for play-by-play data for a specific season.
        
        Args:
            season_year: Season in format "YYYY-YYYY" (e.g., "2024-2025")
                        If None, uses current season_year
        
        Returns:
            Full file path to the play-by-play CSV file
        """
        target_season = season_year if season_year is not None else self._season_year
        
        # Define mapping from season year to actual filenames
        season_file_mapping = {
            "2024-2025": "[10-22-2024]-[06-22-2025]-combined-stats.csv",
            "2023-2024": "[10-24-2023]-[06-17-2024]-combined-stats.csv", 
            "2022-2023": "[10-18-2022]-[06-12-2023]-combined-stats.csv"
        }
        
        if target_season not in season_file_mapping:
            print(f"⚠️ Season {target_season} not found. Available seasons: {list(season_file_mapping.keys())}")
            # Default to current season file
            target_season = "2024-2025"
        
        filename = season_file_mapping[target_season]
        file_path = os.path.join(self.play_by_play_dir, filename)
        
        return file_path
    
    def _validate_season_format(self, season: str) -> bool:
        """Validate season format is YYYY-YYYY."""
        try:
            parts = season.split('-')
            if len(parts) != 2:
                return False
            start_year, end_year = parts
            if len(start_year) != 4 or len(end_year) != 4:
                return False
            int(start_year)
            int(end_year)
            return True
        except ValueError:
            return False


# Create global config instance
config = Config()

# Team abbreviations mapping
TEAM_ABBREVIATIONS = {
    'Atlanta': 'ATL', 'Boston': 'BOS', 'Brooklyn': 'BKN', 'Charlotte': 'CHA',
    'Chicago': 'CHI', 'Cleveland': 'CLE', 'Dallas': 'DAL', 'Denver': 'DEN', 
    'Detroit': 'DET', 'Golden State': 'GSW', 'Houston': 'HOU', 'Indiana': 'IND',
    'LA Clippers': 'LAC', 'LA Lakers': 'LAL', 'Memphis': 'MEM', 'Miami': 'MIA',
    'Milwaukee': 'MIL', 'Minnesota': 'MIN', 'New Orleans': 'NOP', 'New York': 'NYK',
    'Oklahoma City': 'OKC', 'Orlando': 'ORL', 'Philadelphia': 'PHI', 'Phoenix': 'PHX',
    'Portland': 'POR', 'Sacramento': 'SAC', 'San Antonio': 'SAS', 'Toronto': 'TOR',
    'Utah': 'UTA', 'Washington': 'WAS'
}

# Constants used by other modules
DEFAULT_MIN_GAMES_THRESHOLD = 10  # Keep high threshold for data quality
DEFAULT_N_TOTAL_PLAYS = 500
FAST_TEST_MODE_THRESHOLD = 100  # Use dummy PCA values below this many rows
