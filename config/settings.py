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
        self._season_year = "2023-2024"  # Default season year
        self._root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
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
            season_year: Season year in format "YYYY-YYYY". If None, uses current season.
            
        Returns:
            str: Full file path to the play-by-play data
            
        Raises:
            ValueError: If season year is not supported
            FileNotFoundError: If the data file doesn't exist
        """
        if season_year is None:
            season_year = self._season_year
            
        file_mapping = self._get_file_mapping()
        
        if season_year not in file_mapping:
            raise ValueError(
                f"Season year {season_year} not supported. "
                f"Available options: {list(file_mapping.keys())}"
            )
        
        file_path = file_mapping[season_year]
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Play-by-play data file not found: {file_path}")
            
        return file_path
    
    def get_prior_season(self, current_season: Optional[str] = None) -> str:
        """
        Get the prior season year from current season.
        
        Args:
            current_season: Current season in format "YYYY-YYYY". 
                          If None, uses current season setting.
                          
        Returns:
            str: Prior season in format "YYYY-YYYY"
        """
        if current_season is None:
            current_season = self._season_year
            
        try:
            # Extract the ending year (e.g., "2023-2024" -> "2024")
            end_year = int(current_season.split('-')[1])
            prior_end_year = end_year - 1
            prior_start_year = prior_end_year - 1
            return f"{prior_start_year}-{prior_end_year}"
        except (ValueError, IndexError):
            return "2022-2023"  # Default fallback
    
    def _get_file_mapping(self) -> Dict[str, str]:
        """Get mapping of season years to file paths."""
        return {
            "2022-2023": os.path.join(
                self.play_by_play_dir, 
                "[10-18-2022]-[06-12-2023]-combined-stats.csv"
            ),
            "2023-2024": os.path.join(
                self.play_by_play_dir,
                "[10-24-2023]-[06-17-2024]-combined-stats.csv"
            ),
            "2024-2025": os.path.join(
                self.play_by_play_dir,
                "[10-22-2024]-[06-22-2025]-combined-stats.csv"
            )
        }
    
    def _validate_season_format(self, season_year: str) -> bool:
        """Validate season year format (YYYY-YYYY)."""
        try:
            parts = season_year.split('-')
            if len(parts) != 2:
                return False
            
            start_year = int(parts[0])
            end_year = int(parts[1])
            
            # End year should be start year + 1
            if end_year != start_year + 1:
                return False
                
            # Should be reasonable year range
            if start_year < 2000 or start_year > 2030:
                return False
                
            return True
        except (ValueError, IndexError):
            return False


# Global configuration instance
config = Config()


# Convenience functions for backward compatibility
def set_season_year(season_year: str) -> None:
    """Set the season year for data loading."""
    config.season_year = season_year


def get_current_season_year() -> str:
    """Get the currently configured season year."""
    return config.season_year


# Constants
TEAM_ABBREVIATIONS = {
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

# Performance settings
DEFAULT_MIN_GAMES_THRESHOLD = 10  # Keep high threshold for data quality, but cache prior season data
DEFAULT_N_TOTAL_PLAYS = 20
FAST_TEST_MODE_THRESHOLD = 100  # Use dummy PCA values below this many rows