#!/usr/bin/env python3
"""
NBA Predictions Configuration Settings
"""

import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

# Global constants
DEFAULT_N_TOTAL_PLAYS = 500

@dataclass
class NBAConfig:
    """Configuration settings for NBA predictions system."""
    
    # Data paths - use local paths on GCP VM
    DATA_BASE_PATH = "/home/micha/data"
    
    # Default season and settings
    SEASON_YEAR = "2024-2025"
    DEFAULT_N_TOTAL_PLAYS = 500
    
    # File paths for different data types
    PLAY_BY_PLAY_PATH = f"{DATA_BASE_PATH}/play_by_play/historical"
    PLAYER_BOXSCORES_PATH = f"{DATA_BASE_PATH}/player_boxscores" 
    TEAM_BOXSCORES_PATH = f"{DATA_BASE_PATH}/team_boxscores"
    SCHEDULES_PATH = f"{DATA_BASE_PATH}/schedules"
    
    def get_play_by_play_file(self, season_year: str = None) -> str:
        """Get the play-by-play file for a specific season."""
        season = season_year or self.SEASON_YEAR
        
        # Map season years to files
        season_files = {
            "2024-2025": "[10-22-2024]-[06-22-2025]-combined-stats.csv",
            "2023-2024": "[10-24-2023]-[06-17-2024]-combined-stats.csv", 
            "2022-2023": "[10-18-2022]-[06-12-2023]-combined-stats.csv"
        }
        
        if season in season_files:
            return f"{self.PLAY_BY_PLAY_PATH}/{season_files[season]}"
        else:
            # Default to 2024-2025 file
            return f"{self.PLAY_BY_PLAY_PATH}/{season_files['2024-2025']}"

# Create global config instance
config = NBAConfig()

# Team abbreviations for compatibility
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

# Export commonly used values for backward compatibility
season_year = config.SEASON_YEAR
DATA_BASE_PATH = config.DATA_BASE_PATH
