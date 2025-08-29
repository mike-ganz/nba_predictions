"""
Configuration package for NBA predictions training data generator.
"""

from .settings import (
    config,
    set_season_year,
    get_current_season_year,
    TEAM_ABBREVIATIONS,
    DEFAULT_MIN_GAMES_THRESHOLD,
    DEFAULT_N_TOTAL_PLAYS,
    FAST_TEST_MODE_THRESHOLD
)

__all__ = [
    'config',
    'set_season_year', 
    'get_current_season_year',
    'TEAM_ABBREVIATIONS',
    'DEFAULT_MIN_GAMES_THRESHOLD',
    'DEFAULT_N_TOTAL_PLAYS', 
    'FAST_TEST_MODE_THRESHOLD'
]
