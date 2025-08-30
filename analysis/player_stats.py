"""
Player statistics and analysis utilities for NBA training data generation.

This module handles player-related analysis including PCA score integration,
lineup management, and player performance metrics.
"""

import hashlib
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from config.settings import config, FAST_TEST_MODE_THRESHOLD


class PlayerAnalyzer:
    """Handles player statistics and analysis."""
    
    def __init__(self):
        self._pca_cache = {}  # Cache for PCA scores to improve performance
    
    def get_player_pca_scores(self, player_name: str, game_date: Optional[str] = None,
                            season: Optional[str] = None, use_cache: bool = True, 
                            force_fast_mode: bool = False, force_real_mode: bool = False) -> Tuple[float, float, float, float]:
        """
        Get PCA scores for a player with caching and fast test mode support.
        
        Args:
            player_name: Name of the player
            game_date: Date of the game for context
            season: Season year for context
            use_cache: Whether to use cached results
            force_fast_mode: If True, always use dummy values (for testing)
            force_real_mode: If True, always use real PCA calculations (for cache building)
            
        Returns:
            tuple: (offense, defense, shot_selection, efficiency) scores
        """
        # Create cache key
        cache_key = f"{player_name}_{game_date}_{season}"
        
        # Return cached result if available and caching is enabled
        if use_cache and cache_key in self._pca_cache:
            return self._pca_cache[cache_key]
        
        try:
            # Determine which mode to use based on parameters
            if force_fast_mode:
                # Explicitly requested dummy values
                scores = self._generate_dummy_pca_scores(player_name)
            elif force_real_mode:
                # Explicitly requested real PCA (cache building mode)
                from pca_optimized import get_player_pca_score
                scores = get_player_pca_score(player_name, game_date, season)
            elif self._should_use_fast_mode():
                # Normal fast mode logic (small cache size)
                scores = self._generate_dummy_pca_scores(player_name)
            else:
                # Use OPTIMIZED PCA computation with your 306K+ cache files
                from pca_optimized import get_player_pca_score
                scores = get_player_pca_score(player_name, game_date, season)
            
            # Cache the result
            if use_cache:
                self._pca_cache[cache_key] = scores
            
            return scores
            
        except Exception as e:
            print(f"Warning: Could not get PCA scores for {player_name}: {e}")
            # Return default values
            return (None, None, None, None)
    
    def _should_use_fast_mode(self) -> bool:
        """
        Determine if we should use fast test mode based on dataset size.
        Uses FAST_TEST_MODE_THRESHOLD to decide when to use dummy values.
        """
        # DISABLED: Always use real PCA values since user has pre-built cache
        # The cache size detection was incorrectly triggering fast mode
        return False
        
        # Old buggy logic that caused dummy values for full season runs:
        # if len(self._pca_cache) < FAST_TEST_MODE_THRESHOLD:
        #     return True  # This was incorrectly triggering for full seasons!
    
    def _generate_dummy_pca_scores(self, player_name: str) -> Tuple[float, float, float, float]:
        """
        Generate consistent dummy PCA values for testing mode.
        
        Args:
            player_name: Player name to generate consistent values for
            
        Returns:
            tuple: (offense, defense, shot_selection, efficiency) scores
        """
        print(f"🚀 FAST TEST MODE: Using dummy PCA values for {player_name}")
        
        # Generate consistent dummy values based on player name hash
        name_hash = int(hashlib.md5(player_name.encode()).hexdigest()[:8], 16)
        
        offense = (name_hash % 200 - 100) / 100.0  # -1.0 to 1.0 range
        defense = ((name_hash >> 8) % 200 - 100) / 100.0
        shot_selection = ((name_hash >> 16) % 200 - 100) / 100.0
        efficiency = ((name_hash >> 24) % 200 - 100) / 100.0
        
        return offense, defense, shot_selection, efficiency
    
    def convert_pca_scores_to_integers(self, offense: Optional[float], defense: Optional[float],
                                     shot_selection: Optional[float], efficiency: Optional[float],
                                     scale_factor: int = 100) -> Dict[str, Optional[int]]:
        """
        Convert PCA scores to integers for JSON serialization.
        
        Args:
            offense: Offense PCA score
            defense: Defense PCA score
            shot_selection: Shot selection PCA score
            efficiency: Efficiency PCA score
            scale_factor: Factor to scale scores by (default 100)
            
        Returns:
            dict: Integer-scaled PCA scores
        """
        return {
            'offense': int(round(offense * scale_factor)) if offense is not None else None,
            'defense': int(round(defense * scale_factor)) if defense is not None else None,
            'shot_selection': int(round(shot_selection * scale_factor)) if shot_selection is not None else None,
            'efficiency': int(round(efficiency * scale_factor)) if efficiency is not None else None
        }
    
    def create_player_object(self, player_name: str, team_abbrev: str,
                           game_date: Optional[str] = None, season: Optional[str] = None,
                           fast_mode: bool = False, real_mode: bool = False) -> Dict[str, Any]:
        """
        Create a complete player object with stats for training data.
        
        Args:
            player_name: Name of the player
            team_abbrev: Team abbreviation
            game_date: Date of the game
            season: Season year
            fast_mode: If True, use dummy PCA values for faster processing
            real_mode: If True, always use real PCA calculations (for cache building)
            
        Returns:
            dict: Complete player object with stats
        """
        # Get PCA scores - USE FULL SEASON FORMAT for cache compatibility
        # Convert "2024" back to "2023-2024" to match your cache files
        full_season = f"2023-{season}" if season == "2024" else season
        offense, defense, shot_selection, efficiency = self.get_player_pca_scores(
            player_name, game_date, full_season, force_fast_mode=fast_mode, force_real_mode=real_mode
        )
        
        # Keep decimal PCA scores (no integer conversion)
        return {
            "name": str(player_name),
            "team": str(team_abbrev),
            "pca_scores": {
                'offense': round(offense, 4) if offense is not None else None,
                'defense': round(defense, 4) if defense is not None else None,
                'shot_selection': round(shot_selection, 4) if shot_selection is not None else None,
                'efficiency': round(efficiency, 4) if efficiency is not None else None
            }
        }
    
    def clear_pca_cache(self) -> int:
        """
        Clear the PCA score cache.
        
        Returns:
            int: Number of cached entries cleared
        """
        cache_size = len(self._pca_cache)
        self._pca_cache.clear()
        print(f"Cleared {cache_size} entries from PCA cache")
        return cache_size
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get statistics about the PCA cache.
        
        Returns:
            dict: Cache statistics
        """
        return {
            'cached_entries': len(self._pca_cache),
            'memory_estimate_kb': len(str(self._pca_cache)) // 1024
        }


class LineupManager:
    """Manages player lineups and team rosters."""
    
    def __init__(self, player_analyzer: Optional[PlayerAnalyzer] = None):
        self.player_analyzer = player_analyzer or PlayerAnalyzer()
    
    def get_lineup_by_game_id(self, game_id: int, boxscore_data: Optional[pd.DataFrame] = None, 
                             max_date: Optional[str] = None, current_season: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Get player lineups for a specific game.
        
        Args:
            game_id: Game ID to get lineups for
            boxscore_data: Player boxscore data (optional, will load if not provided)
            max_date: Only load data up to this date (YYYY-MM-DD format)
            current_season: Only load data for this season and prior (e.g., "2023-2024")
            
        Returns:
            dict: Dictionary mapping team names to player lists
        """
        try:
            from generate_lineup import get_lineup_by_game_id
            return get_lineup_by_game_id(game_id, boxscore_data, max_date, current_season)
        except ImportError as e:
            print(f"Warning: Could not import lineup function: {e}")
            return {}
        except Exception as e:
            print(f"Warning: Could not get lineups for game {game_id}: {e}")
            return {}
    
    def process_lineups_for_training_data(self, lineups: Dict[str, List[str]], 
                                        away_abbrev: str, home_abbrev: str,
                                        away_full_name: str, home_full_name: str,
                                        game_date: Optional[str] = None,
                                        season: Optional[str] = None,
                                        fast_mode: bool = False) -> List[Dict[str, Any]]:
        """
        Process lineups to create player objects with PCA stats for training data.
        
        Args:
            lineups: Dictionary mapping team names to player lists
            away_abbrev: Away team abbreviation
            home_abbrev: Home team abbreviation
            away_full_name: Away team full name
            home_full_name: Home team full name
            game_date: Date of the game
            season: Season year
            fast_mode: If True, use dummy PCA values for faster processing
            
        Returns:
            list: List of player objects with stats
        """
        players = []
        
        for team_name, player_list in lineups.items():
            # Determine if this lineup is for away or home team
            team_abbrev = self._match_team_to_abbreviation(
                team_name, away_abbrev, home_abbrev, away_full_name, home_full_name
            )
            
            if team_abbrev:
                for player_name in player_list:
                    player_obj = self.player_analyzer.create_player_object(
                        player_name, team_abbrev, game_date, season, fast_mode=fast_mode, real_mode=(not fast_mode)
                    )
                    players.append(player_obj)
        
        return players
    
    def _match_team_to_abbreviation(self, team_name: str, away_abbrev: str, home_abbrev: str,
                                  away_full_name: str, home_full_name: str) -> Optional[str]:
        """
        Match a team name from lineup data to the correct abbreviation.
        
        Args:
            team_name: Team name from lineup data
            away_abbrev: Away team abbreviation
            home_abbrev: Home team abbreviation
            away_full_name: Away team full name
            home_full_name: Home team full name
            
        Returns:
            str or None: Matched team abbreviation
        """
        # Check for away team match
        if (away_full_name in team_name or team_name in away_full_name or 
            any(part in team_name for part in away_full_name.split())):
            return away_abbrev
        
        # Check for home team match
        elif (home_full_name in team_name or team_name in home_full_name or
              any(part in team_name for part in home_full_name.split())):
            return home_abbrev
        
        return None
    
    def validate_lineup_data(self, lineups: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Validate lineup data for completeness and correctness.
        
        Args:
            lineups: Lineup data to validate
            
        Returns:
            dict: Validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'team_count': len(lineups),
            'total_players': sum(len(players) for players in lineups.values())
        }
        
        if not lineups:
            validation_results['is_valid'] = False
            validation_results['errors'].append("No lineup data provided")
            return validation_results
        
        for team_name, player_list in lineups.items():
            if not player_list:
                validation_results['warnings'].append(f"No players found for team: {team_name}")
            elif len(player_list) < 5:
                validation_results['warnings'].append(
                    f"Team {team_name} has only {len(player_list)} players (expected 5+)"
                )
            elif len(player_list) > 15:
                validation_results['warnings'].append(
                    f"Team {team_name} has {len(player_list)} players (unusually high)"
                )
        
        return validation_results


# Global instances
player_analyzer = PlayerAnalyzer()
lineup_manager = LineupManager(player_analyzer)

# Convenience functions for backward compatibility
def get_player_pca_score(player_name: str, game_date: Optional[str] = None,
                        season: Optional[str] = None) -> Tuple[float, float, float, float]:
    """Get PCA scores for a player."""
    return player_analyzer.get_player_pca_scores(player_name, game_date, season)


def get_lineup_by_game_id(game_id: int, boxscore_data: Optional[pd.DataFrame] = None) -> Dict[str, List[str]]:
    """Get player lineups for a specific game."""
    return lineup_manager.get_lineup_by_game_id(game_id, boxscore_data)
