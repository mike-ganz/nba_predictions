"""
Team management utilities for NBA game data processing.

This module handles team abbreviation mappings, home/away team determination,
and team-related data processing.
"""

import pandas as pd
from typing import Dict, Tuple, Optional
from config.settings import TEAM_ABBREVIATIONS


class TeamManager:
    """Handles NBA team-related operations and mappings."""
    
    def __init__(self):
        self._team_abbreviations = TEAM_ABBREVIATIONS.copy()
        self._reverse_mapping = {v: k for k, v in self._team_abbreviations.items()}
    
    @property
    def team_abbreviations(self) -> Dict[str, str]:
        """Get mapping from 3-letter team abbreviations to full team names."""
        return self._team_abbreviations.copy()
    
    def get_team_full_name(self, abbreviation: str) -> str:
        """
        Get full team name from abbreviation.
        
        Args:
            abbreviation: 3-letter team abbreviation
            
        Returns:
            str: Full team name or original abbreviation if not found
        """
        return self._team_abbreviations.get(abbreviation, abbreviation)
    
    def get_team_abbreviation(self, full_name: str) -> str:
        """
        Get team abbreviation from full name.
        
        Args:
            full_name: Full team name
            
        Returns:
            str: 3-letter abbreviation or original name if not found
        """
        return self._reverse_mapping.get(full_name, full_name)
    
    def determine_home_away_teams(self, df: pd.DataFrame) -> Dict[int, Dict[str, str]]:
        """
        Determine which team is home and which is away for each game by tracking score increments.
        
        Args:
            df: Play-by-play DataFrame with game_id, team, away_score, home_score columns
            
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
    
    def match_team_names(self, team1: str, team2: str) -> bool:
        """
        Check if two team names refer to the same team (handles abbreviations vs full names).
        
        Args:
            team1: First team name (could be abbreviation or full name)
            team2: Second team name (could be abbreviation or full name)
            
        Returns:
            bool: True if teams match
        """
        # Direct match
        if team1 == team2:
            return True
        
        # Check abbreviation to full name matches
        team1_full = self.get_team_full_name(team1)
        team2_full = self.get_team_full_name(team2)
        
        if team1_full == team2_full:
            return True
        
        # Check full name to abbreviation matches
        team1_abbrev = self.get_team_abbreviation(team1)
        team2_abbrev = self.get_team_abbreviation(team2)
        
        if team1_abbrev == team2_abbrev:
            return True
        
        # Check partial matches (e.g., "Lakers" in "Los Angeles Lakers")
        team1_words = set(team1.lower().split())
        team2_words = set(team2.lower().split())
        
        # If any significant words match (excluding common words)
        common_words = {'the', 'of', 'and', 'in', 'at', 'to', 'for'}
        significant_words1 = team1_words - common_words
        significant_words2 = team2_words - common_words
        
        if significant_words1 & significant_words2:  # Intersection
            return len(significant_words1 & significant_words2) > 0
        
        return False
    
    def validate_team_abbreviation(self, abbreviation: str) -> bool:
        """
        Validate that a team abbreviation is known.
        
        Args:
            abbreviation: Team abbreviation to validate
            
        Returns:
            bool: True if valid abbreviation
        """
        return abbreviation in self._team_abbreviations
    
    def get_division_rivals(self, team_abbrev: str) -> list:
        """
        Get division rivals for a team (simplified implementation).
        
        Args:
            team_abbrev: Team abbreviation
            
        Returns:
            list: List of rival team abbreviations in same division
        """
        # Simplified division mapping - in a real implementation, 
        # this would be more comprehensive
        divisions = {
            'Atlantic': ['BOS', 'BKN', 'NYK', 'PHI', 'TOR'],
            'Central': ['CHI', 'CLE', 'DET', 'IND', 'MIL'],
            'Southeast': ['ATL', 'CHA', 'MIA', 'ORL', 'WAS'],
            'Northwest': ['DEN', 'MIN', 'OKC', 'POR', 'UTA'],
            'Pacific': ['GSW', 'LAC', 'LAL', 'PHX', 'SAC'],
            'Southwest': ['DAL', 'HOU', 'MEM', 'NOP', 'SAS']
        }
        
        for division, teams in divisions.items():
            if team_abbrev in teams:
                return [team for team in teams if team != team_abbrev]
        
        return []
    
    def is_same_conference(self, team1_abbrev: str, team2_abbrev: str) -> bool:
        """
        Check if two teams are in the same conference.
        
        Args:
            team1_abbrev: First team abbreviation
            team2_abbrev: Second team abbreviation
            
        Returns:
            bool: True if same conference
        """
        eastern_teams = {
            'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 
            'DET', 'IND', 'MIA', 'MIL', 'NYK', 'ORL', 
            'PHI', 'TOR', 'WAS'
        }
        
        western_teams = {
            'DAL', 'DEN', 'GSW', 'HOU', 'LAC', 'LAL', 
            'MEM', 'MIN', 'NOP', 'OKC', 'PHX', 'POR', 
            'SAC', 'SAS', 'UTA'
        }
        
        team1_eastern = team1_abbrev in eastern_teams
        team2_eastern = team2_abbrev in eastern_teams
        
        return team1_eastern == team2_eastern


class TeamStatsIntegrator:
    """Integrates team statistics with game data."""
    
    def __init__(self, team_manager: TeamManager):
        self.team_manager = team_manager
    
    def get_team_stats_for_game(self, game_df: pd.DataFrame, 
                              team_mapping: Dict[int, Dict[str, str]], 
                              target_date: Optional[str] = None,
                              min_games_threshold: int = 10) -> Dict[str, any]:
        """
        Get team stats for both teams in a game.
        Falls back to prior season averages if insufficient current season data.
        
        Args:
            game_df: DataFrame for a single game
            team_mapping: Mapping of game_id to home/away teams
            target_date: Date for stats calculation
            min_games_threshold: Minimum games before using current season
            
        Returns:
            dict: Team stats for home and away teams
        """
        from generate_team_stats import generate_team_stats
        from config.settings import config
        
        game_id = game_df.iloc[0]['game_id']
        
        # Get team abbreviations from mapping
        home_abbrev = team_mapping.get(game_id, {}).get('home_team', 'Unknown')
        away_abbrev = team_mapping.get(game_id, {}).get('away_team', 'Unknown')
        
        # Convert to full team names
        home_team_full = self.team_manager.get_team_full_name(home_abbrev)
        away_team_full = self.team_manager.get_team_full_name(away_abbrev)
        
        # Determine prior season for fallback (prevents data leakage)
        prior_season = config.get_prior_season()
        
        def get_stats_with_threshold(team_name: str) -> Dict:
            """Get team stats, falling back to prior season if insufficient games"""
            current_stats = generate_team_stats(team_name, target_date)
            
            # If no current season stats OR fewer than threshold games, use prior season
            if not current_stats or current_stats.get('GAMES_PLAYED', 0) < min_games_threshold:
                fallback_stats = generate_team_stats(team_name, None, fallback_season=prior_season)
                if fallback_stats:
                    # Add metadata to indicate this is a fallback
                    fallback_stats['FALLBACK_REASON'] = (
                        f'Insufficient current season games '
                        f'({current_stats.get("GAMES_PLAYED", 0) if current_stats else 0} < {min_games_threshold})'
                    )
                    fallback_stats['USING_PRIOR_SEASON'] = True
                return fallback_stats or {}
            else:
                # Sufficient current season games, use current stats
                current_stats['USING_PRIOR_SEASON'] = False
                return current_stats
        
        # Get stats for both teams using the threshold logic
        home_stats = get_stats_with_threshold(home_team_full)
        away_stats = get_stats_with_threshold(away_team_full)
        
        return {
            'home_team_stats': home_stats,
            'away_team_stats': away_stats,
            'home_abbrev': home_abbrev,
            'away_abbrev': away_abbrev
        }


# Global instances
team_manager = TeamManager()
team_stats_integrator = TeamStatsIntegrator(team_manager)

# Convenience functions for backward compatibility
def create_team_abbreviation_mapping() -> Dict[str, str]:
    """Create mapping from 3-letter team abbreviations to full team names."""
    return team_manager.team_abbreviations


def determine_home_away_teams(df: pd.DataFrame) -> Dict[int, Dict[str, str]]:
    """Determine which team is home and which is away for each game."""
    return team_manager.determine_home_away_teams(df)


def get_team_stats_for_game(game_df: pd.DataFrame, 
                          team_mapping: Dict[int, Dict[str, str]], 
                          target_date: Optional[str] = None,
                          min_games_threshold: int = 10) -> Dict[str, any]:
    """Get team stats for both teams in a game."""
    return team_stats_integrator.get_team_stats_for_game(
        game_df, team_mapping, target_date, min_games_threshold
    )
