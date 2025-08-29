"""
Scoring utilities for NBA game data processing.

This module handles score tracking, scoring team determination,
and scoring event analysis.
"""

from typing import Tuple, Optional, Dict, Any
import pandas as pd


class ScoringAnalyzer:
    """Handles NBA scoring analysis and calculations."""
    
    @staticmethod
    def determine_scoring_info(prev_away_score: int, prev_home_score: int,
                             curr_away_score: int, curr_home_score: int,
                             away_abbrev: str, home_abbrev: str) -> Tuple[Optional[str], int]:
        """
        Determine scoring team and points scored based on score changes.
        
        Args:
            prev_away_score: Previous away team score
            prev_home_score: Previous home team score
            curr_away_score: Current away team score
            curr_home_score: Current home team score
            away_abbrev: Away team abbreviation
            home_abbrev: Home team abbreviation
            
        Returns:
            tuple: (scoring_team, points_scored)
        """
        away_diff = curr_away_score - prev_away_score
        home_diff = curr_home_score - prev_home_score
        
        if away_diff > 0:
            return away_abbrev, away_diff
        elif home_diff > 0:
            return home_abbrev, home_diff
        else:
            return None, 0
    
    @staticmethod
    def classify_scoring_play(points_scored: int) -> str:
        """
        Classify the type of scoring play based on points.
        
        Args:
            points_scored: Number of points scored
            
        Returns:
            str: Type of scoring play
        """
        if points_scored == 1:
            return "Free Throw"
        elif points_scored == 2:
            return "2-Point Field Goal"
        elif points_scored == 3:
            return "3-Point Field Goal"
        elif points_scored > 3:
            return "Multiple Scoring Plays"
        else:
            return "No Score"
    
    @staticmethod
    def calculate_score_differential(away_score: int, home_score: int) -> int:
        """
        Calculate the score differential (home team perspective).
        
        Args:
            away_score: Away team score
            home_score: Home team score
            
        Returns:
            int: Score differential (positive means home team leads)
        """
        return home_score - away_score
    
    @staticmethod
    def format_score_string(away_score: int, home_score: int, 
                          away_team: str, home_team: str) -> str:
        """
        Format a score string for display.
        
        Args:
            away_score: Away team score
            home_score: Home team score
            away_team: Away team name/abbreviation
            home_team: Home team name/abbreviation
            
        Returns:
            str: Formatted score string
        """
        return f"{away_team} {int(away_score)} - {home_team} {int(home_score)}"
    
    @staticmethod
    def is_tie_game(away_score: int, home_score: int) -> bool:
        """
        Check if the game is tied.
        
        Args:
            away_score: Away team score
            home_score: Home team score
            
        Returns:
            bool: True if scores are tied
        """
        return away_score == home_score
    
    @staticmethod
    def get_leading_team(away_score: int, home_score: int,
                        away_team: str, home_team: str) -> Optional[str]:
        """
        Get the team that is currently leading.
        
        Args:
            away_score: Away team score
            home_score: Home team score
            away_team: Away team name/abbreviation
            home_team: Home team name/abbreviation
            
        Returns:
            str or None: Leading team name or None if tied
        """
        if away_score > home_score:
            return away_team
        elif home_score > away_score:
            return home_team
        else:
            return None  # Tie game
    
    @staticmethod
    def calculate_scoring_efficiency(total_points: int, possessions: int) -> float:
        """
        Calculate scoring efficiency (points per possession).
        
        Args:
            total_points: Total points scored
            possessions: Number of possessions
            
        Returns:
            float: Points per possession
        """
        if possessions == 0:
            return 0.0
        return total_points / possessions
    
    @staticmethod
    def analyze_scoring_run(scores: list, team_indicators: list) -> Dict[str, Any]:
        """
        Analyze a scoring run for momentum shifts.
        
        Args:
            scores: List of consecutive scores
            team_indicators: List indicating which team scored (same length as scores)
            
        Returns:
            dict: Analysis of the scoring run
        """
        if not scores or len(scores) != len(team_indicators):
            return {'error': 'Invalid input data'}
        
        total_points = sum(scores)
        unique_teams = set(team_indicators)
        
        team_points = {}
        for team in unique_teams:
            team_points[team] = sum(score for score, team_ind in zip(scores, team_indicators) 
                                  if team_ind == team)
        
        return {
            'total_points': total_points,
            'scoring_plays': len(scores),
            'teams_involved': len(unique_teams),
            'team_breakdown': team_points,
            'average_points_per_play': total_points / len(scores) if scores else 0,
            'dominant_team': max(team_points.items(), key=lambda x: x[1])[0] if team_points else None
        }


class GameStateTracker:
    """Tracks game state and scoring context."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the game state tracker."""
        self.away_score = 0
        self.home_score = 0
        self.scoring_plays = []
        self.quarter = 1
        self.time_remaining = "12:00"
    
    def update_score(self, away_score: int, home_score: int, 
                    quarter: int, time_remaining: str,
                    description: str = "") -> Dict[str, Any]:
        """
        Update the game state and return scoring information.
        
        Args:
            away_score: New away team score
            home_score: New home team score
            quarter: Current quarter
            time_remaining: Time remaining in quarter
            description: Play description
            
        Returns:
            dict: Scoring information for this update
        """
        # Calculate score changes
        away_diff = away_score - self.away_score
        home_diff = home_score - self.home_score
        
        scoring_info = {
            'away_score_change': away_diff,
            'home_score_change': home_diff,
            'total_score_change': away_diff + home_diff,
            'quarter': quarter,
            'time_remaining': time_remaining,
            'description': description
        }
        
        # Track scoring play if there was a score change
        if away_diff > 0 or home_diff > 0:
            play_info = {
                'quarter': quarter,
                'time_remaining': time_remaining,
                'away_points': away_diff,
                'home_points': home_diff,
                'description': description,
                'new_score': f"{away_score}-{home_score}"
            }
            self.scoring_plays.append(play_info)
        
        # Update internal state
        self.away_score = away_score
        self.home_score = home_score
        self.quarter = quarter
        self.time_remaining = time_remaining
        
        return scoring_info
    
    def get_recent_scoring_plays(self, n_plays: int = 5) -> list:
        """
        Get the most recent scoring plays.
        
        Args:
            n_plays: Number of recent plays to return
            
        Returns:
            list: Recent scoring plays
        """
        return self.scoring_plays[-n_plays:] if self.scoring_plays else []
    
    def get_scoring_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all scoring in the game.
        
        Returns:
            dict: Scoring summary statistics
        """
        if not self.scoring_plays:
            return {'total_plays': 0, 'total_away_points': 0, 'total_home_points': 0}
        
        total_away_points = sum(play['away_points'] for play in self.scoring_plays)
        total_home_points = sum(play['home_points'] for play in self.scoring_plays)
        
        return {
            'total_plays': len(self.scoring_plays),
            'total_away_points': total_away_points,
            'total_home_points': total_home_points,
            'current_score': f"{self.away_score}-{self.home_score}",
            'scoring_plays': self.scoring_plays
        }


# Global instances
scoring_analyzer = ScoringAnalyzer()

# Convenience functions for backward compatibility
def determine_scoring_info(prev_away_score: int, prev_home_score: int,
                         curr_away_score: int, curr_home_score: int,
                         away_abbrev: str, home_abbrev: str) -> Tuple[Optional[str], int]:
    """Determine scoring team and points scored based on score changes."""
    return scoring_analyzer.determine_scoring_info(
        prev_away_score, prev_home_score, curr_away_score, curr_home_score,
        away_abbrev, home_abbrev
    )
