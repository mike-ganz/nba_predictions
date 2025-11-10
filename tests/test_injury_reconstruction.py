"""
Unit tests for injury reconstruction logic in player_data_loader.py
"""
import pandas as pd
import pytest
from datetime import datetime, timedelta

from scripts.player_data_loader import get_team_players_with_injuries, _get_baseline_roster


def create_test_player_data(games_data):
    """
    Helper to create a DataFrame from test data.
    
    games_data: List of dicts with keys:
        - player_name
        - date (datetime)
        - team
        - minutes
        - pts, fga, fta (for TS% calculation)
        - usage_rate
    """
    rows = []
    for game in games_data:
        rows.append({
            'PLAYER \nFULL NAME': game['player_name'],
            'DATE': game['date'],
            'OWN \nTEAM': game['team'],
            'TEAM_CITY': game['team'].split()[0],  # e.g., "Milwaukee Bucks" -> "Milwaukee"
            'MIN': game['minutes'],
            'PTS': game.get('pts', game['minutes'] * 0.8),  # Default ~0.8 pts/min
            'FGA': game.get('fga', game['minutes'] * 0.4),  # Default
            'FTA': game.get('fta', game['minutes'] * 0.2),  # Default
            'USAGE \nRATE (%)': game.get('usage_rate', 22.0)
        })
    return pd.DataFrame(rows)


class TestInjuryReconstruction:
    """Test cases for injury reconstruction logic"""
    
    def test_case_1_player_missing_high_baseline(self):
        """
        Test Case 1: Player in recent games, missing from current game.
        Player with 25 baseline minutes who doesn't appear in current boxscore
        should be marked as injured (projected_minutes=0.0).
        """
        # Create 10 prior games where player played 25 minutes
        base_date = datetime(2024, 1, 1)
        prior_games = []
        for i in range(10):
            prior_games.append({
                'player_name': 'John Star',
                'date': base_date + timedelta(days=i),
                'team': 'Milwaukee Bucks',
                'minutes': 25.0,
                'pts': 20.0,
                'fga': 10.0,
                'fta': 5.0,
                'usage_rate': 28.0
            })
        
        # Current game date: player is missing from boxscore
        current_date = base_date + timedelta(days=10)
        
        # Add some other players to the current game
        current_game = [
            {
                'player_name': 'Other Player',
                'date': current_date,
                'team': 'Milwaukee Bucks',
                'minutes': 30.0,
                'pts': 15.0,
                'fga': 8.0,
                'fta': 3.0,
                'usage_rate': 20.0
            }
        ]
        
        # Combine all data
        all_data = prior_games + current_game
        df = create_test_player_data(all_data)
        
        # Get roster for current game
        players = get_team_players_with_injuries(
            team_name='Milwaukee',
            game_date=current_date.strftime('%Y-%m-%d'),
            season='2023-2024',
            player_boxscore_df=df,
            lookback_games=10,
            injury_threshold_mins=10.0,
            min_games_for_roster=3
        )
        
        # Find John Star in the roster
        john = next((p for p in players if p['player_name'] == 'John Star'), None)
        
        assert john is not None, "Player should be in roster"
        assert john['baseline_minutes'] == 25.0, "Baseline minutes should be 25.0"
        assert john['projected_minutes'] == 0.0, "Player should be marked as injured (projected_minutes=0.0)"
    
    def test_case_2_player_played_zero_minutes(self):
        """
        Test Case 2: Player in recent games, played 0 minutes in current game.
        Player with 20 baseline minutes who appears in boxscore with 0 MIN
        should be marked as OUT (projected_minutes=0.0).
        """
        base_date = datetime(2024, 1, 1)
        
        # Prior games: player played 20 minutes
        prior_games = []
        for i in range(10):
            prior_games.append({
                'player_name': 'Jane Rotation',
                'date': base_date + timedelta(days=i),
                'team': 'Milwaukee Bucks',
                'minutes': 20.0,
                'pts': 12.0,
                'fga': 6.0,
                'fta': 3.0,
                'usage_rate': 22.0
            })
        
        # Current game: player in boxscore with 0 minutes
        current_date = base_date + timedelta(days=10)
        current_game = [
            {
                'player_name': 'Jane Rotation',
                'date': current_date,
                'team': 'Milwaukee Bucks',
                'minutes': 0.0,  # DNP
                'pts': 0.0,
                'fga': 0.0,
                'fta': 0.0,
                'usage_rate': 22.0
            }
        ]
        
        all_data = prior_games + current_game
        df = create_test_player_data(all_data)
        
        players = get_team_players_with_injuries(
            team_name='Milwaukee',
            game_date=current_date.strftime('%Y-%m-%d'),
            season='2023-2024',
            player_boxscore_df=df
        )
        
        jane = next((p for p in players if p['player_name'] == 'Jane Rotation'), None)
        
        assert jane is not None, "Player should be in roster"
        assert jane['projected_minutes'] == 0.0, "Player with 0 minutes should be marked OUT"
    
    def test_case_3_bench_player_missing(self):
        """
        Test Case 3: Bench player missing from current game.
        Player with 8 baseline minutes (below 10.0 threshold) who doesn't appear
        should NOT be in roster (DNP-Coach's Decision).
        """
        base_date = datetime(2024, 1, 1)
        
        # Prior games: bench player with 8 minutes
        prior_games = []
        for i in range(10):
            prior_games.append({
                'player_name': 'Bench Guy',
                'date': base_date + timedelta(days=i),
                'team': 'Milwaukee Bucks',
                'minutes': 8.0,
                'pts': 4.0,
                'fga': 2.0,
                'fta': 1.0,
                'usage_rate': 18.0
            })
        
        # Current game: bench player not in boxscore
        current_date = base_date + timedelta(days=10)
        current_game = [
            {
                'player_name': 'Other Player',
                'date': current_date,
                'team': 'Milwaukee Bucks',
                'minutes': 30.0,
                'pts': 15.0,
                'fga': 8.0,
                'fta': 3.0,
                'usage_rate': 20.0
            }
        ]
        
        all_data = prior_games + current_game
        df = create_test_player_data(all_data)
        
        players = get_team_players_with_injuries(
            team_name='Milwaukee',
            game_date=current_date.strftime('%Y-%m-%d'),
            season='2023-2024',
            player_boxscore_df=df,
            injury_threshold_mins=10.0
        )
        
        bench_guy = next((p for p in players if p['player_name'] == 'Bench Guy'), None)
        
        assert bench_guy is None, "Bench player below threshold should NOT be in roster"
    
    def test_case_4_player_played_normally(self):
        """
        Test Case 4: Player played normally.
        Player with 30 baseline minutes who plays 28 minutes in current game
        should be marked as healthy (projected_minutes=None).
        """
        base_date = datetime(2024, 1, 1)
        
        # Prior games: player averaged 30 minutes
        prior_games = []
        for i in range(10):
            prior_games.append({
                'player_name': 'Superstar',
                'date': base_date + timedelta(days=i),
                'team': 'Milwaukee Bucks',
                'minutes': 30.0,
                'pts': 25.0,
                'fga': 15.0,
                'fta': 7.0,
                'usage_rate': 32.0
            })
        
        # Current game: player plays 28 minutes
        current_date = base_date + timedelta(days=10)
        current_game = [
            {
                'player_name': 'Superstar',
                'date': current_date,
                'team': 'Milwaukee Bucks',
                'minutes': 28.0,
                'pts': 24.0,
                'fga': 14.0,
                'fta': 6.0,
                'usage_rate': 32.0
            }
        ]
        
        all_data = prior_games + current_game
        df = create_test_player_data(all_data)
        
        players = get_team_players_with_injuries(
            team_name='Milwaukee',
            game_date=current_date.strftime('%Y-%m-%d'),
            season='2023-2024',
            player_boxscore_df=df
        )
        
        superstar = next((p for p in players if p['player_name'] == 'Superstar'), None)
        
        assert superstar is not None, "Player should be in roster"
        assert superstar['projected_minutes'] is None, "Healthy player should have projected_minutes=None"
    
    def test_min_games_threshold(self):
        """
        Test that players need to appear in at least min_games_for_roster games
        to be included in the baseline roster.
        """
        base_date = datetime(2024, 1, 1)
        
        # Player only appeared in 2 games (below 3-game minimum)
        prior_games = []
        for i in [0, 1]:  # Only 2 games
            prior_games.append({
                'player_name': 'Call Up',
                'date': base_date + timedelta(days=i),
                'team': 'Milwaukee Bucks',
                'minutes': 15.0,
                'pts': 10.0,
                'fga': 5.0,
                'fta': 2.0,
                'usage_rate': 20.0
            })
        
        # Add filler games for other players
        for i in range(10):
            prior_games.append({
                'player_name': 'Regular',
                'date': base_date + timedelta(days=i),
                'team': 'Milwaukee Bucks',
                'minutes': 25.0,
                'pts': 15.0,
                'fga': 8.0,
                'fta': 3.0,
                'usage_rate': 22.0
            })
        
        current_date = base_date + timedelta(days=10)
        df = create_test_player_data(prior_games)
        
        players = get_team_players_with_injuries(
            team_name='Milwaukee',
            game_date=current_date.strftime('%Y-%m-%d'),
            season='2023-2024',
            player_boxscore_df=df,
            min_games_for_roster=3
        )
        
        call_up = next((p for p in players if p['player_name'] == 'Call Up'), None)
        
        assert call_up is None, "Player with only 2 games should not be in baseline roster"
    
    def test_baseline_roster_helper(self):
        """Test the _get_baseline_roster helper function directly."""
        base_date = datetime(2024, 1, 1)
        
        # Create data for 3 players over 10 games
        games_data = []
        for i in range(10):
            games_data.extend([
                {
                    'player_name': 'Player A',
                    'date': base_date + timedelta(days=i),
                    'team': 'Milwaukee Bucks',
                    'minutes': 30.0,
                    'pts': 22.0,
                    'fga': 12.0,
                    'fta': 5.0,
                    'usage_rate': 28.0
                },
                {
                    'player_name': 'Player B',
                    'date': base_date + timedelta(days=i),
                    'team': 'Milwaukee Bucks',
                    'minutes': 20.0,
                    'pts': 12.0,
                    'fga': 7.0,
                    'fta': 3.0,
                    'usage_rate': 22.0
                }
            ])
        
        df = create_test_player_data(games_data)
        
        current_date = base_date + timedelta(days=10)
        baseline = _get_baseline_roster(
            team_name='Milwaukee',
            game_date=current_date.strftime('%Y-%m-%d'),
            season='2023-2024',
            player_boxscore_df=df,
            lookback_games=10,
            min_games_for_roster=3
        )
        
        assert 'Player A' in baseline, "Player A should be in baseline roster"
        assert 'Player B' in baseline, "Player B should be in baseline roster"
        assert baseline['Player A']['baseline_minutes'] == 30.0
        assert baseline['Player B']['baseline_minutes'] == 20.0
        assert baseline['Player A']['games_played'] == 10
        assert baseline['Player B']['games_played'] == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

