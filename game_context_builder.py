#!/usr/bin/env python3
"""
NBA Game Context Builder

Builds game contexts from your existing training data for use in orchestration.
Integrates with your existing data loading system to create realistic starting points.
"""

import pandas as pd
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Import your existing data loading system
try:
    from data.loaders import load_play_by_play_data
    from analysis.player_stats import get_player_pca_score
    from game.team_utils import determine_home_away_teams
    from config.settings import config
    DATA_SYSTEM_AVAILABLE = True
except ImportError:
    print("⚠️ Data loading system not available - using sample contexts only")
    DATA_SYSTEM_AVAILABLE = False


class GameContextBuilder:
    """Builds realistic game contexts from historical data."""
    
    def __init__(self, season_year: str = "2023-2024"):
        self.season_year = season_year
        self.play_by_play_data = None
        self.team_stats_cache = {}
        
        if DATA_SYSTEM_AVAILABLE:
            self._load_play_by_play_data()
    
    def _load_play_by_play_data(self):
        """Load play-by-play data for the season."""
        try:
            print(f"📥 Loading play-by-play data for {self.season_year}...")
            self.play_by_play_data = load_play_by_play_data(self.season_year)
            print(f"✅ Loaded {len(self.play_by_play_data)} plays")
        except Exception as e:
            print(f"⚠️ Could not load play-by-play data: {e}")
            self.play_by_play_data = None
    
    def get_available_games(self) -> List[str]:
        """Get list of available game IDs from the data."""
        if self.play_by_play_data is None:
            # Return sample game IDs if no data available
            return ["22200001", "22200002", "22200003", "22200004", "22200005"]
        
        unique_games = self.play_by_play_data['game_id'].unique()
        return sorted([str(game_id) for game_id in unique_games])
    
    def build_game_context(self, game_id: str, start_quarter: int = 1, 
                          start_time: str = "12:00", recent_plays_count: int = 20, 
                          for_first_n_plays: bool = False) -> Dict[str, Any]:
        """
        Build a complete game context for a specific game.
        
        Args:
            game_id: NBA game ID
            start_quarter: Starting quarter (1-4)
            start_time: Starting time (MM:SS format)
            recent_plays_count: Number of recent plays to include
            for_first_n_plays: If True, excludes recent_plays (for Stage 1 / first_N_plays mode)
        """
        if self.play_by_play_data is not None:
            return self._build_from_data(game_id, start_quarter, start_time, recent_plays_count, for_first_n_plays)
        else:
            return self._build_sample_context(game_id, for_first_n_plays)
    
    def _build_from_data(self, game_id: str, start_quarter: int, 
                        start_time: str, recent_plays_count: int, for_first_n_plays: bool = False) -> Dict[str, Any]:
        """Build context from actual game data."""
        
        # Filter data for this game
        game_data = self.play_by_play_data[self.play_by_play_data['game_id'] == int(game_id)]
        
        if game_data.empty:
            print(f"⚠️ No data found for game {game_id}, using sample context")
            return self._build_sample_context(game_id)
        
        print(f"🔨 Building context for game {game_id} from {len(game_data)} plays")
        
        # Extract game date for PCA calculations
        if 'date' in game_data.columns:
            game_date = game_data['date'].iloc[0]
            if hasattr(game_date, 'strftime'):
                self.game_date = game_date.strftime('%Y-%m-%d')
            else:
                self.game_date = str(game_date)
            print(f"📅 Game date: {self.game_date}")
        else:
            self.game_date = None
            print("⚠️ No date column found in game data")
        
        # Determine home vs away teams using player assignments
        away_team_abbr, home_team_abbr = self._determine_away_home_teams(game_data)
        print(f"🏟️ Determined: {away_team_abbr} @ {home_team_abbr} (away @ home)")
        
        # Build team information with stats
        away_team = self._build_team_info(away_team_abbr, game_data, is_home=False)
        home_team = self._build_team_info(home_team_abbr, game_data, is_home=True)
        
        # Build base context (always include teams and players)
        context = {
            "away_team": away_team,
            "home_team": home_team
        }
        
        # Only include recent_plays if not for first_n_plays mode
        if not for_first_n_plays:
            recent_plays = self._extract_recent_plays(
                game_data, start_quarter, start_time, recent_plays_count
            )
            context["recent_plays"] = recent_plays
            print(f"✅ Built context: {away_team['name']} @ {home_team['name']}, {len(recent_plays)} recent plays")
        else:
            print("🎯 Building clean context for first_N_plays mode (no recent_plays)")
            print(f"✅ Built clean context: {away_team['name']} @ {home_team['name']}, ready for Stage 1")
        return context
    
    def _determine_away_home_teams(self, game_data: pd.DataFrame) -> tuple[str, str]:
        """Determine which team is away vs home using player assignments (optimized)."""
        # Single combined operation to get unique teams and players
        unique_teams = game_data['team'].dropna().unique()
        
        if len(unique_teams) >= 2:
            # Optimized: combine operations and use sets for faster lookups
            away_series = game_data['away'].dropna()
            home_series = game_data['home'].dropna()
            
            # Convert to sets once (faster lookups)
            away_players_set = set(away_series.unique()[:10])  # Slightly more players for better accuracy
            home_players_set = set(home_series.unique()[:10])
            
            team1, team2 = unique_teams[0], unique_teams[1]
            
            # Get player sets for each team in one operation
            team1_mask = game_data['team'] == team1
            team2_mask = game_data['team'] == team2
            
            team1_players = set(game_data.loc[team1_mask, 'player'].dropna().unique())
            team2_players = set(game_data.loc[team2_mask, 'player'].dropna().unique())
            
            # Optimized set intersection operations
            team1_away_matches = len(team1_players & away_players_set)
            team1_home_matches = len(team1_players & home_players_set)
            
            if team1_away_matches > team1_home_matches:
                # Team1 is away, Team2 is home
                return team1, team2
            else:
                # Team1 is home, Team2 is away  
                return team2, team1
        else:
            # Fallback if only one team found
            away_team = unique_teams[0] if len(unique_teams) > 0 else "AWAY"
            home_team = "HOME"
            return away_team, home_team
    
    def _build_team_info(self, team_abbr: str, game_data: pd.DataFrame, is_home: bool) -> Dict[str, Any]:
        """Build team information with stats and player profiles."""
        
        # Get team stats (simplified - you could enhance this)
        team_stats = {
            "OEFF": round(random.uniform(108, 118), 1),  # Rounded to 1 decimal place
            "DEFF": round(random.uniform(108, 118), 1),  # Rounded to 1 decimal place
            "PACE": round(random.uniform(95, 105), 1),   # Rounded to 1 decimal place
            "REST_DAYS": random.randint(1, 4)            # Whole number
        }
        
        # Optimized: single operation to get unique players for this team
        team_mask = game_data['team'] == team_abbr
        players = game_data.loc[team_mask, 'player'].dropna().unique()
        
        # Build player profiles
        player_profiles = []
        for player_name in players[:15]:  # Limit to reasonable roster size
            if pd.isna(player_name) or player_name == '':
                continue
                
            # Get or generate player profile
            profile = self._get_player_profile(player_name)
            player_profiles.append({
                "name": player_name,
                "profile": profile
            })
        
        # Ensure we have at least a few players
        if len(player_profiles) < 5:
            print(f"⚠️ Only found {len(player_profiles)} players for {team_abbr}, adding generic players")
            for i in range(5 - len(player_profiles)):
                player_profiles.append({
                    "name": f"{team_abbr} Player {i+1}",
                    "profile": self._generate_generic_player_profile()
                })
        
        return {
            "name": team_abbr,
            "stats": team_stats,
            "players": player_profiles[:12]  # Limit to 12 players
        }
    
    def _get_player_profile(self, player_name: str) -> Dict[str, Any]:
        """Get player profile with PCA scores."""
        
        if DATA_SYSTEM_AVAILABLE:
            try:
                # Try to get actual PCA scores with proper date and season parameters
                game_date = getattr(self, 'game_date', None)
                pca_scores = get_player_pca_score(player_name, game_date=game_date, season=self.season_year)
                if pca_scores is not None:
                    # Extract individual scores from tuple (offense, defense, shot_selection, efficiency)
                    offense, defense, shot_selection, efficiency = pca_scores
                    return {
                        "offense": round(float(offense), 2),           # 2 decimal places
                        "defense": round(float(defense), 2),           # 2 decimal places
                        "shot_selection": round(float(shot_selection), 2),  # 2 decimal places
                        "efficiency": round(float(efficiency), 2),     # 2 decimal places
                        "MPG": random.randint(15, 40),                # Whole number
                        "usage": random.randint(10, 35)               # Whole number
                    }
            except Exception as e:
                print(f"⚠️ Could not get PCA score for {player_name}: {e}")
        
        # Fallback to generic profile
        return self._generate_generic_player_profile()
    
    def _generate_generic_player_profile(self) -> Dict[str, Any]:
        """Generate a generic but realistic player profile."""
        return {
            "offense": round(random.uniform(-1, 3), 2),        # 2 decimal places
            "defense": round(random.uniform(-1, 3), 2),        # 2 decimal places
            "shot_selection": round(random.uniform(-2, 2), 2), # 2 decimal places
            "efficiency": round(random.uniform(-1.5, 1), 2),   # 2 decimal places
            "MPG": random.randint(15, 35),                     # Whole number
            "usage": random.randint(12, 28)                    # Whole number
        }
    
    def _extract_recent_plays(self, game_data: pd.DataFrame, start_quarter: int, 
                             start_time: str, count: int) -> List[Dict[str, Any]]:
        """Extract recent plays leading up to a specific game state."""
        
        # Filter plays up to the starting point (simplified - take most recent)
        filtered_data = game_data[
            game_data['period'] <= start_quarter
        ].copy()
        
        # Sort by game order (latest first) and take the most recent plays
        filtered_data = filtered_data.tail(count)
        
        recent_plays = []
        # Use the same team assignment logic
        away_team, home_team = self._determine_away_home_teams(game_data)
        
        for _, play in filtered_data.iterrows():
            
            # Build play object with correct column names
            play_obj = {
                "quarter": int(play['period']) if pd.notna(play['period']) else 1,
                "time_remaining": str(play['remaining_time']) if pd.notna(play['remaining_time']) else "12:00",
                "score": f"{away_team} {int(play.get('away_score', 0))} - {home_team} {int(play.get('home_score', 0))}",
                "players_on_court": [
                    {"team": away_team, "players": ["Player1", "Player2", "Player3", "Player4", "Player5"]},
                    {"team": home_team, "players": ["PlayerA", "PlayerB", "PlayerC", "PlayerD", "PlayerE"]}
                ],
                "player": str(play.get('player', '')) if pd.notna(play.get('player')) else None,
                "description": str(play.get('description', '')) if pd.notna(play.get('description')) else 'Unknown play',
                "shot_details": {
                    "team": play.get('team') if pd.notna(play.get('team')) else None,
                    "points": int(play.get('points', 0)) if pd.notna(play.get('points')) else None
                }
            }
            
            recent_plays.append(play_obj)
        
        # Reverse to get chronological order
        return list(reversed(recent_plays))
    
    def _build_sample_context(self, game_id: str, for_first_n_plays: bool = False) -> Dict[str, Any]:
        """Build a sample context when no data is available."""
        
        # Simple mapping of game IDs to team matchups
        sample_matchups = {
            "22200001": ("ATL", "DET"),
            "22200002": ("ORL", "CLE"), 
            "22200003": ("LAL", "GSW"),
            "22200004": ("BOS", "MIA"),
            "22200005": ("PHX", "DEN")
        }
        
        if game_id in sample_matchups:
            away, home = sample_matchups[game_id]
        else:
            # Random teams if game ID not recognized
            teams = ["ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW"]
            away, home = random.sample(teams, 2)
        
        context = {
            "away_team": {
                "name": away,
                "stats": {"OEFF": random.uniform(110, 116), "DEFF": random.uniform(110, 116), 
                         "PACE": random.uniform(96, 104), "REST_DAYS": random.randint(1, 3)},
                "players": [{"name": f"{away} Player {i}", "profile": self._generate_generic_player_profile()} 
                           for i in range(1, 11)]
            },
            "home_team": {
                "name": home,
                "stats": {"OEFF": random.uniform(110, 116), "DEFF": random.uniform(110, 116), 
                         "PACE": random.uniform(96, 104), "REST_DAYS": random.randint(1, 3)},
                "players": [{"name": f"{home} Player {i}", "profile": self._generate_generic_player_profile()} 
                           for i in range(1, 11)]
            }
        }
        
        # Only add recent_plays if not for first_n_plays mode
        if not for_first_n_plays:
            context["recent_plays"] = [
                {
                    "quarter": 1,
                    "time_remaining": "08:30",
                    "score": f"{away} 12 - {home} 15",
                    "players_on_court": [
                        {"team": away, "players": [f"{away} Player {i}" for i in range(1, 6)]},
                        {"team": home, "players": [f"{home} Player {i}" for i in range(1, 6)]}
                    ],
                    "player": f"{away} Player 1",
                    "description": f"{away} Player 1 18' Jump Shot",
                    "shot_details": {"team": away, "points": 2}
                }
            ]
        
        return context
    
    def save_context_to_file(self, context: Dict[str, Any], file_path: str):
        """Save a game context to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump(context, f, indent=2)
        print(f"💾 Context saved to {file_path}")
    
    def batch_build_contexts(self, game_ids: List[str], output_dir: str = "game_contexts"):
        """Build contexts for multiple games and save them."""
        Path(output_dir).mkdir(exist_ok=True)
        
        contexts = {}
        for game_id in game_ids:
            print(f"\n🔨 Building context for game {game_id}...")
            context = self.build_game_context(game_id)
            contexts[game_id] = context
            
            # Save individual file
            file_path = Path(output_dir) / f"game_{game_id}_context.json"
            self.save_context_to_file(context, str(file_path))
        
        # Save all contexts in one file
        all_contexts_path = Path(output_dir) / "all_game_contexts.json"
        with open(all_contexts_path, 'w') as f:
            json.dump(contexts, f, indent=2)
        
        print(f"\n✅ Built {len(contexts)} game contexts in {output_dir}/")
        return contexts


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Build NBA game contexts')
    parser.add_argument('--season', type=str, default='2023-2024', help='Season year')
    parser.add_argument('--games', type=str, help='Comma-separated list of game IDs')
    parser.add_argument('--list-games', action='store_true', help='List available games')
    parser.add_argument('--output-dir', type=str, default='game_contexts', help='Output directory')
    parser.add_argument('--sample', type=int, help='Build contexts for N random games')
    
    args = parser.parse_args()
    
    builder = GameContextBuilder(args.season)
    
    if args.list_games:
        available_games = builder.get_available_games()
        print(f"📋 Available games for {args.season}:")
        for game in available_games[:20]:  # Show first 20
            print(f"  • {game}")
        if len(available_games) > 20:
            print(f"  ... and {len(available_games) - 20} more games")
        return
    
    if args.games:
        game_ids = [g.strip() for g in args.games.split(',')]
    elif args.sample:
        available_games = builder.get_available_games()
        game_ids = random.sample(available_games, min(args.sample, len(available_games)))
        print(f"🎲 Selected {len(game_ids)} random games: {game_ids}")
    else:
        # Default to first few games
        available_games = builder.get_available_games()
        game_ids = available_games[:5]
        print(f"📋 Using default games: {game_ids}")
    
    # Build contexts
    contexts = builder.batch_build_contexts(game_ids, args.output_dir)
    
    # Show summary
    print(f"\n📊 Summary:")
    for game_id, context in contexts.items():
        away = context['away_team']['name']
        home = context['home_team']['name']
        plays = len(context['recent_plays'])
        print(f"  {game_id}: {away} @ {home} ({plays} recent plays)")


if __name__ == "__main__":
    main()
