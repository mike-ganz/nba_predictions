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
import copy

# Import your existing data loading system
try:
    from data.loaders import load_play_by_play_data
    from analysis.player_stats import get_player_pca_score
    from game.team_utils import determine_home_away_teams
    from config.settings import config
    # Import compact format conversion functions
    from generate_training_data import convert_verbose_to_compact
    DATA_SYSTEM_AVAILABLE = True
except ImportError:
    print("⚠️ Data loading system not available - using sample contexts only")
    DATA_SYSTEM_AVAILABLE = False

try:
    from transform_player_stats_optimized import get_player_stats_array
    from generate_team_stats import get_team_stats_array as _get_team_stats_array
    _OPT_STATS_AVAILABLE = True
except Exception:
    _OPT_STATS_AVAILABLE = False


class GameContextBuilder:
    """Builds realistic game contexts from historical data."""
    
    def __init__(self, season_year: str = "2023-2024"):
        self.season_year = season_year
        self.play_by_play_data = None
        self.team_stats_cache = {}
        self._context_cache: Dict[Tuple[str, bool], Dict[str, Any]] = {}
        
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
        Build a complete game context for a specific game in compact format.
        
        Args:
            game_id: NBA game ID
            start_quarter: Starting quarter (1-4)
            start_time: Starting time (MM:SS format)
            recent_plays_count: Number of recent plays to include
            for_first_n_plays: If True, excludes plays array (for Stage 1 / first_N_plays mode)
            
        Returns:
            Dict in compact format with 'A', 'H', 'as', 'hs', 'ap', 'hp', 'L', 'p' fields
        """
        cache_key = (game_id, for_first_n_plays)
        if cache_key in self._context_cache:
            cached_context = self._context_cache[cache_key]
            if isinstance(cached_context, dict):
                return copy.deepcopy(cached_context)
            return cached_context
        # Prefer compact context using optimized stats if available
        if _OPT_STATS_AVAILABLE and self.play_by_play_data is not None:
            try:
                compact_context = self._build_compact_with_optimized_stats(game_id, for_first_n_plays)
                self._context_cache[cache_key] = copy.deepcopy(compact_context)
                return copy.deepcopy(compact_context)
            except Exception as _e:
                print(f"⚠️ Optimized compact build failed, falling back to verbose→compact: {_e}")
        
        # Fallback to previous behavior
        if self.play_by_play_data is not None:
            verbose_context = self._build_from_data(game_id, start_quarter, start_time, recent_plays_count, for_first_n_plays)
        else:
            verbose_context = self._build_sample_context(game_id, for_first_n_plays)
        
        try:
            from generate_training_data import convert_verbose_to_compact
            compact_context = convert_verbose_to_compact(verbose_context)
            self._context_cache[cache_key] = copy.deepcopy(compact_context)
            return copy.deepcopy(compact_context)
        except Exception as e:
            print(f"⚠️ Warning: Failed to convert to compact format: {e}")
            print("Falling back to verbose format")
            self._context_cache[cache_key] = copy.deepcopy(verbose_context)
            return copy.deepcopy(verbose_context)
    
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
            # Building clean context for first_N_plays mode (no recent_plays)
            pass
        return context
    
    def _determine_away_home_teams(self, game_data: pd.DataFrame) -> tuple[str, str]:
        """Determine away/home teams using first scoring plays.
        
        Simple and reliable: First team to increment away_score = AWAY team,
        first team to increment home_score = HOME team.
        """
        # Sort by play_id if available for chronological order
        df = game_data.sort_values('play_id') if 'play_id' in game_data.columns else game_data.copy()
        
        prev_away, prev_home = 0, 0
        away_team, home_team = None, None
        
        for _, row in df.iterrows():
            curr_away = int(row.get('away_score', 0) or 0)
            curr_home = int(row.get('home_score', 0) or 0)
            team = str(row.get('team', '') or '').strip()
            
            # First team to score away points = away team
            if away_team is None and curr_away > prev_away and team:
                away_team = team
                
            # First team to score home points = home team  
            if home_team is None and curr_home > prev_home and team:
                home_team = team
                
            prev_away, prev_home = curr_away, curr_home
            
            # Stop once we have both teams
            if away_team and home_team:
                break
        
        # Return results or fallback
        if away_team and home_team:
            return away_team, home_team
            
        # Fallback: use unique teams from 'team' column
        unique_teams = [str(t).strip() for t in game_data.get('team', pd.Series(dtype=str)).dropna().unique() if str(t).strip()]
        if len(unique_teams) >= 2:
            # Alphabetical ordering as best-effort fallback
            teams_sorted = sorted(unique_teams)
            return teams_sorted[0], teams_sorted[1]
        elif len(unique_teams) == 1:
            return unique_teams[0], "HOME"
        else:
            return "AWAY", "HOME"
    
    def _build_team_info(self, team_abbr: str, game_data: pd.DataFrame, is_home: bool) -> Dict[str, Any]:
        """Build team information with stats and player profiles."""
        
        # Get team stats (simplified - you could enhance this)
        team_stats = {
            "OEFF": round(random.uniform(108, 118), 1),  # Rounded to 1 decimal place
            "DEFF": round(random.uniform(108, 118), 1),  # Rounded to 1 decimal place
            "PACE": round(random.uniform(95, 105), 1),   # Rounded to 1 decimal place
            "REST_DAYS": random.randint(1, 4)            # Whole number
        }
        
        # Use proper roster columns for reliable player-team assignment
        if is_home:
            # Home team: use h1, h2, h3, h4, h5 columns
            roster_cols = ['h1', 'h2', 'h3', 'h4', 'h5']
        else:
            # Away team: use a1, a2, a3, a4, a5 columns
            roster_cols = ['a1', 'a2', 'a3', 'a4', 'a5']
        
        # Collect all unique players from roster columns
        players_set = set()
        for col in roster_cols:
            if col in game_data.columns:
                col_players = game_data[col].dropna().unique()
                players_set.update(col_players)
        players = list(players_set)
        
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

    def _build_compact_with_optimized_stats(self, game_id: str, for_first_n_plays: bool) -> Dict[str, Any]:
        """Build compact context using optimized player/team stats with real game date."""
        import pandas as pd
        df = self.play_by_play_data[self.play_by_play_data['game_id'] == int(game_id)]
        if df.empty:
            raise ValueError(f"No data for game {game_id}")
        # Game date
        game_date = df['date'].iloc[0]
        if hasattr(game_date, 'strftime'):
            game_date_str = game_date.strftime('%Y-%m-%d')
        else:
            game_date_str = str(game_date)
        # Determine teams
        away_abbrev, home_abbrev = self._determine_away_home_teams(df)
        
        # CRITICAL: Convert abbreviations to full team names (training pipeline requirement)
        # Training pipeline passes full names like "Cleveland Cavaliers", not "CLE"
        # See generate_training_data_OPTIMIZED.py line 895
        from generate_training_data import create_team_abbreviation_mapping
        abbrev_to_full_name = create_team_abbreviation_mapping()
        away_full_name = abbrev_to_full_name.get(away_abbrev, away_abbrev)
        home_full_name = abbrev_to_full_name.get(home_abbrev, home_abbrev)
        
        # Team stats (10-value array) → use full array for compact format
        # CRITICAL: Must pass fallback_season AND full team names to match training pipeline
        # See generate_training_data_OPTIMIZED.py lines 895, 903
        away_stats_ext = _get_team_stats_array(away_full_name, target_date=game_date_str, fallback_season=self.season_year)
        home_stats_ext = _get_team_stats_array(home_full_name, target_date=game_date_str, fallback_season=self.season_year)
        def _validate_team_stats(arr, team_name="Unknown"):
            """Validate and format team stats array to 10 values."""
            if not arr or len(arr) < 10:
                # Return default values: [OEFF, DEFF, PACE, 3PAr, FTr, ORr, DRr, ASTr, TOr, REST]
                print(f"⚠️ WARNING: {team_name} team stats using DEFAULTS (no real stats found!)")
                return [110.0, 110.0, 100.0, 0.38, 0.22, 0.25, 0.75, 0.65, 0.13, 2]
            
            # Check if array looks like defaults (telltale 110.0, 110.0, 100.0 pattern)
            if len(arr) >= 3 and arr[0] == 110.0 and arr[1] == 110.0 and arr[2] == 100.0:
                print(f"⚠️ WARNING: {team_name} team stats appear to be DEFAULTS!")
            
            # Ensure all values are properly formatted
            return [
                round(float(arr[0]), 2),  # OEFF
                round(float(arr[1]), 2),  # DEFF
                round(float(arr[2]), 2),  # PACE
                round(float(arr[3]), 3),  # 3PAr
                round(float(arr[4]), 3),  # FTr
                round(float(arr[5]), 3),  # ORr
                round(float(arr[6]), 3),  # DRr
                round(float(arr[7]), 3),  # ASTr
                round(float(arr[8]), 3),  # TOr
                int(arr[9])               # REST
            ]
        away_stats = _validate_team_stats(away_stats_ext, away_abbrev)
        home_stats = _validate_team_stats(home_stats_ext, home_abbrev)
        # Roster from lineups columns (a1..a5 / h1..h5) across the game
        def _collect_roster(df_team_cols):
            s = set()
            for c in df_team_cols:
                if c in df.columns:
                    s.update(df[c].dropna().astype(str).tolist())
            # cap to 13 (match training pipeline)
            return list(s)[:13]
        away_roster = _collect_roster(['a1','a2','a3','a4','a5'])
        home_roster = _collect_roster(['h1','h2','h3','h4','h5'])
        # Build player arrays using optimized stats with date cutoff
        def _build_players(team_roster):
            # CRITICAL: Must match training pipeline's approach
            # See generate_training_data_OPTIMIZED.py lines 684-696
            from transform_player_stats_optimized import calculate_player_stats
            
            players = []
            for name in team_roster:
                try:
                    # STEP 1: Calculate base stats to get MPG and usage (like training pipeline)
                    base_stats = calculate_player_stats(name, game_date_str, self.season_year)
                    
                    # STEP 2: Extract MPG and usage with fallbacks
                    if base_stats:
                        mpg = base_stats.get('MPG', 25)
                        usage = base_stats.get('USAGE_RATE', 0.18)
                    else:
                        mpg = 25
                        usage = 0.18
                    
                    # STEP 3: Get enhanced array WITH mpg and usage_rate parameters
                    # This matches training pipeline exactly (lines 691-693)
                    stats = get_player_stats_array(
                        name, game_date_str, 
                        mpg=mpg,              # ← Training pipeline passes this
                        usage_rate=usage,     # ← Training pipeline passes this
                        season=self.season_year, 
                        use_rolling=True
                    )
                    
                    if not stats:
                        continue
                    # stats format: [name, mpg, usg, pts/100, fga/100, ast/100, stl/100, blk/100, rim%, rim_fg%, c3%, c3_fg%, nc3%, nc3_fg%, mid%, mid_fg%, a2%, a3%, ft%]
                    # Convert to training data format (20 elements) by adding fouls
                    player_arr = stats + [0]  # Add fouls placeholder at end
                    players.append(player_arr)
                except Exception:
                    # Minimal fallback - 20 elements matching training format
                    # [name, mpg, usg, pts/poss, fga/poss, ast/poss, stl/poss, blk/poss, rim%, rim_fg%, c3%, c3_fg%, nc3%, nc3_fg%, mid%, mid_fg%, a2%, a3%, ft%, fouls]
                    players.append([name, 25, 0.18, 1.0, 0.15, 0.05, 0.015, 0.005, 0.3, 0.62, 0.05, 0.38, 0.2, 0.36, 0.3, 0.42, 0.5, 0.5, 0.75, 0])
            # Sort by MPG desc (index 1 in the 20-field format)
            # Training pipeline does this at lines 878-879
            players.sort(key=lambda p: p[1], reverse=True)
            return players
        away_players = _build_players(away_roster)
        home_players = _build_players(home_roster)
        # Build compact context - NO LINEUP LOOKUP TABLES
        # Lineups will be embedded directly in play arrays (14-value format)
        compact = {
            "A": away_abbrev,
            "H": home_abbrev,
            "as": away_stats,
            "hs": home_stats,
            "ap": away_players,
            "hp": home_players,
            "ap_count": len(away_players),
            "hp_count": len(home_players)
        }

        # For Stage 1 (first_N_plays), we omit plays and auxiliary fields to match training
        if not for_first_n_plays:
            compact["p"] = []
            compact["pos"] = "N"
            compact["tb"] = [0, 0]
            compact["sd"] = 0

        return compact


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
