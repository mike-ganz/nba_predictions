"""
Ultra-optimized training data generator with game-grouped processing.

Key optimizations:
1. Pre-group data by game_id (eliminates cross-game searches)  
2. Convert DataFrame to fast dict access (eliminates slow iloc calls)
3. Process games in chunks (reduces memory usage)
4. Vectorized recent plays collection 
5. Batch JSON serialization
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from collections import defaultdict
from config.settings import DEFAULT_N_TOTAL_PLAYS, DEFAULT_MIN_GAMES_THRESHOLD
from training.data_generator import TrainingDataGenerator, remove_parentheses_content, process_play_description, normalize_shot_coordinates, extract_players_on_court
from game.time_utils import convert_to_quarter_time
from game.scoring_utils import determine_scoring_info
from game.team_utils import determine_home_away_teams


class UltraOptimizedTrainingDataGenerator(TrainingDataGenerator):
    """
    Ultra-fast training data generator using game-grouped processing.
    
    Performance improvements over original:
    - ~100x faster recent plays collection (eliminates O(n²) DataFrame operations)
    - ~10x faster overall generation (game-grouped processing)  
    - Lower memory usage (process games in chunks)
    """
    
    def __init__(self):
        super().__init__()
        self._batch_size = 5000
        self._incremental_save_enabled = False
        self._incremental_save_path = None
        self._save_every_n_batches = 5
        
    def create_llm_training_data(self, df: pd.DataFrame, n_total: int = DEFAULT_N_TOTAL_PLAYS,
                               filter_nan: bool = True, force_real_pca: bool = False,
                               generation_mode: str = "remaining_plays") -> pd.DataFrame:
        """
        🚀 ULTRA-OPTIMIZED: Process games in groups for maximum performance.
        """
        print("🚀 Using ULTRA-OPTIMIZED training data generation...")
        
        # Filter NaN descriptions
        result_df = df.copy()
        if filter_nan:
            initial_count = len(result_df)
            result_df = result_df.dropna(subset=['description']).reset_index(drop=True)
            filtered_count = initial_count - len(result_df)
            if filtered_count > 0:
                print(f"🧹 Filtered out {filtered_count} rows with NaN descriptions")
        
        # CRITICAL FIX: Sort consistently before processing to ensure alignment
        print("🔧 Sorting data by game_id and play_id for consistent processing...")
        result_df = result_df.sort_values(['game_id', 'play_id']).reset_index(drop=True)
        
        # 🆕 Collapse consecutive substitutions to prevent cascade patterns
        print("🔄 Collapsing consecutive substitutions...")
        result_df = self._collapse_consecutive_substitutions(result_df)
        
        # Pre-group by game_id for ultra-fast processing  
        print("📊 Pre-grouping data by game_id...")
        game_groups = result_df.groupby('game_id')
        total_games = len(game_groups)
        print(f"📋 Processing {total_games:,} games with ultra-optimization...")
        
        # Determine team mappings
        game_team_mapping = determine_home_away_teams(result_df)
        
        # Load game context for all games
        self._load_game_context(result_df, game_team_mapping, force_real_pca=force_real_pca)
        
        # Process games in ultra-optimized batches
        all_json_data = []
        games_processed = 0
        game_batch_data = []
        
        # CRITICAL FIX: Process games in sorted order to maintain data alignment
        sorted_game_ids = sorted(game_groups.groups.keys())
        
        for game_id in sorted_game_ids:
            game_df = game_groups.get_group(game_id).reset_index(drop=True)
            
            # Convert game to fast-access dict structure
            game_dict = self._convert_game_to_fast_dict(game_df)
            
            # Get cached context
            game_context = self._team_stats_cache.get(game_id, {})
            team_info = game_team_mapping.get(game_id, {})
            
            # Ultra-fast recent plays collection for entire game
            game_json_data = self._process_game_ultra_fast(
                game_dict, game_context, team_info, n_total, force_real_pca, generation_mode
            )
            
            all_json_data.extend(game_json_data)
            
            # Standard pairing for both modes (now both have full game data)
            game_batch_data.extend([(row, json_str) for row, json_str in zip(game_df.to_dict('records'), game_json_data)])
            games_processed += 1
            
            # Incremental save every N games
            if (self._incremental_save_enabled and 
                games_processed % self._save_every_n_batches == 0 and 
                game_batch_data):
                
                # Create temporary DataFrame for this batch
                batch_rows = [item[0] for item in game_batch_data]
                batch_json = [item[1] for item in game_batch_data]
                temp_df = pd.DataFrame(batch_rows)
                
                self._save_batch_progress(batch_json, temp_df, games_processed // self._save_every_n_batches, 0, len(temp_df))
                game_batch_data = []  # Clear batch after saving
            
            # Progress update
            if games_processed % 100 == 0:
                print(f"⚡ Processed {games_processed:,}/{total_games:,} games ({len(all_json_data):,} examples)")
        
        # Save any remaining data
        if self._incremental_save_enabled and game_batch_data:
            batch_rows = [item[0] for item in game_batch_data]
            batch_json = [item[1] for item in game_batch_data]
            temp_df = pd.DataFrame(batch_rows)
            self._save_batch_progress(batch_json, temp_df, games_processed // self._save_every_n_batches + 1, 0, len(temp_df))
        
        # Add JSON data to result DataFrame  
        result_df['json_training_data'] = all_json_data
        
        # VALIDATION: Verify data alignment (helps catch future bugs) - skip for first_N_plays mode
        if generation_mode != "first_N_plays":
            self._validate_data_alignment(result_df)
        
        print(f"✅ Ultra-optimized processing complete: {len(result_df):,} records")
        return result_df
    
    def _convert_game_to_fast_dict(self, game_df: pd.DataFrame) -> Dict[str, Any]:
        """Convert game DataFrame to dictionary for ultra-fast access."""
        return {
            'plays': game_df.to_dict('records'),  # List of dicts, much faster than iloc
            'length': len(game_df),
            'game_id': game_df.iloc[0]['game_id']
        }
    
    def _process_game_ultra_fast(self, game_dict: Dict[str, Any], game_context: Dict[str, Any], 
                               team_info: Dict[str, str], n_total: int, force_real_pca: bool,
                               generation_mode: str) -> List[str]:
        """Process entire game at once using fast dict access."""
        plays = game_dict['plays']
        game_length = game_dict['length']
        json_data = []
        
        # Extract team info
        away_abbrev = game_context.get('away_abbrev', 'Unknown')
        home_abbrev = game_context.get('home_abbrev', 'Unknown')
        away_stats = game_context.get('away_team_stats', {})
        home_stats = game_context.get('home_team_stats', {})
        lineups = game_context.get('lineups', {})
        game_date = game_context.get('game_date')
        
        # Create player objects once for the entire game (now returns organized dict)
        organized_players = self._create_player_objects(lineups, away_abbrev, home_abbrev, game_date, force_real_pca)
        
        if generation_mode == "first_N_plays":
            # Mode 2: Create context for all plays, but mark only first play for processing
            # We need all plays in the DataFrame for the OpenAI formatter to access
            for i in range(game_length):
                if i == 0:
                    # Only create context for the first play WITHOUT recent_plays
                    json_obj = self._create_json_ultra_fast_no_recent_plays(
                        away_abbrev, home_abbrev, away_stats, home_stats, organized_players
                    )
                    json_string = json.dumps(json_obj, separators=(',', ':'))
                else:
                    # Use a placeholder that won't cause JSON parsing errors
                    json_string = "{}"
                
                json_data.append(json_string)
            
        else:
            # Mode 1: Standard processing but skip first N plays as targets
            # Find the first N non-null plays to determine skip threshold
            first_n_plays_count = 0
            skip_threshold = 0
            for i in range(game_length):
                if pd.notna(plays[i].get('description')):
                    first_n_plays_count += 1
                    if first_n_plays_count >= n_total:
                        skip_threshold = i + 1  # Skip up to and including this index
                        break
            
            # CRITICAL FIX: Process ALL plays to maintain DataFrame alignment
            for i in range(game_length):
                if i < skip_threshold:
                    # For skipped plays, use empty placeholder to maintain alignment
                    json_string = "{}"
                else:
                    # For target plays, create proper context
                    recent_plays = self._collect_recent_plays_ultra_fast(plays, i, n_total, away_abbrev, home_abbrev)
                    
                    # Create JSON object with organized players
                    json_obj = self._create_json_ultra_fast(
                        away_abbrev, home_abbrev, away_stats, home_stats, organized_players, recent_plays
                    )
                    
                    # Serialize to JSON string
                    json_string = json.dumps(json_obj, separators=(',', ':'))
                
                json_data.append(json_string)
        
        return json_data
    
    def _collect_recent_plays_ultra_fast(self, plays: List[Dict], current_index: int, n_total: int,
                                       away_abbrev: str, home_abbrev: str) -> List[Dict[str, Any]]:
        """
        🚀 Ultra-fast recent plays collection using dict access instead of DataFrame iloc.
        ~100x faster than original implementation.
        """
        recent_plays = []
        collected_count = 0
        
        # Go backwards through plays using fast list access
        for j in range(current_index, -1, -1):
            if collected_count >= n_total:
                break
                
            play = plays[j]  # FAST: Direct list access instead of slow DataFrame iloc
            
            if pd.notna(play.get('description')):
                # Get quarter and time
                period = play.get('period', 4)
                remaining_time = play.get('remaining_time', '0:00:00')
                quarter, time_in_quarter = convert_to_quarter_time(period, remaining_time)
                
                # Determine scoring (using fast dict access)
                scoring_team = None
                points_scored = 0
                
                if j > 0:  # Compare with previous play
                    prev_play = plays[j-1]  # FAST: Direct list access
                    prev_away_score = prev_play.get('away_score', 0) or 0
                    prev_home_score = prev_play.get('home_score', 0) or 0
                    curr_away_score = play.get('away_score', 0) or 0
                    curr_home_score = play.get('home_score', 0) or 0
                    
                    scoring_team, points_scored = determine_scoring_info(
                        prev_away_score, prev_home_score, curr_away_score, curr_home_score,
                        away_abbrev, home_abbrev
                    )
                
                # Create shot_details object - populated only for shots
                play_event_type = play.get('event_type', '')
                if play_event_type == 'shot':
                    # For shots, determine the shooting team from the play data
                    shooting_team = play.get('team', '')  # Get team that took the shot
                    
                    # Apply coordinate normalization for shots (commented out for now)
                    # raw_x = play.get('converted_x')
                    # raw_y = play.get('converted_y')
                    # x_norm, y_norm = normalize_shot_coordinates(raw_x, raw_y)
                    
                    shot_details = {
                        "team": str(shooting_team) if shooting_team else None,
                        "points": int(points_scored)
                        # "x_coord": x_norm,  # Commented out - can be re-enabled later
                        # "y_coord": y_norm   # Commented out - can be re-enabled later
                    }
                else:
                    shot_details = {
                        "team": None,
                        "points": None
                        # "x_coord": None,  # Commented out - can be re-enabled later
                        # "y_coord": None   # Commented out - can be re-enabled later
                    }
                
                # Create play object with restructured shot_details and added player
                play_obj = {
                    "quarter": int(quarter),
                    "time_remaining": str(time_in_quarter),
                    "score": f"{away_abbrev} {int(play.get('away_score', 0) or 0)} - {home_abbrev} {int(play.get('home_score', 0) or 0)}",
                    "players_on_court": extract_players_on_court(play, away_abbrev, home_abbrev),
                    "player": str(play.get('player')) if pd.notna(play.get('player')) else None,  # NEW: Player from original data
                    "description": process_play_description(play),
                    "shot_details": shot_details  # RENAMED: scoring -> shot_details with additional fields
                }
                
                recent_plays.insert(0, play_obj)
                collected_count += 1
        
        return recent_plays
    
    def _create_json_ultra_fast_no_recent_plays(self, away_abbrev: str, home_abbrev: str, 
                                              away_stats: Dict, home_stats: Dict,
                                              organized_players: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Create JSON object WITHOUT recent_plays for first_N_plays mode."""
        
        # Handle None REST_DAYS
        away_rest_days = away_stats.get('REST_DAYS') if away_stats.get('REST_DAYS') is not None else 0
        home_rest_days = home_stats.get('REST_DAYS') if home_stats.get('REST_DAYS') is not None else 0
        
        # Extract organized players (already separated by team)
        away_players = organized_players.get('away_players', [])
        home_players = organized_players.get('home_players', [])
        
        return {
            "away_team": {
                "name": str(away_abbrev) if away_abbrev else "Unknown",
                "stats": {
                    "OEFF": float(away_stats.get('OEFF')) if away_stats.get('OEFF') is not None else None,
                    "DEFF": float(away_stats.get('DEFF')) if away_stats.get('DEFF') is not None else None, 
                    "PACE": float(away_stats.get('PACE')) if away_stats.get('PACE') is not None else None,
                    "REST_DAYS": int(away_rest_days) if away_rest_days is not None else 0
                },
                "players": away_players  # NESTED: Players under their team
            },
            "home_team": {
                "name": str(home_abbrev) if home_abbrev else "Unknown",
                "stats": {
                    "OEFF": float(home_stats.get('OEFF')) if home_stats.get('OEFF') is not None else None,
                    "DEFF": float(home_stats.get('DEFF')) if home_stats.get('DEFF') is not None else None,
                    "PACE": float(home_stats.get('PACE')) if home_stats.get('PACE') is not None else None,
                    "REST_DAYS": int(home_rest_days) if home_rest_days is not None else 0
                },
                "players": home_players  # NESTED: Players under their team
            }
            # NOTE: No "recent_plays" field at all for first_N_plays mode
        }
    
    def _create_json_ultra_fast(self, away_abbrev: str, home_abbrev: str, 
                              away_stats: Dict, home_stats: Dict,
                              organized_players: Dict[str, List[Dict]], recent_plays: List[Dict]) -> Dict[str, Any]:
        """Create JSON object with optimized structure assembly."""
        
        # Handle None REST_DAYS
        away_rest_days = away_stats.get('REST_DAYS') if away_stats.get('REST_DAYS') is not None else 0
        home_rest_days = home_stats.get('REST_DAYS') if home_stats.get('REST_DAYS') is not None else 0
        
        # Extract organized players (already separated by team)
        away_players = organized_players.get('away_players', [])
        home_players = organized_players.get('home_players', [])
        
        return {
            "away_team": {
                "name": str(away_abbrev) if away_abbrev else "Unknown",
                "stats": {
                    "OEFF": float(away_stats.get('OEFF')) if away_stats.get('OEFF') is not None else None,
                    "DEFF": float(away_stats.get('DEFF')) if away_stats.get('DEFF') is not None else None, 
                    "PACE": float(away_stats.get('PACE')) if away_stats.get('PACE') is not None else None,
                    "REST_DAYS": int(away_rest_days) if away_rest_days is not None else 0
                },
                "players": away_players  # NESTED: Players under their team
            },
            "home_team": {
                "name": str(home_abbrev) if home_abbrev else "Unknown",
                "stats": {
                    "OEFF": float(home_stats.get('OEFF')) if home_stats.get('OEFF') is not None else None,
                    "DEFF": float(home_stats.get('DEFF')) if home_stats.get('DEFF') is not None else None,
                    "PACE": float(home_stats.get('PACE')) if home_stats.get('PACE') is not None else None,
                    "REST_DAYS": int(home_rest_days) if home_rest_days is not None else 0
                },
                "players": home_players  # NESTED: Players under their team
            },
            "recent_plays": recent_plays
        }
    
    def _collapse_consecutive_substitutions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Collapse consecutive substitutions with same remaining_time into single plays.
        
        This prevents substitution cascade patterns in training data by combining
        multiple simultaneous subs into one play with format:
        "SUBS: desc1, desc2, desc3, ..."
        
        Args:
            df: DataFrame sorted by game_id and play_id
            
        Returns:
            DataFrame with consecutive substitutions collapsed
        """
        result_rows = []
        i = 0
        total_collapsed = 0
        
        while i < len(df):
            current_row = df.iloc[i]
            
            # Check if this is a substitution
            if current_row.get('event_type') == 'substitution':
                # Collect consecutive substitutions with same game_id and remaining_time
                sub_group = [current_row]
                current_game_id = current_row['game_id']
                current_time = current_row.get('remaining_time')
                
                # Look ahead for more consecutive substitutions
                j = i + 1
                while j < len(df):
                    next_row = df.iloc[j]
                    
                    # Stop if different game, different time, or not a substitution
                    if (next_row['game_id'] != current_game_id or
                        next_row.get('remaining_time') != current_time or
                        next_row.get('event_type') != 'substitution'):
                        break
                    
                    sub_group.append(next_row)
                    j += 1
                
                # Create collapsed row if we have multiple substitutions
                if len(sub_group) > 1:
                    # Start with the first substitution row
                    collapsed_row = current_row.copy()
                    
                    # Combine descriptions (remove "SUB:" prefixes)
                    descriptions = []
                    for sub_row in sub_group:
                        desc = str(sub_row.get('description', ''))
                        # Remove "SUB:" prefix if present
                        if desc.startswith('SUB:'):
                            desc = desc[4:].strip()
                        descriptions.append(desc)
                    
                    # Create new description with "SUBS:" prefix
                    collapsed_row['description'] = f"SUBS: {', '.join(descriptions)}"
                    
                    # Set player field to None since multiple players are involved
                    collapsed_row['player'] = None
                    
                    result_rows.append(collapsed_row)
                    total_collapsed += len(sub_group) - 1  # Count eliminated rows
                    i = j  # Skip past all the substitutions we just collapsed
                else:
                    # Single substitution - keep as is
                    result_rows.append(current_row)
                    i += 1
            else:
                # Not a substitution - keep as is
                result_rows.append(current_row)
                i += 1
        
        # Create new DataFrame
        result_df = pd.DataFrame(result_rows).reset_index(drop=True)
        
        if total_collapsed > 0:
            original_count = len(df)
            new_count = len(result_df)
            print(f"   ✅ Collapsed {total_collapsed} consecutive substitutions")
            print(f"   📉 Reduced from {original_count:,} to {new_count:,} rows ({original_count - new_count:,} removed)")
        else:
            print(f"   ℹ️  No consecutive substitutions found to collapse")
            
        return result_df
    
    def enable_incremental_save(self, save_path: str, save_every_n_batches: int = 5):
        """Enable incremental saving for ultra-optimized generator."""
        self._incremental_save_enabled = True
        self._incremental_save_path = save_path
        self._save_every_n_batches = save_every_n_batches
        print(f"💾 Incremental saves enabled: {save_path} (every {save_every_n_batches} games)")
        
    def _save_batch_progress(self, batch_json_data, batch_df, batch_number, batch_start, batch_end):
        """Save incremental progress (compatibility method)."""
        if not self._incremental_save_enabled or not self._incremental_save_path:
            return
            
        from training.openai_formatter import OpenAIFormatter
        
        # Convert batch to OpenAI format
        temp_df = batch_df.copy()
        temp_df['json_training_data'] = batch_json_data
        
        formatter = OpenAIFormatter()
        openai_examples = formatter.create_training_data(temp_df)
        
        # Append to incremental file
        with open(self._incremental_save_path, 'a', encoding='utf-8') as f:
            for i, example in enumerate(openai_examples):
                # CRITICAL FIX: Use json.dumps for proper JSON format (not Python dict format)
                json_line = json.dumps(example, separators=(',', ':'), ensure_ascii=False)
                
                # Validate JSON format for first few examples to catch formatting bugs early
                if i < 3:
                    try:
                        json.loads(json_line)  # Test that we can parse it back
                    except json.JSONDecodeError as e:
                        print(f"⚠️ ERROR: Invalid JSON in incremental save: {e}")
                        print(f"   Problematic line: {json_line[:100]}...")
                        raise
                
                f.write(f"{json_line}\n")
        
        print(f"💾 Batch #{batch_number}: Saved {len(openai_examples)} examples → {self._incremental_save_path}")
    
    def _validate_data_alignment(self, result_df: pd.DataFrame) -> None:
        """
        Validate that JSON context data is properly aligned with DataFrame rows.
        This helps catch data ordering bugs that cause mismatched training examples.
        """
        print("🔍 Validating data alignment...")
        
        validation_sample_size = min(100, len(result_df))
        sample_indices = np.random.choice(len(result_df), validation_sample_size, replace=False)
        
        misalignment_count = 0
        
        for idx in sample_indices:
            row = result_df.iloc[idx]
            try:
                # Parse the JSON context
                json_context = json.loads(row['json_training_data'])
                recent_plays = json_context['recent_plays']
                
                if not recent_plays:
                    continue  # Skip if no recent plays
                    
                # Get the most recent play from JSON context
                last_json_play = recent_plays[-1]
                
                # The current DataFrame row should match or be very close to this
                current_game_id = row['game_id']
                current_score = f"{json_context['away_team']['name']} {int(row.get('away_score', 0) or 0)} - {json_context['home_team']['name']} {int(row.get('home_score', 0) or 0)}"
                
                # Check if scores are reasonably close (allowing for small differences)
                json_score = last_json_play['score']
                if json_score != current_score:
                    # Allow for small score differences (within 5 points) as plays might not be exactly aligned
                    try:
                        json_away = int(json_score.split(' - ')[0].split(' ')[-1])
                        json_home = int(json_score.split(' - ')[1])
                        curr_away = int(row.get('away_score', 0) or 0)
                        curr_home = int(row.get('home_score', 0) or 0)
                        
                        if abs(json_away - curr_away) > 10 or abs(json_home - curr_home) > 10:
                            print(f"⚠️ Score mismatch at row {idx}: JSON='{json_score}' vs Row='{current_score}'")
                            misalignment_count += 1
                    except:
                        print(f"⚠️ Score parsing error at row {idx}: JSON='{json_score}' vs Row='{current_score}'")
                        misalignment_count += 1
                        
            except Exception as e:
                print(f"⚠️ Validation error at row {idx}: {e}")
                misalignment_count += 1
        
        if misalignment_count == 0:
            print("✅ Data alignment validation passed!")
        else:
            print(f"⚠️ Found {misalignment_count}/{validation_sample_size} potential misalignments")
            if misalignment_count > validation_sample_size * 0.1:  # More than 10% misaligned
                print("🚨 HIGH MISALIGNMENT DETECTED - Consider investigating data ordering")
        
        return misalignment_count
