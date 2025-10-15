"""
Google Gemini training data formatting utilities.

This module handles conversion of NBA training data to the official Google Cloud 
Vertex AI Gemini fine-tuning format using the "GenerateContent" structure with user/model roles.

Format: {"contents": [{"role": "user", "parts": [{"text": "..."}]}, {"role": "model", "parts": [{"text": "..."}]}]}
"""

import json
from typing import List, Dict, Any, Optional
import pandas as pd
from game.time_utils import convert_to_quarter_time
from game.scoring_utils import determine_scoring_info
from data.file_utils import file_manager
from training.data_generator import extract_players_on_court, process_play_description, normalize_shot_coordinates
from training.base_formatter import BaseFormatter, FormatterFactory
from config.settings import DEFAULT_N_TOTAL_PLAYS


class GeminiFormatter(BaseFormatter):
    """Handles conversion to Google Gemini fine-tuning format."""
    
    @property
    def platform_name(self) -> str:
        """Return the platform name."""
        return "Gemini"
    
    def _is_compact_format(self, json_data: Dict[str, Any]) -> bool:
        """Check if the JSON data is in compact format."""
        if not isinstance(json_data, dict):
            return False
        # Quick check for compact vs verbose format
        return ('A' in json_data and 'H' in json_data and 
                'away_team' not in json_data and 'home_team' not in json_data)
    
    def _convert_compact_to_verbose_for_gemini(self, compact_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert compact format to verbose format for Gemini formatter processing.
        This is a simplified conversion focused on what the formatter needs.
        """
        try:
            away_abbrev = compact_data.get('a', 'AWAY')
            home_abbrev = compact_data.get('h', 'HOME')
            
            # Convert team stats
            away_stats_array = compact_data.get('as', [110.0, 110.0, 100.0, 2])
            home_stats_array = compact_data.get('hs', [110.0, 110.0, 100.0, 2])
            
            # Convert plays to verbose format
            plays_array = compact_data.get('p', [])
            recent_plays = []
            
            for play_tuple in plays_array:
                if len(play_tuple) >= 6:
                    quarter = play_tuple[0]
                    time_seconds = play_tuple[1]
                    score_array = play_tuple[2]
                    actor = play_tuple[3]
                    event_code = play_tuple[4]
                    
                    # Handle both scoring and non-scoring formats
                    if len(play_tuple) == 7:
                        points = play_tuple[5]
                        lineup_id = play_tuple[6]
                    else:
                        points = None
                        lineup_id = play_tuple[5]
                    
                    # Convert time back to MM:SS
                    minutes = time_seconds // 60
                    seconds = time_seconds % 60
                    time_remaining = f"{minutes:02d}:{seconds:02d}"
                    
                    # Build score string
                    score_str = f"{away_abbrev} {score_array[0]} - {home_abbrev} {score_array[1]}"
                    
                    # Create verbose play object
                    play = {
                        'quarter': quarter,
                        'time_remaining': time_remaining,
                        'description': f"Play: {event_code}",
                        'score': score_str,
                        'shot_details': {
                            'team': away_abbrev if actor[0] == 'A' else home_abbrev,
                            'points': points
                        } if points else {'team': None, 'points': None}
                    }
                    recent_plays.append(play)
            
            # Build verbose format
            verbose_data = {
                'away_team': {
                    'name': away_abbrev,
                    'stats': {
                        'OEFF': away_stats_array[0],
                        'DEFF': away_stats_array[1],
                        'PACE': away_stats_array[2],
                        'REST_DAYS': away_stats_array[3]
                    }
                },
                'home_team': {
                    'name': home_abbrev,
                    'stats': {
                        'OEFF': home_stats_array[0],
                        'DEFF': home_stats_array[1],
                        'PACE': home_stats_array[2],
                        'REST_DAYS': home_stats_array[3]
                    }
                },
                'recent_plays': recent_plays
            }
            
            return verbose_data
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to convert compact to verbose for Gemini: {e}")
            return compact_data  # Return original if conversion fails
    
    def create_training_data(self, df: pd.DataFrame, generation_mode: str = "remaining_plays", n_total: int = None, season: str = None) -> List[Dict[str, Any]]:
        """
        Convert our JSON training data into Gemini fine-tuning JSONL format.
        Handles two generation modes:
        - remaining_plays: Skip first N plays, then create input-output pairs with full context
        - first_N_plays: Single entry per game with first N plays as targets
        
        Args:
            df: DataFrame with 'json_training_data' column
            generation_mode: "remaining_plays" or "first_N_plays"
            
        Returns:
            list: List of training examples in Gemini format
        """
        training_examples = []
        
        if generation_mode == "first_N_plays":
            # Mode 2: Single entry per game with first N plays as targets
            from generate_training_data import load_play_by_play_data
            
            # Determine season from data or parameter
            if season is None:
                # Try to detect season from game IDs
                if len(df) > 0:
                    sample_game_id = str(df['game_id'].iloc[0])
                    if sample_game_id.startswith('222'):
                        season = '2022-2023'
                    elif sample_game_id.startswith('223'):
                        season = '2023-2024'
                    elif sample_game_id.startswith('224'):
                        season = '2024-2025'
                    else:
                        season = '2023-2024'  # Default fallback
                else:
                    season = '2023-2024'  # Default fallback
            
            # Load original play-by-play data for complete game information
            raw_df = load_play_by_play_data(season)
            
            for game_id in sorted(df['game_id'].unique()):
                # Get filtered training data for context
                game_df = df[df['game_id'] == game_id].sort_values('play_id').reset_index(drop=True)
                # Get full raw game data for play generation
                raw_game_df = raw_df[raw_df['game_id'] == game_id].sort_values('play_id').reset_index(drop=True)
                
                # Find the first row with valid JSON context (should be the first row)
                context_row = None
                for i in range(len(game_df)):
                    json_data = game_df.iloc[i]['json_training_data']
                    if json_data and json_data.strip() and json_data.strip() != "{}":
                        try:
                            # Test if it's valid JSON and has team data (verbose or compact)
                            parsed = json.loads(json_data)
                            if 'away_team' in parsed and 'home_team' in parsed:
                                context_row = game_df.iloc[i]
                                break
                            elif self._is_compact_format(parsed):
                                context_row = game_df.iloc[i]
                                break
                        except json.JSONDecodeError:
                            continue
                
                if context_row is not None and len(raw_game_df) > 0:
                    example = self._create_first_n_plays_example(raw_game_df, context_row, n_total)
                    if example:
                        training_examples.append(example)
        else:
            # Mode 1: Standard remaining_plays pairs
            # Load raw data to ensure proper sequential play selection
            from generate_training_data import load_play_by_play_data
            
            # Determine season from data or parameter (same logic as first_N_plays)
            if season is None:
                if len(df) > 0:
                    sample_game_id = str(df['game_id'].iloc[0])
                    # Support both 8-char (e.g., 22200001) and 10-char (e.g., 0022200001) formats
                    season_code = None
                    if len(sample_game_id) >= 3:
                        # 8-char form: first 3 are code (e.g., 222, 223, 224)
                        season_code = sample_game_id[:3]
                    if len(sample_game_id) >= 5 and sample_game_id.startswith('00'):
                        # 10-char form: characters 2:5 form the code (e.g., 0022200001 -> 222)
                        season_code = sample_game_id[2:5]
                    
                    if season_code == '021':
                        season = '2020-2021'
                    elif season_code == '022':
                        season = '2021-2022'
                    elif season_code == '222':
                        season = '2022-2023'
                    elif season_code == '223':
                        season = '2023-2024'
                    elif season_code == '224':
                        season = '2024-2025'
                    else:
                        season = '2023-2024'  # Default fallback
                else:
                    season = '2023-2024'  # Default fallback
            
            raw_df = load_play_by_play_data(season)
            
            for game_id in sorted(df['game_id'].unique()):
                game_df = df[df['game_id'] == game_id].sort_values('play_id').reset_index(drop=True)
                raw_game_df = raw_df[raw_df['game_id'] == game_id].sort_values('play_id').reset_index(drop=True)
                
                # Create training examples for consecutive plays  
                for i in range(len(game_df) - 1):  # -1 because we need a next play
                    # Skip rows with invalid JSON data (placeholders from first N plays)
                    current_json_data = game_df.iloc[i]['json_training_data']
                    if not current_json_data or current_json_data.strip() == "" or current_json_data.strip() == "{}":
                        continue
                    
                    example = self._create_training_example(game_df, i, raw_game_df)
                    if example:
                        training_examples.append(example)
        
        return training_examples
    
    def _create_verbose_play_object(self, row: pd.Series, context_json: Dict[str, Any], game_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Create a verbose play object from a DataFrame row.
        
        Args:
            row: DataFrame row containing play data
            context_json: Game context in verbose format
            game_df: Full game DataFrame
            
        Returns:
            dict: Verbose play object or None if creation failed
        """
        try:
            import pandas as pd
            
            # Get basic play info
            description = row.get('description', '')
            if pd.isna(description):
                return None
            
            # Extract score
            away_score = row.get('away_score', 0) or 0
            home_score = row.get('home_score', 0) or 0
            away_name = context_json.get('away_team', {}).get('name', 'AWAY')
            home_name = context_json.get('home_team', {}).get('name', 'HOME')
            
            # Extract lineup information
            away_lineup = []
            home_lineup = []
            for i in range(1, 6):
                away_player = row.get(f'a{i}')
                if pd.notna(away_player):
                    away_lineup.append(str(away_player))
                home_player = row.get(f'h{i}')
                if pd.notna(home_player):
                    home_lineup.append(str(home_player))
                    
            players_on_court = [
                {'team': away_name, 'players': away_lineup},
                {'team': home_name, 'players': home_lineup}
            ]
            
            # Extract other fields with proper null handling
            quarter = int(row.get('quarter', 1))
            time_remaining = str(row.get('time_remaining', '12:00'))
            player = str(row.get('player', '')) if pd.notna(row.get('player')) else None
            event_type = str(row.get('event_type', '')) if pd.notna(row.get('event_type')) else None
            play_type = str(row.get('type', '')) if pd.notna(row.get('type')) else None
            
            # Handle result and points with proper null handling
            result = row.get('result')
            if pd.isna(result):
                result = None
            else:
                result = str(result)
                
            points = row.get('points')
            if pd.isna(points):
                points = None
            else:
                points = float(points)
                
            shot_distance = row.get('shot_distance')
            if pd.isna(shot_distance):
                shot_distance = None
            else:
                shot_distance = float(shot_distance)
            
            # Build shot_details
            shot_details = {
                'team': None,
                'points': points
            }
            
            # Create verbose play object
            play_obj = {
                "quarter": quarter,
                "time_remaining": time_remaining,
                "description": description,
                "score": f"{int(away_score)} - {int(home_score)}",
                "player": player,
                "players_on_court": players_on_court,
                "type": play_type,
                "event_type": event_type,
                "result": result,
                "points": points,
                "shot_distance": shot_distance,
                "shot_details": shot_details
            }
            
            return play_obj
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to create verbose play object: {e}")
            return None
    
    def _create_compact_play_tuple(self, row: pd.Series, context_json: Dict[str, Any], game_df: pd.DataFrame) -> Optional[List]:
        """
        Create a compact play tuple from a DataFrame row for first_N_plays mode.
        
        Args:
            row: DataFrame row with play data
            context_json: Current game context for team names and player lookups
            game_df: Full game DataFrame for scoring calculations
            
        Returns:
            list: Compact play tuple [quarter, time_seconds, score_array, actor, event_code, points?, lineup_id?]
        """
        try:
            # Get quarter and time
            quarter, time_in_quarter = convert_to_quarter_time(
                row['period'], row['remaining_time']
            )
            
            # Convert time to seconds using same logic as parse_time_to_seconds
            time_parts = str(time_in_quarter).split(':')
            if len(time_parts) >= 2:
                minutes = int(time_parts[-2])  # Second to last part (minutes)
                seconds = int(time_parts[-1])  # Last part (seconds) 
                time_seconds = 60 * minutes + seconds
            else:
                time_seconds = 720  # Default 12:00
            
            # Create score array
            away_score = int(row.get('away_score', 0) or 0)
            home_score = int(row.get('home_score', 0) or 0)
            score_array = [away_score, home_score]
            
            # Determine team names based on format
            if self._is_compact_format(context_json):
                away_team_name = context_json['A']
                home_team_name = context_json['H']
            else:
                away_team_name = context_json['away_team']['name']
                home_team_name = context_json['home_team']['name']
            
            # Create actor - simplified for first_N_plays
            player_name = row.get('player')
            team_name = row.get('team', '')
            
            if team_name == away_team_name or (not team_name and player_name):
                actor = ["A", -1]  # Away team, simplified index
            elif team_name == home_team_name:
                actor = ["H", -1]  # Home team, simplified index  
            else:
                actor = ["A", -1]  # Default to away team
            
            # Use structured event code mapping for better accuracy
            from generate_training_data import map_structured_to_event_code, map_description_to_event_code
            
            # Check if structured data is available (type, event_type fields)
            if all(key in row for key in ['type', 'event_type']):
                # Use structured mapping when available (much more accurate)
                event_code, mapped_points = map_structured_to_event_code(row)
            else:
                # Fall back to description parsing
                description = row.get('description', '')
                
                # Calculate score delta for proper event mapping
                prev_away_score, prev_home_score = 0, 0  # Simplified for first plays
                if game_df is not None:
                    current_row_index = None
                    for pos_idx in range(len(game_df)):
                        if (game_df.iloc[pos_idx]['play_id'] == row['play_id'] and 
                            game_df.iloc[pos_idx]['game_id'] == row['game_id']):
                            current_row_index = pos_idx
                            break
                    
                    if current_row_index is not None and current_row_index > 0:
                        prev_row = game_df.iloc[current_row_index - 1]
                        prev_away_score = prev_row.get('away_score', 0) or 0
                        prev_home_score = prev_row.get('home_score', 0) or 0
                
                score_delta = max(0, max(
                    away_score - prev_away_score,  # Away team scored
                    home_score - prev_home_score   # Home team scored
                ))
                
                # Build shot_details for event mapping
                shot_details = {
                    'team': team_name,
                    'points': score_delta if score_delta > 0 else None
                }
                
                # Use the comprehensive mapping function
                event_code, mapped_points = map_description_to_event_code(
                    description, shot_details, score_delta
                )
            
            # Build the compact play tuple
            play_tuple = [
                int(quarter),
                int(time_seconds),
                score_array,
                actor,
                event_code
            ]
            
            # Add points if this is a scoring event
            if mapped_points is not None and mapped_points > 0:
                play_tuple.append(int(mapped_points))
            else:
                play_tuple.append(0)
            
            # Add lineup ID if available (simplified for first_N_plays)
            # For now, omit lineup_id since it's optional and complex to calculate
            
            return play_tuple
            
        except Exception as e:
            # Silent failure - this is expected for some plays with incomplete data
            return None
    
    def _create_training_example(self, game_df: pd.DataFrame, current_index: int, raw_game_df: pd.DataFrame = None) -> Optional[Dict[str, Any]]:
        """
        Create a single Gemini training example from current and next plays.
        
        Args:
            game_df: DataFrame for a single game (filtered training data)
            current_index: Index of current play in filtered data
            raw_game_df: Raw game DataFrame for finding sequential next play
            
        Returns:
            dict or None: Gemini training example or None if creation failed
        """
        try:
            current_row = game_df.iloc[current_index]
            
            # Find the actual next play in sequence using play_id
            current_play_id = current_row['play_id']
            
            if raw_game_df is not None:
                # Find the next sequential play in raw data
                current_raw_index = None
                for idx, row in raw_game_df.iterrows():
                    if row['play_id'] == current_play_id:
                        current_raw_index = idx
                        break
                
                if current_raw_index is not None and current_raw_index < len(raw_game_df) - 1:
                    next_row = raw_game_df.iloc[current_raw_index + 1]
                else:
                    return None  # No next play found
            else:
                # Fallback to old method if raw data not available
                if current_index + 1 >= len(game_df):
                    return None
            next_row = game_df.iloc[current_index + 1]
            
            # Parse the current JSON context and convert if compact
            current_json_raw = json.loads(current_row['json_training_data'])
            if self._is_compact_format(current_json_raw):
                # For Gemini, we need to convert compact to verbose for compatibility with existing logic
                current_json = self._convert_compact_to_verbose_for_gemini(current_json_raw)
            else:
                current_json = current_json_raw
            
            # Create the next play response
            assistant_response = self._create_next_play_response(
                game_df, current_index, next_row, current_json
            )
            
            # Create Gemini training example using GenerateContent format
            # OLD (messages format - commented out for easy revert):
            # training_example = {
            #     "messages": [
            #         {
            #             "role": "user",
            #             "content": current_row['json_training_data']  # Our JSON context
            #         },
            #         {
            #             "role": "model",
            #             "content": json.dumps(assistant_response, separators=(',', ':'))
            #         }
            #     ]
            # }
            
            # NEW (GenerateContent format):
            training_example = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": current_row['json_training_data']}]  # Our JSON context
                    },
                    {
                        "role": "model", 
                        "parts": [{"text": json.dumps(assistant_response, separators=(',', ':'))}]
                    }
                ]
            }
            
            return training_example
            
        except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
            print(f"Warning: Skipped Gemini training example due to error: {e}")
            return None
    
    def _create_first_n_plays_example(self, game_df: pd.DataFrame, context_row: pd.Series, n_total: int = None) -> Optional[Dict[str, Any]]:
        """
        Create a single first-N-plays Gemini training example.
        
        Args:
            game_df: DataFrame for a single game
            context_row: Row containing the game context without recent_plays
            
        Returns:
            dict or None: Gemini training example or None if creation failed
        """
        try:
            # Parse the context JSON
            context_json = json.loads(context_row['json_training_data'])
            
            # Get first N non-null plays from the raw DataFrame data
            first_plays = []
            if n_total is None:
                n_total = DEFAULT_N_TOTAL_PLAYS  # Fallback to default if not specified
            
            for i in range(len(game_df)):
                if len(first_plays) >= n_total:
                    break
                    
                row = game_df.iloc[i]
                if pd.notna(row.get('description')):
                    # Check if input is verbose format - if so, create verbose response
                    if self._is_compact_format(context_json):
                        # Create compact play tuple for compact input
                        play_tuple = self._create_compact_play_tuple(row, context_json, game_df)
                        if play_tuple:
                            first_plays.append(play_tuple)
                    else:
                        # Create verbose play object for verbose input
                        play_obj = self._create_verbose_play_object(row, context_json, game_df)
                        if play_obj:
                            first_plays.append(play_obj)
            
            # Return raw array for maximum efficiency  
            assistant_response = first_plays
            
            # Create Gemini training example using GenerateContent format
            # OLD (messages format - commented out for easy revert):
            # training_example = {
            #     "messages": [
            #         {
            #             "role": "user",
            #             "content": context_row['json_training_data']  # Context without recent_plays
            #         },
            #         {
            #             "role": "model",
            #             "content": json.dumps(assistant_response, separators=(',', ':'))
            #         }
            #     ]
            # }
            
            # NEW (GenerateContent format):
            training_example = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": context_row['json_training_data']}]  # Context without recent_plays
                    },
                    {
                        "role": "model",
                        "parts": [{"text": json.dumps(assistant_response, separators=(',', ':'))}]
                    }
                ]
            }
            
            return training_example
            
        except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
            print(f"Warning: Skipped first_N_plays Gemini training example due to error: {e}")
            return None
    
    def _create_play_object(self, row: pd.Series, current_json: Dict[str, Any], game_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create a play object from a DataFrame row.
        
        Args:
            row: DataFrame row with play data
            current_json: Current game context for team names
            game_df: Full game DataFrame for scoring calculations
            
        Returns:
            dict: Play object in the expected format
        """
        # Get quarter and time
        quarter, time_in_quarter = convert_to_quarter_time(
            row['period'], row['remaining_time']
        )
        
        # Calculate scoring by comparing with previous play in the game
        scoring_team = None
        points_scored = 0
        
        # Determine team names based on format (needed for score formatting)
        if self._is_compact_format(current_json):
            away_team_name = current_json['A']
            home_team_name = current_json['H']
        else:
            away_team_name = current_json['away_team']['name']
            home_team_name = current_json['home_team']['name']
        
        # Find the current row index in the game DataFrame using positional index
        current_row_index = None
        for pos_idx in range(len(game_df)):
            game_row = game_df.iloc[pos_idx]
            if (game_row['play_id'] == row['play_id'] and 
                game_row['game_id'] == row['game_id']):
                current_row_index = pos_idx
                break
        
        # If we found the current row and it's not the first row, calculate scoring
        if current_row_index is not None and current_row_index > 0:
            prev_row = game_df.iloc[current_row_index - 1]
            prev_away_score = prev_row.get('away_score', 0) or 0
            prev_home_score = prev_row.get('home_score', 0) or 0
            curr_away_score = row.get('away_score', 0) or 0
            curr_home_score = row.get('home_score', 0) or 0
            
            scoring_team, points_scored = determine_scoring_info(
                prev_away_score, prev_home_score,
                curr_away_score, curr_home_score,
                away_team_name, home_team_name
            )
        
        # Format score (use the already determined team names)
        # away_team_name and home_team_name are set above based on format
        score = f"{away_team_name} {int(row.get('away_score', 0) or 0)} - {home_team_name} {int(row.get('home_score', 0) or 0)}"
        
        # Get player
        player = row.get('player')
        
        # Create shot_details object - populated only for shots
        event_type = row.get('event_type', '')
        if event_type == 'shot':
            # For shots, determine the shooting team from the row data
            shooting_team = row.get('team', '')  # Get team that took the shot
            
            # Apply coordinate normalization for shots (commented out for now)
            # raw_x = row.get('converted_x')
            # raw_y = row.get('converted_y')
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
        
        return {
            "quarter": int(quarter),
            "time_remaining": str(time_in_quarter),
            "score": score,
            "players_on_court": extract_players_on_court(row, away_team_name, home_team_name),
            "player": str(player) if pd.notna(player) else None,
            "description": process_play_description(row),
            "shot_details": shot_details  # Use shot_details format consistent with other formatters
        }
    
    def _create_next_play_response(self, game_df: pd.DataFrame, current_index: int, 
                                 next_row: pd.Series, current_json: Dict[str, Any]) -> List[Any]:
        """
        Create the assistant response (next play prediction) for Gemini training.
        
        Args:
            game_df: DataFrame for the game
            current_index: Index of current play
            next_row: Next play data
            current_json: Parsed JSON context from current play
            
        Returns:
            list: Raw compact play tuple for maximum efficiency
        """
        # Get quarter and time for next play
        next_quarter, next_time = convert_to_quarter_time(
            next_row['period'], next_row['remaining_time']
        )
        
        # Determine team names based on format
        if self._is_compact_format(current_json):
            away_team_name = current_json['A']
            home_team_name = current_json['H']
        else:
            away_team_name = current_json['away_team']['name']
            home_team_name = current_json['home_team']['name']
        
        # Determine scoring info for next play
        prev_away_score, prev_home_score = self._get_previous_scores(game_df, current_index)
        
        scoring_team, points_scored = determine_scoring_info(
            prev_away_score, prev_home_score,
            next_row['away_score'], next_row['home_score'], 
            away_team_name, home_team_name
        )
        
        # VALIDATION: Check for obvious mismatches (steal/turnover/miss with scoring)
        next_desc = str(next_row['description']).upper()
        is_non_scoring_play = any(word in next_desc for word in [
            'STEAL', 'TURNOVER', 'MISS', 'REBOUND', 'FOUL', 'SUB:', 'TIMEOUT'
        ])
        
        if is_non_scoring_play and points_scored > 0:
            # Silent correction for obvious non-scoring plays
            # (Data gaps in play-by-play are common and expected)
            scoring_team, points_scored = None, 0
        
        # Format score
        score = (f"{away_team_name} {next_row['away_score']} - "
                f"{home_team_name} {next_row['home_score']}")
        
        # Create shot_details object - populated only for shots
        next_event_type = next_row.get('event_type', '')
        if next_event_type == 'shot':
            # For shots, determine the shooting team from the row data
            shooting_team = next_row.get('team', '')  # Get team that took the shot
            
            # Apply coordinate normalization for shots (commented out for now)
            # raw_x = next_row.get('converted_x')
            # raw_y = next_row.get('converted_y')
            # x_norm, y_norm = normalize_shot_coordinates(raw_x, raw_y)
            
            shot_details = {
                "team": str(shooting_team) if shooting_team else None,
                "points": int(points_scored) if points_scored else 0
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
        
        # Create compact format assistant response for Gemini
        next_player = next_row.get('player')
        
        # Convert time to seconds (next_time is already in MM:SS format)
        time_parts = str(next_time).split(':')
        if len(time_parts) >= 2:
            minutes = int(time_parts[-2])  # Second to last part (minutes)
            seconds = int(time_parts[-1])  # Last part (seconds) 
            time_seconds = 60 * minutes + seconds
        else:
            time_seconds = 720  # Default 12:00
        
        # Create score array
        next_away_score = int(next_row.get('away_score', 0) or 0)
        next_home_score = int(next_row.get('home_score', 0) or 0)
        score_array = [next_away_score, next_home_score]
        
        # Create actor
        away_team = away_team_name
        home_team = home_team_name
        
        # Simple actor mapping based on team (simplified for formatter)
        if scoring_team == away_team:
            actor = ["A", 0]  # Away team, simplified index
        elif scoring_team == home_team:
            actor = ["H", 0]  # Home team, simplified index
        else:
            actor = ["A", -1]  # Default to away team event
        
        # Use structured event code mapping for better accuracy
        from generate_training_data import map_structured_to_event_code, map_description_to_event_code
        
        # Check if structured data is available (type, event_type fields)
        if all(key in next_row for key in ['type', 'event_type']):
            # Use structured mapping when available (much more accurate)
            event_code, mapped_points = map_structured_to_event_code(next_row)
            
            # Use mapped points if available, otherwise use calculated points
            if mapped_points is not None:
                points_scored = mapped_points
        else:
            # Fall back to description parsing
            description = next_row.get('description', '')
            
            # Calculate score delta for proper event mapping
            prev_away_score, prev_home_score = self._get_previous_scores(game_df, current_index)
            score_delta = max(0, max(
                next_away_score - prev_away_score,  # Away team scored
                next_home_score - prev_home_score   # Home team scored  
            ))
            
            # Build shot_details for event mapping
            shot_details = {
                'team': scoring_team,
                'points': points_scored if points_scored > 0 else None
            }
            
            # Use the comprehensive mapping function
            event_code, mapped_points = map_description_to_event_code(
                description, shot_details, score_delta
            )
            
            # Use mapped points if available, otherwise use calculated points
            if mapped_points is not None:
                points_scored = mapped_points
        
        # Default lineup ID
        lineup_id = 0
        
        # Check if input context is in verbose format and respond accordingly
        if self._is_compact_format(current_json):
            # Create compact play tuple for compact input
            # Always standardize to 7 elements; non-scoring uses points=0
            pts_val = int(points_scored) if points_scored and points_scored > 0 else 0
            play_tuple = [
                int(next_quarter), 
                int(time_seconds), 
                [int(score_array[0]), int(score_array[1])], 
                actor, 
                event_code, 
                pts_val, 
                int(lineup_id)
            ]
            return play_tuple
        else:
            # Create verbose play object for verbose input
            import pandas as pd
            
            # Extract lineup information from next_row
            away_lineup = []
            home_lineup = []
            for i in range(1, 6):
                away_player = next_row.get(f'a{i}')
                if pd.notna(away_player):
                    away_lineup.append(str(away_player))
                home_player = next_row.get(f'h{i}')
                if pd.notna(home_player):
                    home_lineup.append(str(home_player))
                    
            players_on_court = [
                {'team': away_team_name, 'players': away_lineup},
                {'team': home_team_name, 'players': home_lineup}
            ]
            
            # Create verbose play object
            verbose_response = {
                "quarter": int(next_quarter),
                "time_remaining": str(next_time),
                "description": str(next_row.get('description', '')),
                "score": f"{int(score_array[0])} - {int(score_array[1])}",
                "player": str(next_player) if pd.notna(next_player) else None,
                "players_on_court": players_on_court,
                "type": str(next_row.get('type')) if pd.notna(next_row.get('type')) else None,
                "event_type": str(next_row.get('event_type')) if pd.notna(next_row.get('event_type')) else None,
                "result": str(next_row.get('result')) if pd.notna(next_row.get('result')) else None,
                "points": float(points_scored) if points_scored and points_scored > 0 else None,
                "shot_distance": float(next_row.get('shot_distance')) if pd.notna(next_row.get('shot_distance')) else None,
                "shot_details": shot_details
            }
            
            return verbose_response
    
    def _get_previous_scores(self, game_df: pd.DataFrame, current_index: int) -> tuple[int, int]:
        """
        Get the scores from the current context play for next play scoring calculation.
        
        Args:
            game_df: DataFrame for the game
            current_index: Index of current context play
            
        Returns:
            tuple: (context_away_score, context_home_score)
        """
        current_row = game_df.iloc[current_index]
        return int(current_row.get('away_score', 0) or 0), int(current_row.get('home_score', 0) or 0)
    
    def validate_training_examples(self, training_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate Gemini training examples for format compliance.
        
        Args:
            training_examples: List of training examples to validate
            
        Returns:
            dict: Validation results with errors, warnings, and statistics
        """
        validation_results = {
            'total_examples': len(training_examples),
            'valid_examples': 0,
            'errors': [],
            'warnings': [],
            'error_rate': 0.0,
            'success_rate': 0.0
        }
        
        for i, example in enumerate(training_examples):
            try:
                # Check required fields for Gemini GenerateContent format
                # OLD (messages format - commented out for easy revert):
                # if 'messages' not in example:
                #     validation_results['errors'].append(f"Example {i}: Missing 'messages' field")
                #     continue
                # 
                # messages = example['messages']
                # if not isinstance(messages, list) or len(messages) != 2:
                #     validation_results['errors'].append(f"Example {i}: 'messages' must be a list with exactly 2 entries")
                #     continue
                # 
                # # Validate user message
                # user_msg = messages[0]
                # if user_msg.get('role') != 'user':
                #     validation_results['errors'].append(f"Example {i}: First message must have role 'user'")
                #     continue
                # 
                # if 'content' not in user_msg:
                #     validation_results['errors'].append(f"Example {i}: User message missing 'content' field")
                #     continue
                # 
                # # Validate model message
                # model_msg = messages[1]
                # if model_msg.get('role') != 'model':
                #     validation_results['errors'].append(f"Example {i}: Second message must have role 'model'")
                #     continue
                # 
                # if 'content' not in model_msg:
                #     validation_results['errors'].append(f"Example {i}: Model message missing 'content' field")
                #     continue
                
                # NEW (GenerateContent format):
                if 'contents' not in example:
                    validation_results['errors'].append(f"Example {i}: Missing 'contents' field")
                    continue
                
                contents = example['contents']
                if not isinstance(contents, list) or len(contents) != 2:
                    validation_results['errors'].append(f"Example {i}: 'contents' must be a list with exactly 2 entries")
                    continue
                
                # Validate user content
                user_content = contents[0]
                if user_content.get('role') != 'user':
                    validation_results['errors'].append(f"Example {i}: First content must have role 'user'")
                    continue
                
                if 'parts' not in user_content or not isinstance(user_content['parts'], list):
                    validation_results['errors'].append(f"Example {i}: User content missing 'parts' array")
                    continue
                
                if len(user_content['parts']) != 1 or 'text' not in user_content['parts'][0]:
                    validation_results['errors'].append(f"Example {i}: User content 'parts' must contain single 'text' entry")
                    continue
                
                # Validate model content
                model_content = contents[1]
                if model_content.get('role') != 'model':
                    validation_results['errors'].append(f"Example {i}: Second content must have role 'model'")
                    continue
                
                if 'parts' not in model_content or not isinstance(model_content['parts'], list):
                    validation_results['errors'].append(f"Example {i}: Model content missing 'parts' array")
                    continue
                
                if len(model_content['parts']) != 1 or 'text' not in model_content['parts'][0]:
                    validation_results['errors'].append(f"Example {i}: Model content 'parts' must contain single 'text' entry")
                    continue
                
                # Validate user content is valid JSON with team info
                try:
                    input_text = user_content['parts'][0]['text']
                    input_content = json.loads(input_text)
                    if 'away_team' not in input_content or 'home_team' not in input_content:
                        validation_results['warnings'].append(
                            f"Example {i}: User content missing team information"
                        )
                except json.JSONDecodeError:
                    validation_results['errors'].append(f"Example {i}: User content is not valid JSON")
                    continue
                
                # Validate model content is valid JSON
                try:
                    output_text = model_content['parts'][0]['text']
                    output_content = json.loads(output_text)
                    if 'next_play' not in output_content and 'first_N_plays' not in output_content:
                        validation_results['warnings'].append(
                            f"Example {i}: Model content missing expected keys"
                        )
                except json.JSONDecodeError:
                    validation_results['errors'].append(f"Example {i}: Model content is not valid JSON")
                    continue
                
                validation_results['valid_examples'] += 1
                
            except Exception as e:
                validation_results['errors'].append(f"Example {i}: Unexpected error - {str(e)}")
        
        validation_results['error_rate'] = len(validation_results['errors']) / len(training_examples) if training_examples else 0
        validation_results['success_rate'] = validation_results['valid_examples'] / len(training_examples) if training_examples else 0
        
        return validation_results
    
    def save_training_data(self, training_examples: List[Dict[str, Any]], 
                          filename: Optional[str] = None, 
                          season_year: Optional[str] = None) -> str:
        """
        Save training examples in JSONL format for Gemini fine-tuning.
        
        Args:
            training_examples: List of training examples
            filename: Custom filename (will append '_gemini' suffix)
            season_year: Season year for filename
            
        Returns:
            str: Path to saved JSONL file
        """
        # Add Gemini-specific suffix to filename
        if filename:
            if not filename.endswith('_gemini.jsonl'):
                # Remove existing extension and add gemini suffix
                base_name = filename.replace('.jsonl', '').replace('_openai', '')
                filename = f"{base_name}_gemini.jsonl"
        
        return file_manager.save_openai_training_data(training_examples, filename, season_year)


class GeminiDatasetGenerator:
    """High-level interface for generating Gemini datasets."""
    
    def __init__(self):
        self.formatter = GeminiFormatter()
    
    def generate_for_game(self, game_id: int, season_year: str = "2023-2024", 
                         n_total: int = 5, max_plays: Optional[int] = None) -> tuple[List[Dict[str, Any]], str]:
        """
        Generate Gemini fine-tuning data for a specific game.
        
        Args:
            game_id: Game ID to generate training data for
            season_year: Season year
            n_total: Number of recent plays in context
            max_plays: Max plays to process (None for all)
            
        Returns:
            tuple: (training_examples_list, jsonl_filepath)
        """
        from training.data_generator import training_data_generator
        
        print(f"Generating Gemini training data for game {game_id}...")
        
        # Generate our structured JSON data first
        df = training_data_generator.generate_for_game(game_id, season_year, n_total, max_plays)
        
        # Convert to Gemini format
        training_examples = self.formatter.create_training_data(df)
        
        # Save as JSONL
        jsonl_path = self.formatter.save_training_data(
            training_examples, 
            filename=f"game_{game_id}_gemini_training.jsonl",
            season_year=season_year
        )
        
        return training_examples, jsonl_path
    
    def generate_dataset(self, season_year: Optional[str] = None, 
                        n_total: int = 5, sample_size: Optional[int] = None, 
                        game_id_filter: Optional[List[int]] = None) -> tuple[List[Dict[str, Any]], str]:
        """
        Generate Gemini fine-tuning data for multiple games.
        
        Args:
            season_year: Season to process (None for latest)
            n_total: Number of recent plays in context
            sample_size: Number of games to sample (None for all)
            game_id_filter: Specific game IDs to process
            
        Returns:
            tuple: (training_examples_list, jsonl_filepath)
        """
        from training.data_generator import training_data_generator
        
        print(f"Generating Gemini dataset for season {season_year}...")
        
        # Generate structured JSON data
        df = training_data_generator.generate_dataset(
            season_year=season_year,
            n_total=n_total,
            sample_size=sample_size,
            game_id_filter=game_id_filter
        )
        
        # Convert to Gemini format
        training_examples = self.formatter.create_training_data(df)
        
        # Save as JSONL
        jsonl_path = self.formatter.save_training_data(
            training_examples,
            season_year=season_year
        )
        
        return training_examples, jsonl_path


# Global instances for easy access
gemini_formatter = GeminiFormatter()
gemini_dataset_generator = GeminiDatasetGenerator()

# Register Gemini formatter with the factory
FormatterFactory.register_formatter("gemini", GeminiFormatter)


# Convenience functions for Gemini training data
def save_gemini_training_data(training_examples: List[Dict[str, Any]], 
                            filename: Optional[str] = None, 
                            season_year: Optional[str] = None) -> str:
    """Save training examples in JSONL format for Gemini fine-tuning."""
    return gemini_formatter.save_training_data(training_examples, filename, season_year)


def generate_gemini_training_for_game(game_id: int, season_year: str = "2023-2024", 
                                    n_total: int = 5, max_plays: Optional[int] = None) -> tuple[List[Dict[str, Any]], str]:
    """Generate Gemini fine-tuning data for a specific game."""
    return gemini_dataset_generator.generate_for_game(game_id, season_year, n_total, max_plays)
