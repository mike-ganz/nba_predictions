"""
OpenAI training data formatting utilities.

This module handles conversion of NBA training data to OpenAI fine-tuning format,
creating user-assistant conversation pairs suitable for model training.
"""

import json
from typing import List, Dict, Any, Tuple, Optional, Iterator
import pandas as pd
import re
from game.time_utils import convert_to_quarter_time
from game.scoring_utils import determine_scoring_info
from data.file_utils import file_manager
from training.data_generator import extract_players_on_court
from training.base_formatter import BaseFormatter, FormatterFactory

_RAW_SEASON_CACHE: Dict[str, pd.DataFrame] = {}
_SEASON_INDEX_CACHE: Dict[str, Dict[Any, Tuple[pd.DataFrame, Dict[Any, int]]]] = {}

def _get_raw_df_for_season(season_year: str) -> pd.DataFrame:
    """Load and cache raw play-by-play data for a given season."""
    if season_year in _RAW_SEASON_CACHE:
        return _RAW_SEASON_CACHE[season_year]
    from generate_training_data import load_play_by_play_data
    raw_df = load_play_by_play_data(season_year)
    _RAW_SEASON_CACHE[season_year] = raw_df
    return raw_df


def _get_season_index(season_year: str) -> Dict[Any, Tuple[pd.DataFrame, Dict[Any, int]]]:
    """Build and cache season-level per-game indices for fast lookups.
    Returns mapping: game_id -> (raw_game_df_sorted_by_play_id, play_id_to_index)
    """
    if season_year in _SEASON_INDEX_CACHE:
        return _SEASON_INDEX_CACHE[season_year]
    raw_df = _get_raw_df_for_season(season_year)
    # Sort once for the season
    raw_sorted = raw_df.sort_values(['game_id', 'play_id']).reset_index(drop=True)
    season_index: Dict[Any, Tuple[pd.DataFrame, Dict[Any, int]]] = {}
    for game_id, grp in raw_sorted.groupby('game_id', sort=False):
        # grp already sorted by play_id due to prior sort
        play_ids = grp['play_id'].tolist()
        try:
            pid_map = {int(pid): i for i, pid in enumerate(play_ids)}
        except Exception:
            pid_map = {pid: i for i, pid in enumerate(play_ids)}
        season_index[game_id] = (grp.reset_index(drop=True), pid_map)
    _SEASON_INDEX_CACHE[season_year] = season_index
    return season_index

def remove_parentheses_content(text):
    """
    Remove content within parentheses (including the parentheses) from text.
    
    Args:
        text (str): Input text that may contain parentheses
        
    Returns:
        str: Text with parentheses content removed and extra spaces cleaned up
    
    Example:
        "Lebron James 3-pt make (17 pts)" -> "Lebron James 3-pt make"
    """
    if not text or pd.isna(text):
        return text
    
    # Remove content within parentheses using regex
    # \([^)]*\) matches opening paren, any chars except closing paren, closing paren
    cleaned_text = re.sub(r'\([^)]*\)', '', str(text))
    
    # Clean up extra whitespace that may result from removal
    cleaned_text = ' '.join(cleaned_text.split())
    
    return cleaned_text


class OpenAIFormatter(BaseFormatter):
    """Handles conversion to OpenAI fine-tuning format."""
    
    @property
    def platform_name(self) -> str:
        """Return the platform name."""
        return "OpenAI"
    
    def _is_compact_format(self, json_data: Dict[str, Any]) -> bool:
        """Check if the JSON data is in compact format."""
        if not isinstance(json_data, dict):
            return False
        
        # Check for compact format indicators
        compact_keys = {'a', 'h', 'as', 'hs', 'ap', 'hp'}
        verbose_keys = {'away_team', 'home_team'}
        
        has_compact = any(key in json_data for key in compact_keys)
        has_verbose = any(key in json_data for key in verbose_keys)
        
        return has_compact and not has_verbose
    
    def _convert_compact_to_verbose_for_formatter(self, compact_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert compact format to verbose format for formatter processing.
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
            print(f"⚠️ Warning: Failed to convert compact to verbose: {e}")
            return compact_data  # Return original if conversion fails
    
    def create_training_data(self, df: pd.DataFrame, generation_mode: str = "remaining_plays", n_total: int = None, season_year: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Convert our JSON training data into OpenAI fine-tuning JSONL format.
        Handles two generation modes:
        - remaining_plays: Standard user-assistant pairs with next play
        - first_N_plays: Single entry per game with first N plays as targets
        
        Args:
            df: DataFrame with 'json_training_data' column
            generation_mode: "remaining_plays" or "first_N_plays"
            
        Returns:
            list: List of training examples in OpenAI format
        """
        training_examples = []
        
        if generation_mode == "first_N_plays":
            # Mode 2: Single entry per game with first N plays as targets
            # Load raw data to ensure proper sequential play selection
            if season_year is None:
                from config.settings import get_current_season_year
                season_year = get_current_season_year()
            raw_df = _get_raw_df_for_season(season_year)
            
            for game_id in sorted(df['game_id'].unique()):
                game_df = df[df['game_id'] == game_id].sort_values('play_id')
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
                                # Verbose format
                                context_row = game_df.iloc[i]
                                break
                            elif self._is_compact_format(parsed):
                                # Compact format - convert and use
                                context_row = game_df.iloc[i]
                                break
                        except json.JSONDecodeError:
                            continue
                
                if context_row is not None:
                    example = self._create_first_n_plays_example(game_df, context_row, n_total=n_total)
                    if example:
                        training_examples.append(example)
        else:
            # Mode 1: Standard processing (remaining_plays)
            # Load raw data to ensure proper sequential play selection
            if season_year is None:
                from config.settings import get_current_season_year
                season_year = get_current_season_year()
            raw_df = _get_raw_df_for_season(season_year)
            
            for game_id in sorted(df['game_id'].unique()):
                game_df = df[df['game_id'] == game_id].sort_values('play_id').reset_index(drop=True)
                raw_game_df = raw_df[raw_df['game_id'] == game_id].sort_values('play_id').reset_index(drop=True)
                # Build O(1) lookup from play_id to index to avoid O(n^2)
                try:
                    play_id_to_raw_index = {int(pid): i for i, pid in enumerate(raw_game_df['play_id'].tolist())}
                except Exception:
                    play_id_to_raw_index = {pid: i for i, pid in enumerate(raw_game_df['play_id'].tolist())}
                
                for i in range(len(game_df) - 1):  # -1 because we need a next play
                    # Skip rows with invalid JSON data
                    current_json_data = game_df.iloc[i]['json_training_data']
                    if not current_json_data or current_json_data.strip() == "" or current_json_data.strip() == "{}":
                        continue
                    
                    example = self._create_training_example(game_df, i, raw_game_df, play_id_to_raw_index)
                    if example:
                        training_examples.append(example)
        
        return training_examples

    def iter_training_data(self, df: pd.DataFrame, generation_mode: str = "remaining_plays",
                           n_total: int = None, season_year: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        """Yield OpenAI-format training examples as they are created (streaming)."""
        if generation_mode == "first_N_plays":
            # Prepare raw df if needed (not strictly required here)
            for game_id in sorted(df['game_id'].unique()):
                game_df = df[df['game_id'] == game_id].sort_values('play_id')
                # Find a context row
                context_row = None
                for i in range(len(game_df)):
                    json_data = game_df.iloc[i]['json_training_data']
                    if json_data and json_data.strip() and json_data.strip() != "{}":
                        try:
                            parsed = json.loads(json_data)
                            if 'away_team' in parsed and 'home_team' in parsed:
                                context_row = game_df.iloc[i]
                                break
                            elif self._is_compact_format(parsed):
                                context_row = game_df.iloc[i]
                                break
                        except json.JSONDecodeError:
                            continue
                if context_row is not None:
                    example = self._create_first_n_plays_example(game_df, context_row, n_total=n_total)
                    if example:
                        yield example
        else:
            # remaining_plays
            if season_year is None:
                from config.settings import get_current_season_year
                season_year = get_current_season_year()
            season_index = _get_season_index(season_year)
            for game_id in sorted(df['game_id'].unique()):
                game_df = df[df['game_id'] == game_id].sort_values('play_id').reset_index(drop=True)
                if game_id in season_index:
                    raw_game_df, play_id_to_raw_index = season_index[game_id]
                else:
                    # Fallback (should be rare)
                    raw_df = _get_raw_df_for_season(season_year)
                    raw_game_df = raw_df[raw_df['game_id'] == game_id].sort_values('play_id').reset_index(drop=True)
                    try:
                        play_id_to_raw_index = {int(pid): i for i, pid in enumerate(raw_game_df['play_id'].tolist())}
                    except Exception:
                        play_id_to_raw_index = {pid: i for i, pid in enumerate(raw_game_df['play_id'].tolist())}
                for i in range(len(game_df) - 1):
                    current_json_data = game_df.iloc[i]['json_training_data']
                    if not current_json_data or current_json_data.strip() == "" or current_json_data.strip() == "{}":
                        continue
                    example = self._create_training_example(game_df, i, raw_game_df, play_id_to_raw_index)
                    if example:
                        yield example
    
    def _create_training_example(self, game_df: pd.DataFrame, current_index: int, raw_game_df: pd.DataFrame = None, play_id_to_raw_index: Optional[dict] = None) -> Optional[Dict[str, Any]]:
        """
        Create a single OpenAI training example from current and next plays.
        
        Args:
            game_df: DataFrame for a single game (filtered training data)
            current_index: Index of current play in filtered data
            raw_game_df: Raw game DataFrame for finding sequential next play
            
        Returns:
            dict or None: OpenAI training example or None if creation failed
        """
        try:
            current_row = game_df.iloc[current_index]
            
            # Find the actual next play in sequence using play_id
            current_play_id = current_row['play_id']
            
            if raw_game_df is not None:
                # Find the next sequential play in raw data
                if play_id_to_raw_index is not None:
                    current_raw_index = play_id_to_raw_index.get(current_play_id)
                else:
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
            is_compact_context = self._is_compact_format(current_json_raw)
            if is_compact_context:
                current_json = self._convert_compact_to_verbose_for_formatter(current_json_raw)
            else:
                current_json = current_json_raw
            
            # Create the next play response
            assistant_response = self._create_next_play_response(
                game_df, current_index, next_row, current_json
            )
            
            # Create OpenAI training example
            training_example = {
                "messages": [
                    {
                        "role": "user",
                        "content": current_row['json_training_data']  # Our JSON context
                    },
                    {
                        "role": "assistant", 
                        "content": json.dumps(assistant_response, separators=(',', ':'))
                    }
                ]
            }
            
            return training_example
            
        except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
            print(f"Warning: Skipped training example due to error: {e}")
            return None
    
    def _create_first_n_plays_example(self, game_df: pd.DataFrame, context_row: pd.Series,
                                      n_total: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Create a training example for first_N_plays mode.
        
        Args:
            game_df: DataFrame for a single game, sorted by play_id
            context_row: Row containing the JSON context (with empty recent_plays)
            
        Returns:
            dict or None: OpenAI training example or None if creation failed
        """
        try:
            # Use the context row's JSON context (which should have empty recent_plays)
            current_json_raw = json.loads(context_row['json_training_data'])
            # Detect original schema (needed for fallbacks) and convert if compact
            is_compact_context = self._is_compact_format(current_json_raw)
            if is_compact_context:
                current_json = self._convert_compact_to_verbose_for_formatter(current_json_raw)
            else:
                current_json = current_json_raw
            
            # Build first N plays as COMPACT tuples (no players_on_court)
            from config.settings import DEFAULT_N_TOTAL_PLAYS
            from game.time_utils import convert_to_quarter_time
            from game.scoring_utils import determine_scoring_info
            from generate_training_data import map_structured_to_event_code, map_description_to_event_code, resolve_actor

            if n_total is None:
                n_total = DEFAULT_N_TOTAL_PLAYS
            compact_plays = []

            # Prepare team names for scoring/team mapping
            away_team = current_json['away_team']['name']
            home_team = current_json['home_team']['name']

            # Build player name → index maps
            away_name_to_idx: Dict[str, int] = {}
            home_name_to_idx: Dict[str, int] = {}

            away_players_verbose = current_json.get('away_team', {}).get('players')
            home_players_verbose = current_json.get('home_team', {}).get('players')

            if isinstance(away_players_verbose, list):
                for idx, player in enumerate(away_players_verbose):
                    if isinstance(player, dict):
                        name = player.get('name')
                        if name:
                            away_name_to_idx[str(name)] = idx

            if isinstance(home_players_verbose, list):
                for idx, player in enumerate(home_players_verbose):
                    if isinstance(player, dict):
                        name = player.get('name')
                        if name:
                            home_name_to_idx[str(name)] = idx

            # Fallback to compact arrays if verbose players missing or empty
            if (not away_name_to_idx or not home_name_to_idx) and is_compact_context:
                for idx, player_data in enumerate(current_json_raw.get('ap', []) or []):
                    if isinstance(player_data, list) and player_data:
                        away_name_to_idx[str(player_data[0])] = idx
                for idx, player_data in enumerate(current_json_raw.get('hp', []) or []):
                    if isinstance(player_data, list) and player_data:
                        home_name_to_idx[str(player_data[0])] = idx

            # Iterate first N valid rows and convert each to compact tuple
            for i in range(len(game_df)):
                if len(compact_plays) >= n_total:
                    break

                row = game_df.iloc[i]
                if pd.isna(row.get('description')):
                    continue

                # Quarter/time
                q, t_str = convert_to_quarter_time(row['period'], row['remaining_time'])
                # Time to seconds
                ts = 0
                try:
                    parts = str(t_str).split(':')
                    if len(parts) >= 2:
                        ts = int(parts[-2]) * 60 + int(parts[-1])
                except Exception:
                    ts = 720

                # Scores
                curr_away = int(row.get('away_score', 0) or 0)
                curr_home = int(row.get('home_score', 0) or 0)
                score_arr = [curr_away, curr_home]

                # Previous scores (for first row, assume 0-0)
                if i == 0:
                    prev_away, prev_home = 0, 0
                else:
                    prev = game_df.iloc[i - 1]
                    prev_away = int(prev.get('away_score', 0) or 0)
                    prev_home = int(prev.get('home_score', 0) or 0)

                scoring_team, points_scored = determine_scoring_info(
                    prev_away, prev_home, curr_away, curr_home, away_team, home_team
                )

                # Resolve actor using shared helper
                player_name = row.get('player')
                if pd.isna(player_name):
                    player_name = None

                team_name = row.get('team')
                if pd.isna(team_name):
                    team_name = None

                inferred_team = scoring_team
                if team_name:
                    inferred_team = team_name

                shot_details = {
                    'team': inferred_team,
                    'points': points_scored if points_scored else None
                }

                actor = resolve_actor(
                    player_name,
                    shot_details,
                    away_team,
                    home_team,
                    away_name_to_idx,
                    home_name_to_idx
                )

                # Event mapping (prefer structured)
                try:
                    if ('type' in row) or ('event_type' in row):
                        event_code, mapped_points = map_structured_to_event_code(row)
                        if mapped_points is not None:
                            points_scored = mapped_points
                    else:
                        score_delta = max(0, max(curr_away - prev_away, curr_home - prev_home))
                        event_code, mapped_points = map_description_to_event_code(
                            row.get('description', ''),
                            { 'team': scoring_team, 'points': points_scored if points_scored else None },
                            score_delta
                        )
                        if mapped_points is not None:
                            points_scored = mapped_points
                except Exception:
                    event_code = "unknown"

                # Build compact play tuple: scoring -> 7-tuple, non-scoring -> 6-tuple
                lineup_id = 0
                if points_scored and points_scored > 0:
                    play_tuple = [int(q), int(ts), [int(score_arr[0]), int(score_arr[1])], actor, event_code, int(points_scored), int(lineup_id)]
                else:
                    play_tuple = [int(q), int(ts), [int(score_arr[0]), int(score_arr[1])], actor, event_code, int(lineup_id)]

                compact_plays.append(play_tuple)

            # Assistant response matches remaining_plays key ('y') but with a list of plays
            assistant_response = { "y": compact_plays }
            
            # Create OpenAI training example
            training_example = {
                "messages": [
                    {
                        "role": "user",
                        "content": context_row['json_training_data']  # Context without recent_plays
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(assistant_response, separators=(',', ':'))
                    }
                ]
            }
            
            return training_example
            
        except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
            print(f"Warning: Skipped first_N_plays training example due to error: {e}")
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
        from game.time_utils import convert_to_quarter_time
        from game.scoring_utils import determine_scoring_info
        
        # Get quarter and time
        quarter, time_in_quarter = convert_to_quarter_time(
            row['period'], row['remaining_time']
        )
        
        # Calculate scoring by comparing with previous play in the game
        scoring_team = None
        points_scored = 0
        
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
                current_json['away_team']['name'], current_json['home_team']['name']
            )
        
        # Format score
        away_team_name = current_json['away_team']['name']
        home_team_name = current_json['home_team']['name']
        score = f"{away_team_name} {int(row.get('away_score', 0) or 0)} - {home_team_name} {int(row.get('home_score', 0) or 0)}"
        
        # Get player
        player = row.get('player')
        
        return {
            "quarter": int(quarter),
            "time_remaining": str(time_in_quarter),
            "score": score,
            "players_on_court": extract_players_on_court(row, current_json['away_team']['name'], current_json['home_team']['name']),
            "player": str(player) if pd.notna(player) else None,
            "description": remove_parentheses_content(row['description']),
            "scoring": {
                "team": str(scoring_team) if scoring_team else None,
                "points": int(points_scored)
            }
        }
    
    def _create_next_play_response(self, game_df: pd.DataFrame, current_index: int, 
                                 next_row: pd.Series, current_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create the assistant response (next play prediction) for OpenAI training.
        
        Args:
            game_df: DataFrame for the game
            current_index: Index of current play
            next_row: Next play data
            current_json: Parsed JSON context from current play
            
        Returns:
            dict: Assistant response with next play information
        """
        # Get quarter and time for next play
        next_quarter, next_time = convert_to_quarter_time(
            next_row['period'], next_row['remaining_time']
        )
        
        # Determine scoring info for next play
        prev_away_score, prev_home_score = self._get_previous_scores(game_df, current_index)
        
        scoring_team, points_scored = determine_scoring_info(
            prev_away_score, prev_home_score,
            next_row['away_score'], next_row['home_score'], 
            current_json['away_team']['name'], current_json['home_team']['name']
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
        score = (f"{current_json['away_team']['name']} {next_row['away_score']} - "
                f"{current_json['home_team']['name']} {next_row['home_score']}")
        
        # Import enhanced description processing
        from training.data_generator import process_play_description, extract_players_on_court
        
        # Create shot_details object - populated only for shots
        next_event_type = next_row.get('event_type', '')
        if next_event_type == 'shot':
            # For shots, determine the shooting team from the row data
            shooting_team = next_row.get('team', '')  # Get team that took the shot
            
            # Apply coordinate normalization for shots (commented out for now)
            # from training.data_generator import normalize_shot_coordinates
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
        
        # Create compact format assistant response
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
        away_team = current_json['away_team']['name']
        home_team = current_json['home_team']['name']
        
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
        
        # Create compact play tuple - ensure all numbers are regular Python ints for JSON serialization
        if points_scored and points_scored > 0:
            # Scoring play: [q, t, score, actor, event, pts, lineup_id]
            play_tuple = [
                int(next_quarter), 
                int(time_seconds), 
                [int(score_array[0]), int(score_array[1])], 
                actor, 
                event_code, 
                int(points_scored), 
                int(lineup_id)
            ]
        else:
            # Non-scoring play: [q, t, score, actor, event, lineup_id]
            play_tuple = [
                int(next_quarter), 
                int(time_seconds), 
                [int(score_array[0]), int(score_array[1])], 
                actor, 
                event_code, 
                int(lineup_id)
            ]
        
        # Return compact format response
        return {
            "y": play_tuple
        }
    
    def _get_previous_scores(self, game_df: pd.DataFrame, current_index: int) -> Tuple[int, int]:
        """
        Get the scores from the current context play for next play scoring calculation.
        
        Args:
            game_df: DataFrame for the game
            current_index: Index of current context play
            
        Returns:
            tuple: (context_away_score, context_home_score)
        """
        # CRITICAL FIX: Get score from current_index (context play), not current_index - 1
        # We want to compare: context_play_score → next_play_score to detect scoring
        current_row = game_df.iloc[current_index]
        return current_row['away_score'], current_row['home_score']
    
    def validate_training_examples(self, training_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate OpenAI training examples for completeness and format.
        
        Args:
            training_examples: List of OpenAI training examples
            
        Returns:
            dict: Validation results
        """
        validation_results = {
            'total_examples': len(training_examples),
            'valid_examples': 0,
            'errors': [],
            'warnings': []
        }
        
        for i, example in enumerate(training_examples):
            try:
                # Check basic structure
                if 'messages' not in example:
                    validation_results['errors'].append(f"Example {i}: Missing 'messages' key")
                    continue
                
                messages = example['messages']
                if len(messages) != 2:
                    validation_results['errors'].append(f"Example {i}: Expected 2 messages, got {len(messages)}")
                    continue
                
                # Check user message
                user_msg = messages[0]
                if user_msg.get('role') != 'user':
                    validation_results['errors'].append(f"Example {i}: First message should have role 'user'")
                    continue
                
                # Validate user content is valid JSON
                try:
                    user_content = json.loads(user_msg['content'])
                    required_keys = ['away_team', 'home_team', 'players', 'recent_plays']
                    missing_keys = [key for key in required_keys if key not in user_content]
                    if missing_keys:
                        validation_results['warnings'].append(
                            f"Example {i}: User content missing keys: {missing_keys}"
                        )
                except json.JSONDecodeError:
                    validation_results['errors'].append(f"Example {i}: User content is not valid JSON")
                    continue
                
                # Check assistant message
                assistant_msg = messages[1]
                if assistant_msg.get('role') != 'assistant':
                    validation_results['errors'].append(f"Example {i}: Second message should have role 'assistant'")
                    continue
                
                # Validate assistant content is valid JSON
                try:
                    assistant_content = json.loads(assistant_msg['content'])
                    if 'next_play' not in assistant_content:
                        validation_results['warnings'].append(
                            f"Example {i}: Assistant content missing 'next_play' key"
                        )
                except json.JSONDecodeError:
                    validation_results['errors'].append(f"Example {i}: Assistant content is not valid JSON")
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
        Save training examples in JSONL format for OpenAI fine-tuning.
        
        Args:
            training_examples: List of training examples
            filename: Custom filename
            season_year: Season year for filename
            
        Returns:
            str: Path to saved JSONL file
        """
        return file_manager.save_openai_training_data(training_examples, filename, season_year)


class OpenAIDatasetGenerator:
    """High-level interface for generating OpenAI datasets."""
    
    def __init__(self):
        self.formatter = OpenAIFormatter()
    
    def generate_for_game(self, game_id: int, season_year: str = "2023-2024", 
                         n_total: int = 5, max_plays: Optional[int] = None) -> Tuple[List[Dict[str, Any]], str]:
        """
        Generate OpenAI fine-tuning data for a specific game.
        
        Args:
            game_id: Game ID to generate training data for
            season_year: Season year
            n_total: Number of recent plays in context
            max_plays: Max plays to process (None for all)
            
        Returns:
            tuple: (training_examples_list, jsonl_filepath)
        """
        from training.data_generator import training_data_generator
        
        print(f"Generating OpenAI training data for game {game_id}...")
        
        # Generate our structured JSON data first
        df = training_data_generator.generate_for_game(game_id, season_year, n_total, max_plays)
        
        # Convert to OpenAI format
        training_examples = self.formatter.create_training_data(df)
        
        # Save as JSONL
        jsonl_path = self.formatter.save_training_data(
            training_examples, 
            filename=f"game_{game_id}_openai_training.jsonl",
            season_year=season_year
        )
        
        return training_examples, jsonl_path
    
    def generate_dataset(self, season_year: Optional[str] = None, 
                        n_total: int = 5, sample_size: Optional[int] = None, 
                        game_id_filter: Optional[List[int]] = None) -> Tuple[List[Dict[str, Any]], str]:
        """
        Generate a complete OpenAI training dataset.
        
        Args:
            season_year: Season year to use
            n_total: Number of recent plays in context
            sample_size: If provided, randomly sample this many rows
            game_id_filter: If provided, only include these game IDs
            
        Returns:
            tuple: (training_examples_list, jsonl_filepath)
        """
        from training.data_generator import training_data_generator
        
        print(f"Generating OpenAI training dataset...")
        
        # Generate our structured JSON data first
        df = training_data_generator.generate_dataset(season_year, n_total, sample_size, game_id_filter)
        
        # Convert to OpenAI format
        training_examples = self.formatter.create_training_data(df)
        
        # Validate the training examples
        validation_results = self.formatter.validate_training_examples(training_examples)
        print(f"Validation: {validation_results['valid_examples']}/{validation_results['total_examples']} examples valid")
        
        if validation_results['errors']:
            print(f"Found {len(validation_results['errors'])} validation errors")
            for error in validation_results['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
        
        # Save as JSONL
        jsonl_path = self.formatter.save_training_data(training_examples, season_year=season_year)
        
        return training_examples, jsonl_path


# Global instances
openai_formatter = OpenAIFormatter()
openai_dataset_generator = OpenAIDatasetGenerator()

# Convenience functions for backward compatibility
def create_openai_training_data(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert our JSON training data into OpenAI fine-tuning JSONL format."""
    return openai_formatter.create_training_data(df)


def save_openai_training_data(training_examples: List[Dict[str, Any]], 
                            filename: Optional[str] = None, 
                            season_year: Optional[str] = None) -> str:
    """Save training examples in JSONL format for OpenAI fine-tuning."""
    return openai_formatter.save_training_data(training_examples, filename, season_year)


def generate_openai_training_for_game(game_id: int, season_year: str = "2023-2024", 
                                    n_total: int = 5, max_plays: Optional[int] = None) -> Tuple[List[Dict[str, Any]], str]:
    """Generate OpenAI fine-tuning data for a specific game."""
    return openai_dataset_generator.generate_for_game(game_id, season_year, n_total, max_plays)


# Global instances for backward compatibility
openai_formatter = OpenAIFormatter()
openai_dataset_generator = OpenAIDatasetGenerator()

# Register OpenAI formatter with the factory
FormatterFactory.register_formatter("openai", OpenAIFormatter)
