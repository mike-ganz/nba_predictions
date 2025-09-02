"""
Google Gemini training data formatting utilities.

This module handles conversion of NBA training data to Gemini fine-tuning format,
creating input-output pairs suitable for Gemini model training.
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
    
    def create_training_data(self, df: pd.DataFrame, generation_mode: str = "remaining_plays") -> List[Dict[str, Any]]:
        """
        Convert our JSON training data into Gemini fine-tuning JSONL format.
        Handles two generation modes:
        - remaining_plays: Standard input-output pairs with next play
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
            for game_id in df['game_id'].unique():
                game_df = df[df['game_id'] == game_id].sort_values('play_id').copy()
                
                # Find the first game context row (should have no recent_plays)
                context_rows = game_df[game_df['json_training_data'].str.contains('"recent_plays":[]')]
                if len(context_rows) > 0:
                    context_row = context_rows.iloc[0]
                    example = self._create_first_n_plays_example(game_df, context_row)
                    if example:
                        training_examples.append(example)
        else:
            # Mode 1: Standard remaining_plays pairs
            for game_id in df['game_id'].unique():
                game_df = df[df['game_id'] == game_id].sort_values('play_id').copy()
                
                # Create training examples for consecutive plays
                for i in range(len(game_df) - 1):
                    example = self._create_training_example(game_df, i)
                    if example:
                        training_examples.append(example)
        
        return training_examples
    
    def _create_training_example(self, game_df: pd.DataFrame, current_index: int) -> Optional[Dict[str, Any]]:
        """
        Create a single Gemini training example from current and next plays.
        
        Args:
            game_df: DataFrame for a single game, sorted by play_id
            current_index: Index of current play
            
        Returns:
            dict or None: Gemini training example or None if creation failed
        """
        try:
            current_row = game_df.iloc[current_index]
            next_row = game_df.iloc[current_index + 1]
            
            # Parse the current JSON context
            current_json = json.loads(current_row['json_training_data'])
            
            # Create the next play response
            assistant_response = self._create_next_play_response(
                game_df, current_index, next_row, current_json
            )
            
            # Create Gemini training example with input_text and output_text
            training_example = {
                "input_text": current_row['json_training_data'],  # Our JSON context
                "output_text": json.dumps(assistant_response, separators=(',', ':'))
            }
            
            return training_example
            
        except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
            print(f"Warning: Skipped Gemini training example due to error: {e}")
            return None
    
    def _create_first_n_plays_example(self, game_df: pd.DataFrame, context_row: pd.Series) -> Optional[Dict[str, Any]]:
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
            
            # Get first DEFAULT_N_TOTAL_PLAYS plays from the game
            plays_df = game_df.head(DEFAULT_N_TOTAL_PLAYS).copy()
            first_plays = []
            
            for _, row in plays_df.iterrows():
                play_json = json.loads(row['json_training_data'])
                if 'recent_plays' in play_json and len(play_json['recent_plays']) > 0:
                    # Get the most recent play (last in the recent_plays array)
                    recent_play = play_json['recent_plays'][-1]
                    first_plays.append(recent_play)
            
            # Create the response with first N plays
            assistant_response = {
                "first_N_plays": first_plays
            }
            
            # Create Gemini training example
            training_example = {
                "input_text": context_row['json_training_data'],  # Context without recent_plays
                "output_text": json.dumps(assistant_response, separators=(',', ':'))
            }
            
            return training_example
            
        except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
            print(f"Warning: Skipped first_N_plays Gemini training example due to error: {e}")
            return None
    
    def _create_next_play_response(self, game_df: pd.DataFrame, current_index: int, 
                                 next_row: pd.Series, current_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create the assistant response (next play prediction) for Gemini training.
        
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
            # This should not happen - debug log the issue
            print(f"⚠️ Scoring mismatch detected:")
            print(f"   Description: {next_row['description']}")
            print(f"   Scores: {prev_away_score}-{prev_home_score} → {next_row['away_score']}-{next_row['home_score']}")
            print(f"   Calculated: {scoring_team} +{points_scored}")
            # Force correction for obvious non-scoring plays
            scoring_team, points_scored = None, 0
        
        # Format score
        score = (f"{current_json['away_team']['name']} {next_row['away_score']} - "
                f"{current_json['home_team']['name']} {next_row['home_score']}")
        
        # Create shot_details object - populated only for shots
        next_event_type = next_row.get('event_type', '')
        if next_event_type == 'shot':
            # For shots, determine the shooting team from the row data
            shooting_team = next_row.get('team', '')  # Get team that took the shot
            
            # Apply coordinate normalization for shots
            raw_x = next_row.get('converted_x')
            raw_y = next_row.get('converted_y')
            x_norm, y_norm = normalize_shot_coordinates(raw_x, raw_y)
            
            shot_details = {
                "team": str(shooting_team) if shooting_team else None,
                "points": int(points_scored) if points_scored else 0,
                "x_coord": x_norm,
                "y_coord": y_norm
            }
        else:
            shot_details = {
                "team": None,
                "points": None,
                "x_coord": None,
                "y_coord": None
            }
        
        # Create the assistant response with shot_details and player info
        next_player = next_row.get('player')
        return {
            "next_play": {
                "quarter": int(next_quarter),
                "time_remaining": str(next_time),
                "score": str(score),
                "players_on_court": extract_players_on_court(next_row, current_json['away_team']['name'], current_json['home_team']['name']),
                "player": str(next_player) if pd.notna(next_player) else None,
                "description": process_play_description(next_row),
                "shot_details": shot_details
            }
        }
    
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
                # Check required fields for Gemini format
                if 'input_text' not in example:
                    validation_results['errors'].append(f"Example {i}: Missing 'input_text' field")
                    continue
                    
                if 'output_text' not in example:
                    validation_results['errors'].append(f"Example {i}: Missing 'output_text' field")
                    continue
                
                # Validate input_text is valid JSON
                try:
                    input_content = json.loads(example['input_text'])
                    if 'away_team' not in input_content or 'home_team' not in input_content:
                        validation_results['warnings'].append(
                            f"Example {i}: Input text missing team information"
                        )
                except json.JSONDecodeError:
                    validation_results['errors'].append(f"Example {i}: Input text is not valid JSON")
                    continue
                
                # Validate output_text is valid JSON
                try:
                    output_content = json.loads(example['output_text'])
                    if 'next_play' not in output_content and 'first_N_plays' not in output_content:
                        validation_results['warnings'].append(
                            f"Example {i}: Output text missing expected keys"
                        )
                except json.JSONDecodeError:
                    validation_results['errors'].append(f"Example {i}: Output text is not valid JSON")
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
