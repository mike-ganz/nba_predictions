"""
OpenAI training data formatting utilities.

This module handles conversion of NBA training data to OpenAI fine-tuning format,
creating user-assistant conversation pairs suitable for model training.
"""

import json
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from game.time_utils import convert_to_quarter_time
from game.scoring_utils import determine_scoring_info
from data.file_utils import file_manager


class OpenAIFormatter:
    """Handles conversion to OpenAI fine-tuning format."""
    
    def create_training_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert our JSON training data into OpenAI fine-tuning JSONL format.
        Each row becomes a user-assistant pair where:
        - User: Our JSON context (team stats, players, recent plays)  
        - Assistant: The next play in the sequence
        
        Args:
            df: DataFrame with 'json_training_data' column
            
        Returns:
            list: List of training examples in OpenAI format
        """
        training_examples = []
        
        # Group by game to ensure we can find next plays
        for game_id in df['game_id'].unique():
            game_df = df[df['game_id'] == game_id].sort_values('play_id').reset_index(drop=True)
            
            for i in range(len(game_df) - 1):  # -1 because we need a next play
                example = self._create_training_example(game_df, i)
                if example:
                    training_examples.append(example)
        
        return training_examples
    
    def _create_training_example(self, game_df: pd.DataFrame, current_index: int) -> Optional[Dict[str, Any]]:
        """
        Create a single OpenAI training example from current and next plays.
        
        Args:
            game_df: DataFrame for a single game, sorted by play_id
            current_index: Index of current play
            
        Returns:
            dict or None: OpenAI training example or None if creation failed
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
        
        # Format score
        score = (f"{current_json['away_team']['name']} {next_row['away_score']} - "
                f"{current_json['home_team']['name']} {next_row['home_score']}")
        
        # Create the assistant response
        return {
            "next_play": {
                "quarter": int(next_quarter),
                "time_remaining": str(next_time),
                "description": str(next_row['description']),
                "score": str(score),
                "scoring_team": str(scoring_team) if scoring_team else None,
                "points_scored": int(points_scored) if points_scored else 0
            }
        }
    
    def _get_previous_scores(self, game_df: pd.DataFrame, current_index: int) -> Tuple[int, int]:
        """
        Get the scores from the previous play for scoring calculation.
        
        Args:
            game_df: DataFrame for the game
            current_index: Index of current play
            
        Returns:
            tuple: (prev_away_score, prev_home_score)
        """
        if current_index == 0:
            return 0, 0
        else:
            prev_row = game_df.iloc[current_index - 1]
            return prev_row['away_score'], prev_row['home_score']
    
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
