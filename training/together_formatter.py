"""
Together.ai training data formatting utilities.

This module handles conversion of NBA training data to Together.ai fine-tuning format.
Together.ai uses the same conversational format as OpenAI (messages array), so we
inherit from OpenAIFormatter and add Together.ai-specific features.
"""

import json
from typing import List, Dict, Any, Optional
import pandas as pd
from training.openai_formatter import OpenAIFormatter
from training.base_formatter import BaseFormatter, FormatterFactory


class TogetherFormatter(OpenAIFormatter):
    """
    Handles conversion to Together.ai fine-tuning format.
    
    Together.ai uses the same conversational data format as OpenAI:
    {
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "{game_context_json}"},
        {"role": "assistant", "content": "{next_play_json}"}
      ]
    }
    
    This formatter inherits from OpenAIFormatter and adds Together.ai-specific
    validation and features.
    """
    
    @property
    def platform_name(self) -> str:
        """Return the platform name."""
        return "Together"
    
    def create_training_data(self, df: pd.DataFrame, generation_mode: str = "remaining_plays", 
                           n_total: int = None, season_year: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Convert our JSON training data into Together.ai fine-tuning JSONL format.
        
        Together.ai uses the same format as OpenAI, so we use the parent implementation
        and add Together.ai-specific validation.
        
        Args:
            df: DataFrame with 'json_training_data' column
            generation_mode: "remaining_plays" or "first_N_plays"
            n_total: Number of plays in sequence (for logging/metadata)
            
        Returns:
            list: List of training examples in Together.ai format
        """
        # Use parent class implementation (OpenAI format = Together.ai format)
        training_examples = super().create_training_data(df, generation_mode, n_total, season_year)
        
        # Add Together.ai-specific metadata if needed
        # (Currently Together.ai uses same format, but this allows future customization)
        
        return training_examples
    
    def validate_training_examples(self, training_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate Together.ai training examples for completeness and format.
        
        Together.ai has the same validation requirements as OpenAI, plus some
        additional recommendations for optimal performance.
        
        Args:
            training_examples: List of Together.ai training examples
            
        Returns:
            dict: Validation results with errors, warnings, and statistics
        """
        # Start with OpenAI validation (same format requirements)
        validation_results = super().validate_training_examples(training_examples)

        # Remove OpenAI-specific warnings that don't apply ('next_play' vs our 'y')
        filtered_warnings = []
        for w in validation_results.get('warnings', []):
            if "missing 'next_play'" in w:
                continue
            filtered_warnings.append(w)
        validation_results['warnings'] = filtered_warnings
        
        # Add Together.ai-specific recommendations
        validation_results['together_recommendations'] = []
        
        # Recommendation: Dataset size
        if len(training_examples) < 100:
            validation_results['together_recommendations'].append(
                f"Together.ai works best with at least 100-1000 examples. "
                f"You have {len(training_examples)} examples."
            )
        
        # Recommendation: Use LoRA for faster training
        validation_results['together_recommendations'].append(
            "Consider using LoRA fine-tuning for faster training and lower cost. "
            "LoRA is the default training method."
        )
        
        return validation_results
    
    def get_file_suffix(self) -> str:
        """Get the file suffix for Together.ai training files."""
        return "_together"
    
    def save_training_data(self, training_examples: List[Dict[str, Any]], 
                          filename: Optional[str] = None, 
                          season_year: Optional[str] = None) -> str:
        """
        Save training examples in JSONL format for Together.ai fine-tuning.
        
        Args:
            training_examples: List of training examples
            filename: Custom filename
            season_year: Season year for filename
            
        Returns:
            str: Path to saved JSONL file
        """
        import os
        from datetime import datetime
        
        # If no custom filename, generate one
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            season_str = season_year.replace('-', '_') if season_year else "unknown"
            filename = f"nba_{season_str}_together_{timestamp}.jsonl"
        
        # Ensure .jsonl extension
        if not filename.endswith('.jsonl'):
            filename = filename.replace('.json', '.jsonl')
            if not filename.endswith('.jsonl'):
                filename += '.jsonl'
        
        # Get output directory
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(script_dir, 'data', 'training')
        os.makedirs(output_dir, exist_ok=True)
        
        # Full path
        filepath = os.path.join(output_dir, filename)
        
        # Write JSONL file (one JSON object per line), prefer ujson if available
        try:
            import ujson as _fastjson  # type: ignore
        except Exception:
            _fastjson = json
        with open(filepath, 'w', encoding='utf-8') as f:
            for example in training_examples:
                f.write(_fastjson.dumps(example, separators=(',', ':')) + '\n')
        
        print(f"✓ Saved {len(training_examples)} examples to: {filepath}")
        
        return filepath

    def stream_save_training_data(self, df: pd.DataFrame, filename: Optional[str] = None,
                                  generation_mode: str = "remaining_plays",
                                  n_total: int = None,
                                  season_year: Optional[str] = None) -> str:
        """Stream examples directly to disk to reduce memory and latency."""
        import os
        from datetime import datetime
        try:
            import ujson as _fastjson  # type: ignore
        except Exception:
            _fastjson = json
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            season_str = season_year.replace('-', '_') if season_year else "unknown"
            filename = f"nba_{season_str}_together_{timestamp}.jsonl"
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(script_dir, 'data', 'training')
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        # Stream from OpenAIFormatter.iter_training_data (same format)
        with open(filepath, 'w', encoding='utf-8') as f:
            for example in super().iter_training_data(df, generation_mode, n_total, season_year):
                f.write(_fastjson.dumps(example, separators=(',', ':')) + '\n')
        print(f"✓ Stream-saved Together.ai dataset to: {filepath}")
        return filepath


class TogetherDatasetGenerator:
    """High-level interface for generating Together.ai datasets."""
    
    def __init__(self):
        self.formatter = TogetherFormatter()
    
    def generate_dataset(self, df: pd.DataFrame, generation_mode: str = "remaining_plays",
                        n_total: int = None, validate: bool = True) -> tuple:
        """
        Generate a complete Together.ai training dataset from DataFrame.
        
        Args:
            df: DataFrame with training data
            generation_mode: "remaining_plays" or "first_N_plays"
            n_total: Number of plays in sequence
            validate: Whether to validate the generated examples
            
        Returns:
            tuple: (training_examples_list, validation_results_dict)
        """
        print(f"Generating Together.ai training dataset...")
        print(f"  Mode: {generation_mode}")
        print(f"  Input rows: {len(df):,}")
        
        # Convert to Together.ai format
        training_examples = self.formatter.create_training_data(df, generation_mode, n_total)
        
        print(f"  Generated examples: {len(training_examples):,}")
        
        # Validate if requested
        validation_results = None
        if validate:
            validation_results = self.formatter.validate_training_examples(training_examples)
            print(f"\nValidation Results:")
            print(f"  Valid examples: {validation_results['valid_examples']}/{validation_results['total_examples']}")
            print(f"  Success rate: {validation_results['success_rate']:.1%}")
            
            if validation_results.get('errors'):
                print(f"  Errors: {len(validation_results['errors'])}")
                for error in validation_results['errors'][:3]:
                    print(f"    - {error}")
            
            if validation_results.get('together_recommendations'):
                print(f"\n  Together.ai Recommendations:")
                for rec in validation_results['together_recommendations']:
                    print(f"    • {rec}")
        
        return training_examples, validation_results
    
    def save_dataset(self, training_examples: List[Dict[str, Any]], 
                    filename: Optional[str] = None,
                    season_year: Optional[str] = None) -> str:
        """
        Save training examples to JSONL file.
        
        Args:
            training_examples: List of training examples
            filename: Custom filename
            season_year: Season year for filename
            
        Returns:
            str: Path to saved file
        """
        return self.formatter.save_training_data(training_examples, filename, season_year)


# Global instances
together_formatter = TogetherFormatter()
together_dataset_generator = TogetherDatasetGenerator()

# Register Together.ai formatter with the factory
FormatterFactory.register_formatter("together", TogetherFormatter)


# Convenience functions
def create_together_training_data(df: pd.DataFrame, generation_mode: str = "remaining_plays",
                                 n_total: int = None) -> List[Dict[str, Any]]:
    """Convert our JSON training data into Together.ai fine-tuning JSONL format."""
    return together_formatter.create_training_data(df, generation_mode, n_total)


def save_together_training_data(training_examples: List[Dict[str, Any]], 
                               filename: Optional[str] = None, 
                               season_year: Optional[str] = None) -> str:
    """Save training examples in JSONL format for Together.ai fine-tuning."""
    return together_formatter.save_training_data(training_examples, filename, season_year)


def validate_together_training_data(training_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate Together.ai training examples."""
    return together_formatter.validate_training_examples(training_examples)

