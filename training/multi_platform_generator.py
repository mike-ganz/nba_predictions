"""
Multi-platform training data generator.

This module provides a unified interface for generating training data
in multiple formats (OpenAI, Gemini, etc.) with a single API.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from training.base_formatter import FormatterFactory

# Import formatters to ensure they're registered with the factory
import training.openai_formatter
import training.gemini_formatter


class MultiPlatformGenerator:
    """
    Unified training data generator supporting multiple platforms.
    
    This class provides a consistent interface for generating training data
    in different formats (OpenAI, Gemini, etc.) by leveraging the formatter
    factory pattern.
    """
    
    def __init__(self, platform: str = "openai"):
        """
        Initialize the multi-platform generator.
        
        Args:
            platform: Target platform name (e.g., 'openai', 'gemini')
        """
        self.platform = platform.lower()
        self.formatter = FormatterFactory.create_formatter(self.platform)
    
    def set_platform(self, platform: str) -> None:
        """
        Change the target platform and update the formatter.
        
        Args:
            platform: New target platform name
        """
        self.platform = platform.lower()
        self.formatter = FormatterFactory.create_formatter(self.platform)
    
    def get_platform(self) -> str:
        """Get the current platform name."""
        return self.platform
    
    def get_available_platforms(self) -> List[str]:
        """Get list of available platforms."""
        return FormatterFactory.get_available_platforms()
    
    def generate_for_game(self, game_id: int, platform: Optional[str] = None,
                         season_year: str = "2023-2024", n_total: int = 5, 
                         max_plays: Optional[int] = None) -> Tuple[List[Dict[str, Any]], str]:
        """
        Generate training data for a specific game in the specified format.
        
        Args:
            game_id: Game ID to generate training data for
            platform: Platform format to use (None to use current platform)
            season_year: Season year
            n_total: Number of recent plays in context
            max_plays: Max plays to process (None for all)
            
        Returns:
            tuple: (training_examples_list, jsonl_filepath)
        """
        # Switch platform if specified
        if platform and platform.lower() != self.platform:
            self.set_platform(platform)
        
        from training.data_generator import training_data_generator
        
        print(f"Generating {self.formatter.platform_name} training data for game {game_id}...")
        
        # Generate our structured JSON data first
        df = training_data_generator.generate_for_game(game_id, season_year, n_total, max_plays)
        
        # Convert to platform-specific format
        training_examples = self.formatter.create_training_data(df)
        
        # Save as JSONL with platform-specific naming
        jsonl_path = self.formatter.save_training_data(
            training_examples, 
            filename=f"game_{game_id}{self.formatter.get_file_suffix()}_training.jsonl",
            season_year=season_year
        )
        
        return training_examples, jsonl_path
    
    def generate_dataset(self, platform: Optional[str] = None,
                        season_year: Optional[str] = None, n_total: int = 5, 
                        sample_size: Optional[int] = None, 
                        game_id_filter: Optional[List[int]] = None,
                        generation_mode: str = "remaining_plays") -> Tuple[List[Dict[str, Any]], str]:
        """
        Generate training data for multiple games in the specified format.
        
        Args:
            platform: Platform format to use (None to use current platform)
            season_year: Season to process (None for latest)
            n_total: Number of recent plays in context
            sample_size: Number of rows to sample (None for all) - will be converted to game limit
            game_id_filter: Specific game IDs to process
            generation_mode: "remaining_plays" or "first_N_plays"
            
        Returns:
            tuple: (training_examples_list, jsonl_filepath)
        """
        # Switch platform if specified
        if platform and platform.lower() != self.platform:
            self.set_platform(platform)
        
        from training.data_generator_ultra_optimized import UltraOptimizedTrainingDataGenerator
        ultra_optimized_training_data_generator = UltraOptimizedTrainingDataGenerator()
        from data.loaders import data_loader
        
        print(f"Generating {self.formatter.platform_name} dataset for season {season_year}...")
        print(f"Mode: {generation_mode}")
        print(f"Platform: {self.formatter.platform_name}")
        
        # Handle game-based sampling instead of row-based sampling
        if sample_size and not game_id_filter:
            # Convert row-based sample_size to game-based filtering
            print(f"🎯 Converting row limit ({sample_size:,}) to game limit...")
            
            # Load data to get game IDs
            from config.settings import config
            if season_year:
                config.season_year = season_year
            
            df = data_loader.load_play_by_play_data(season_year)
            unique_games = df['game_id'].unique()
            
            # Estimate games needed (450 plays per game average)
            estimated_games = max(1, sample_size // 450)
            actual_games = min(estimated_games, len(unique_games))
            
            print(f"📊 Found {len(unique_games):,} total games in season")
            print(f"🎯 Selecting first {actual_games} games (estimated from {sample_size:,} rows)")
            
            # Use first N games for consistent results
            game_id_filter = unique_games[:actual_games].tolist()
            sample_size = None  # Clear sample_size since we're using game_id_filter
            
            print(f"🎮 Selected games: {game_id_filter[:5]}{'...' if len(game_id_filter) > 5 else ''}")
        
        # Load and preprocess data first
        from config.settings import config
        if season_year:
            config.season_year = season_year
        
        df = data_loader.load_play_by_play_data(season_year)
        
        # Apply game filtering if specified
        if game_id_filter:
            df = df[df['game_id'].isin(game_id_filter)]
            print(f"Filtered to {len(df)} rows from {len(game_id_filter)} games")
        
        # Sort by game_id and play sequence for proper order
        df = df.sort_values(['game_id', 'play_id']).reset_index(drop=True)
        
        # Generate structured JSON data using ultra-optimized generator with generation_mode support
        df = ultra_optimized_training_data_generator.create_llm_training_data(
            df,
            n_total=n_total,
            force_real_pca=False,
            generation_mode=generation_mode  # 🔧 CRITICAL FIX: Pass generation_mode directly!
        )
        
        # Apply sampling if specified (after generation to avoid sampling raw plays)
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            print(f"Sampled {sample_size} rows from dataset")
        
        print(f"Generated dataset with {len(df)} rows")
        print(f"Each row contains up to {n_total} descriptions concatenated together")
        
        # Convert to platform-specific format
        training_examples = self.formatter.create_training_data(df, generation_mode)
        
        # Save as JSONL with platform-specific naming
        jsonl_path = self.formatter.save_training_data(
            training_examples,
            season_year=season_year
        )
        
        return training_examples, jsonl_path
    
    def validate_training_data(self, training_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate training examples using platform-specific validation.
        
        Args:
            training_examples: List of training examples to validate
            
        Returns:
            dict: Validation results with platform-specific checks
        """
        return self.formatter.validate_training_examples(training_examples)
    
    def compare_formats(self, game_id: int, season_year: str = "2023-2024",
                       n_total: int = 5, max_plays: int = 10) -> Dict[str, Any]:
        """
        Generate the same game data in all available formats for comparison.
        
        Args:
            game_id: Game ID to generate training data for
            season_year: Season year
            n_total: Number of recent plays in context
            max_plays: Max plays to process (keep small for comparison)
            
        Returns:
            dict: Training examples for each platform format
        """
        from training.data_generator import training_data_generator
        
        print(f"Comparing formats for game {game_id} (first {max_plays} plays)...")
        
        # Generate base structured JSON data once
        df = training_data_generator.generate_for_game(game_id, season_year, n_total, max_plays)
        
        comparison_results = {
            'game_id': game_id,
            'total_examples': len(df),
            'platforms': {}
        }
        
        # Generate in each available format
        available_platforms = self.get_available_platforms()
        for platform in available_platforms:
            try:
                self.set_platform(platform)
                training_examples = self.formatter.create_training_data(df)
                
                comparison_results['platforms'][platform] = {
                    'total_examples': len(training_examples),
                    'sample_example': training_examples[0] if training_examples else None,
                    'format_valid': len(training_examples) > 0
                }
                
                print(f"  ✅ {platform.upper()}: {len(training_examples)} examples generated")
                
            except Exception as e:
                comparison_results['platforms'][platform] = {
                    'error': str(e),
                    'format_valid': False
                }
                print(f"  ❌ {platform.upper()}: Error - {e}")
        
        return comparison_results
    
    def save_training_data(self, training_examples: List[Dict[str, Any]], 
                          filename: Optional[str] = None, 
                          season_year: Optional[str] = None) -> str:
        """
        Save training data using the current platform formatter.
        
        Args:
            training_examples: List of training examples
            filename: Custom filename
            season_year: Season year for filename
            
        Returns:
            str: Path to saved JSONL file
        """
        return self.formatter.save_training_data(training_examples, filename, season_year)


# Global instance for easy access
multi_platform_generator = MultiPlatformGenerator()


# Convenience functions for multi-platform generation
def generate_training_data(game_id: int, platform: str = "openai",
                         season_year: str = "2023-2024", n_total: int = 5, 
                         max_plays: Optional[int] = None) -> Tuple[List[Dict[str, Any]], str]:
    """
    Generate training data for a game in the specified platform format.
    
    Args:
        game_id: Game ID to generate training data for
        platform: Target platform ('openai', 'gemini')
        season_year: Season year
        n_total: Number of recent plays in context
        max_plays: Max plays to process
        
    Returns:
        tuple: (training_examples_list, jsonl_filepath)
    """
    return multi_platform_generator.generate_for_game(
        game_id, platform, season_year, n_total, max_plays
    )


def generate_dataset(platform: str = "openai", season_year: Optional[str] = None,
                    n_total: int = 5, sample_size: Optional[int] = None,
                    game_id_filter: Optional[List[int]] = None,
                    generation_mode: str = "remaining_plays") -> Tuple[List[Dict[str, Any]], str]:
    """
    Generate training dataset in the specified platform format.
    
    Args:
        platform: Target platform ('openai', 'gemini')
        season_year: Season to process
        n_total: Number of recent plays in context
        sample_size: Number of games to sample
        game_id_filter: Specific game IDs to process
        generation_mode: "remaining_plays" or "first_N_plays"
        
    Returns:
        tuple: (training_examples_list, jsonl_filepath)
    """
    return multi_platform_generator.generate_dataset(
        platform, season_year, n_total, sample_size, game_id_filter, generation_mode
    )


def get_available_platforms() -> List[str]:
    """Get list of available platform formats."""
    return FormatterFactory.get_available_platforms()


def compare_platform_formats(game_id: int, season_year: str = "2023-2024",
                            n_total: int = 5, max_plays: int = 10) -> Dict[str, Any]:
    """
    Compare training data formats across all available platforms.
    
    Args:
        game_id: Game ID to compare
        season_year: Season year
        n_total: Number of recent plays in context
        max_plays: Max plays to process for comparison
        
    Returns:
        dict: Comparison results for all platforms
    """
    return multi_platform_generator.compare_formats(game_id, season_year, n_total, max_plays)
