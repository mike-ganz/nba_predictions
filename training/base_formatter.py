"""
Base formatter interface for multi-platform fine-tuning data generation.

This module provides the abstract base class for training data formatters,
enabling support for multiple platforms (OpenAI, Gemini, etc.) with a
consistent interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd


class BaseFormatter(ABC):
    """Abstract base class for training data formatters."""
    
    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return the name of the platform this formatter targets."""
        pass
    
    @abstractmethod
    def create_training_data(self, df: pd.DataFrame, generation_mode: str = "remaining_plays") -> List[Dict[str, Any]]:
        """
        Convert structured training data into platform-specific fine-tuning format.
        
        Args:
            df: DataFrame with 'json_training_data' column
            generation_mode: "remaining_plays" or "first_N_plays"
            
        Returns:
            list: List of training examples in platform-specific format
        """
        pass
    
    @abstractmethod
    def validate_training_examples(self, training_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate training examples for platform-specific requirements.
        
        Args:
            training_examples: List of training examples to validate
            
        Returns:
            dict: Validation results with errors, warnings, and statistics
        """
        pass
    
    @abstractmethod
    def save_training_data(self, training_examples: List[Dict[str, Any]], 
                          filename: Optional[str] = None, 
                          season_year: Optional[str] = None) -> str:
        """
        Save training examples in platform-specific format.
        
        Args:
            training_examples: List of training examples
            filename: Custom filename (platform suffix will be added)
            season_year: Season year for filename
            
        Returns:
            str: Path to saved file
        """
        pass
    
    def get_file_suffix(self) -> str:
        """
        Get the file suffix for this platform's training files.
        
        Returns:
            str: File suffix (e.g., '_openai', '_gemini')
        """
        return f"_{self.platform_name.lower()}"
    
    def get_file_extension(self) -> str:
        """
        Get the file extension for this platform's training files.
        
        Returns:
            str: File extension (default: '.jsonl')
        """
        return '.jsonl'
    
    @abstractmethod
    def _create_training_example(self, game_df: pd.DataFrame, current_index: int) -> Optional[Dict[str, Any]]:
        """
        Create a single training example from current and next plays.
        
        Args:
            game_df: DataFrame for a single game, sorted by play_id
            current_index: Index of current play
            
        Returns:
            dict or None: Platform-specific training example or None if creation failed
        """
        pass
    
    @abstractmethod
    def _create_first_n_plays_example(self, game_df: pd.DataFrame, context_row: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Create a single first-N-plays training example.
        
        Args:
            game_df: DataFrame for a single game
            context_row: Row containing the game context without recent_plays
            
        Returns:
            dict or None: Platform-specific training example or None if creation failed
        """
        pass


class FormatterFactory:
    """Factory for creating platform-specific formatters."""
    
    _formatters = {}
    
    @classmethod
    def register_formatter(cls, platform_name: str, formatter_class: type):
        """
        Register a formatter class for a specific platform.
        
        Args:
            platform_name: Name of the platform (e.g., 'openai', 'gemini')
            formatter_class: Formatter class implementing BaseFormatter
        """
        cls._formatters[platform_name.lower()] = formatter_class
    
    @classmethod
    def create_formatter(cls, platform_name: str) -> BaseFormatter:
        """
        Create a formatter instance for the specified platform.
        
        Args:
            platform_name: Name of the platform
            
        Returns:
            BaseFormatter: Formatter instance for the platform
            
        Raises:
            ValueError: If platform is not supported
        """
        platform_key = platform_name.lower()
        if platform_key not in cls._formatters:
            available = list(cls._formatters.keys())
            raise ValueError(f"Unsupported platform '{platform_name}'. Available platforms: {available}")
        
        return cls._formatters[platform_key]()
    
    @classmethod
    def get_available_platforms(cls) -> List[str]:
        """
        Get list of available platform names.
        
        Returns:
            list: List of registered platform names
        """
        return list(cls._formatters.keys())
    
    @classmethod
    def is_platform_supported(cls, platform_name: str) -> bool:
        """
        Check if a platform is supported.
        
        Args:
            platform_name: Name of the platform
            
        Returns:
            bool: True if platform is supported, False otherwise
        """
        return platform_name.lower() in cls._formatters
