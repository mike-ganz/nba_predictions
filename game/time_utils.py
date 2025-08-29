"""
Time and game clock utilities for NBA game data processing.

This module handles all time-related calculations including game time remaining,
quarter time conversions, and period management.
"""

import pandas as pd
from typing import Tuple


class GameTimeCalculator:
    """Handles NBA game time calculations and conversions."""
    
    # NBA game constants
    MINUTES_PER_PERIOD = 12
    TOTAL_PERIODS = 4
    TOTAL_GAME_MINUTES = MINUTES_PER_PERIOD * TOTAL_PERIODS  # 48 minutes
    
    @staticmethod
    def calculate_game_time_remaining(period: int, remaining_time: str) -> str:
        """
        Calculate total game time remaining based on period and remaining time in current period.
        
        Args:
            period: Current period (1-4)
            remaining_time: Time remaining in current period (format: "0:MM:SS")
            
        Returns:
            str: Total game time remaining (format: "MM:SS")
        """
        try:
            # Parse remaining_time string (format: "0:MM:SS")
            if pd.isna(remaining_time) or not remaining_time:
                return "00:00"
                
            time_parts = str(remaining_time).split(':')
            if len(time_parts) >= 2:
                # Get minutes and seconds from the current period
                minutes = int(time_parts[-2])  # Second to last part is minutes
                seconds = int(time_parts[-1])   # Last part is seconds
            else:
                return "00:00"
            
            # Calculate total minutes remaining based on period
            if period == 1:
                total_minutes_remaining = 36 + minutes  # 3 full periods + current period remaining
            elif period == 2:
                total_minutes_remaining = 24 + minutes  # 2 full periods + current period remaining
            elif period == 3:
                total_minutes_remaining = 12 + minutes  # 1 full period + current period remaining
            elif period == 4:
                total_minutes_remaining = minutes       # Only current period remaining
            else:
                # Handle overtime or invalid periods
                total_minutes_remaining = minutes
                
            return f"{total_minutes_remaining:02d}:{seconds:02d}"
            
        except (ValueError, IndexError, TypeError):
            return "00:00"
    
    @staticmethod
    def convert_to_quarter_time(period: int, remaining_time: str) -> Tuple[int, str]:
        """
        Convert period and remaining_time to quarter and time_remaining for JSON format.
        
        Args:
            period: Current period (1-4)
            remaining_time: Time remaining in current period (format: "0:MM:SS")
            
        Returns:
            tuple: (quarter, time_remaining_in_quarter)
        """
        try:
            if pd.isna(remaining_time) or not remaining_time:
                return period, "00:00"
                
            time_parts = str(remaining_time).split(':')
            if len(time_parts) >= 2:
                minutes = int(time_parts[-2])
                seconds = int(time_parts[-1])
                time_in_quarter = f"{minutes:02d}:{seconds:02d}"
                return period, time_in_quarter
            else:
                return period, "00:00"
        except (ValueError, IndexError, TypeError):
            return period, "00:00"
    
    @staticmethod
    def parse_game_clock(time_string: str) -> Tuple[int, int]:
        """
        Parse a game clock time string into minutes and seconds.
        
        Args:
            time_string: Time string in various formats ("MM:SS", "0:MM:SS", etc.)
            
        Returns:
            tuple: (minutes, seconds)
        """
        try:
            if pd.isna(time_string) or not time_string:
                return 0, 0
                
            time_parts = str(time_string).split(':')
            if len(time_parts) >= 2:
                minutes = int(time_parts[-2])
                seconds = int(time_parts[-1])
                return minutes, seconds
            else:
                return 0, 0
        except (ValueError, IndexError, TypeError):
            return 0, 0
    
    @staticmethod
    def format_time(minutes: int, seconds: int) -> str:
        """
        Format minutes and seconds into standard time format.
        
        Args:
            minutes: Minutes component
            seconds: Seconds component
            
        Returns:
            str: Formatted time string "MM:SS"
        """
        return f"{minutes:02d}:{seconds:02d}"
    
    @staticmethod
    def is_overtime_period(period: int) -> bool:
        """
        Check if the given period is an overtime period.
        
        Args:
            period: Period number
            
        Returns:
            bool: True if overtime period (period > 4)
        """
        return period > 4
    
    @staticmethod
    def get_period_name(period: int) -> str:
        """
        Get the display name for a period.
        
        Args:
            period: Period number
            
        Returns:
            str: Period display name ("1st Quarter", "2nd Quarter", etc.)
        """
        if period == 1:
            return "1st Quarter"
        elif period == 2:
            return "2nd Quarter"
        elif period == 3:
            return "3rd Quarter"
        elif period == 4:
            return "4th Quarter"
        elif period > 4:
            ot_num = period - 4
            if ot_num == 1:
                return "Overtime"
            else:
                return f"{ot_num}OT"
        else:
            return f"Period {period}"


# Global instance for convenience
game_time = GameTimeCalculator()

# Convenience functions for backward compatibility
def calculate_game_time_remaining(period: int, remaining_time: str) -> str:
    """Calculate total game time remaining."""
    return game_time.calculate_game_time_remaining(period, remaining_time)


def convert_to_quarter_time(period: int, remaining_time: str) -> Tuple[int, str]:
    """Convert period and remaining_time to quarter and time_remaining."""
    return game_time.convert_to_quarter_time(period, remaining_time)
