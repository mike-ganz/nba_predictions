#!/usr/bin/env python3
"""
NBA Season and Game ID Validation System

Provides utilities to validate game IDs against seasons and auto-detect 
seasons from game IDs to prevent mismatched configurations.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GameInfo:
    """Information about an NBA game."""
    game_id: str
    detected_season: str
    season_code: str
    game_number: int
    is_valid: bool
    error_message: Optional[str] = None


class SeasonGameValidator:
    """Validates relationships between NBA game IDs and seasons."""
    
    # NBA game ID patterns and their corresponding seasons
    SEASON_PATTERNS = {
        '021': '2020-2021',  # COVID season
        '022': '2021-2022', 
        '222': '2022-2023',  # Changed format starting 2022-2023
        '223': '2023-2024',
        '224': '2024-2025',
        '225': '2025-2026',  # Future seasons
    }
    
    def __init__(self):
        """Initialize the validator."""
        pass
    
    def parse_game_id(self, game_id: str) -> GameInfo:
        """
        Parse NBA game ID to extract season information.
        
        Args:
            game_id: NBA game ID (e.g., "22200001", "0022200001")
            
        Returns:
            GameInfo object with parsed information
        """
        # Clean the game ID
        clean_id = str(game_id).strip()
        
        # Handle different formats
        if len(clean_id) == 10:
            # Format: 0022200001 (with leading zeros)
            season_code = clean_id[2:5]
            game_number = clean_id[5:]
        elif len(clean_id) == 8:
            # Format: 22200001 (without leading zeros)
            season_code = clean_id[:3]
            game_number = clean_id[3:]
        else:
            return GameInfo(
                game_id=clean_id,
                detected_season="unknown",
                season_code="",
                game_number=0,
                is_valid=False,
                error_message=f"Invalid game ID format: {clean_id}"
            )
        
        # Map season code to season year
        if season_code in self.SEASON_PATTERNS:
            detected_season = self.SEASON_PATTERNS[season_code]
            is_valid = True
            error_message = None
        else:
            detected_season = "unknown"
            is_valid = False
            error_message = f"Unknown season code: {season_code}"
        
        try:
            game_num = int(game_number)
        except ValueError:
            game_num = 0
            is_valid = False
            error_message = f"Invalid game number: {game_number}"
        
        return GameInfo(
            game_id=clean_id,
            detected_season=detected_season,
            season_code=season_code,
            game_number=game_num,
            is_valid=is_valid,
            error_message=error_message
        )
    
    def validate_game_season_match(self, game_id: str, expected_season: str) -> Tuple[bool, str]:
        """
        Validate that a game ID belongs to the expected season.
        
        Args:
            game_id: NBA game ID
            expected_season: Expected season (e.g., "2023-2024")
            
        Returns:
            (is_valid, message) tuple
        """
        game_info = self.parse_game_id(game_id)
        
        if not game_info.is_valid:
            return False, game_info.error_message
        
        if game_info.detected_season != expected_season:
            return False, f"Game {game_id} belongs to {game_info.detected_season}, but expected {expected_season}"
        
        return True, "Game ID matches expected season"
    
    def auto_detect_seasons(self, game_ids: List[str]) -> Dict[str, List[str]]:
        """
        Auto-detect seasons for a list of game IDs and group them.
        
        Args:
            game_ids: List of NBA game IDs
            
        Returns:
            Dictionary mapping seasons to game IDs
        """
        season_groups = {}
        invalid_games = []
        
        for game_id in game_ids:
            game_info = self.parse_game_id(game_id)
            
            if game_info.is_valid:
                season = game_info.detected_season
                if season not in season_groups:
                    season_groups[season] = []
                season_groups[season].append(game_id)
            else:
                invalid_games.append((game_id, game_info.error_message))
        
        if invalid_games:
            print("⚠️ Invalid game IDs found:")
            for game_id, error in invalid_games:
                print(f"  • {game_id}: {error}")
        
        return season_groups
    
    def suggest_season_for_games(self, game_ids: List[str]) -> Optional[str]:
        """
        Suggest the most appropriate season for a list of game IDs.
        
        Args:
            game_ids: List of NBA game IDs
            
        Returns:
            Suggested season year or None if can't determine
        """
        season_groups = self.auto_detect_seasons(game_ids)
        
        if not season_groups:
            return None
        
        # Find the season with the most games
        most_common_season = max(season_groups.keys(), key=lambda k: len(season_groups[k]))
        
        return most_common_season
    
    def get_supported_seasons(self) -> List[str]:
        """Get list of supported season years."""
        return sorted(set(self.SEASON_PATTERNS.values()))
    
    def format_validation_report(self, game_ids: List[str], expected_season: str) -> str:
        """
        Generate a detailed validation report for game IDs against expected season.
        
        Args:
            game_ids: List of NBA game IDs to validate
            expected_season: Expected season year
            
        Returns:
            Formatted validation report
        """
        report_lines = []
        report_lines.append(f"🏀 Game ID Validation Report")
        report_lines.append(f"Expected Season: {expected_season}")
        report_lines.append(f"Games to Validate: {len(game_ids)}")
        report_lines.append("=" * 50)
        
        valid_count = 0
        invalid_count = 0
        season_mismatches = []
        
        for game_id in game_ids:
            is_valid, message = self.validate_game_season_match(game_id, expected_season)
            
            if is_valid:
                valid_count += 1
                report_lines.append(f"✅ {game_id}: Valid")
            else:
                invalid_count += 1
                game_info = self.parse_game_id(game_id)
                if game_info.is_valid and game_info.detected_season != expected_season:
                    season_mismatches.append((game_id, game_info.detected_season))
                report_lines.append(f"❌ {game_id}: {message}")
        
        report_lines.append("")
        report_lines.append("📊 Summary:")
        report_lines.append(f"  Valid: {valid_count}/{len(game_ids)}")
        report_lines.append(f"  Invalid: {invalid_count}/{len(game_ids)}")
        
        if season_mismatches:
            report_lines.append("")
            report_lines.append("🔄 Season Mismatches Found:")
            season_groups = {}
            for game_id, detected_season in season_mismatches:
                if detected_season not in season_groups:
                    season_groups[detected_season] = []
                season_groups[detected_season].append(game_id)
            
            for season, games in season_groups.items():
                report_lines.append(f"  {season}: {games}")
        
        return "\n".join(report_lines)


def main():
    """Example usage of the validator."""
    validator = SeasonGameValidator()
    
    # Example game IDs from different seasons
    test_games = [
        "22200001",  # 2022-2023
        "22200002",  # 2022-2023
        "22300001",  # 2023-2024
        "22300002",  # 2023-2024
        "invalid123", # Invalid
    ]
    
    print("🔍 Testing Game ID Parsing:")
    for game_id in test_games:
        info = validator.parse_game_id(game_id)
        print(f"  {game_id} → {info.detected_season} ({'✅' if info.is_valid else '❌'})")
    
    print("\n🔍 Auto-detecting Seasons:")
    season_groups = validator.auto_detect_seasons(test_games)
    for season, games in season_groups.items():
        print(f"  {season}: {games}")
    
    print("\n🔍 Validation Report:")
    report = validator.format_validation_report(test_games, "2023-2024")
    print(report)


if __name__ == "__main__":
    main()
