"""
Response validator for NBA prediction model outputs.
Ensures all model responses adhere to the expected JSON format and data constraints.
"""

import json
import re
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass


@dataclass
class ValidationError:
    """Represents a validation error with context."""
    field_path: str
    error_type: str
    expected: str
    actual: str
    message: str


class NBAResponseValidator:
    """Validates NBA prediction model responses for format consistency."""
    
    def __init__(self):
        self.errors: List[ValidationError] = []
    
    def validate_response(self, response_text: str, context: Dict[str, Any]) -> tuple[bool, List[ValidationError]]:
        """
        Validate a model response against expected format.
        
        Args:
            response_text: Raw response text from the model
            context: Original game context for validation reference
            
        Returns:
            (is_valid, validation_errors)
        """
        self.errors = []
        
        # Step 1: Parse JSON
        try:
            response_data = json.loads(response_text.strip())
        except json.JSONDecodeError as e:
            self.errors.append(ValidationError(
                field_path="root",
                error_type="json_parse_error",
                expected="valid JSON",
                actual=f"JSON parse error: {str(e)}",
                message=f"Response is not valid JSON: {str(e)}"
            ))
            return False, self.errors
        
        # Step 2: Validate top-level structure
        if not self._validate_top_level_structure(response_data):
            return False, self.errors
        
        next_play = response_data["next_play"]
        
        # Step 3: Validate next_play structure and types
        if not self._validate_next_play_structure(next_play):
            return False, self.errors
        
        # Step 4: Validate content against context
        if not self._validate_content_consistency(next_play, context):
            return False, self.errors
        
        return len(self.errors) == 0, self.errors
    
    def _validate_top_level_structure(self, data: Any) -> bool:
        """Validate top-level response structure."""
        if not isinstance(data, dict):
            self.errors.append(ValidationError(
                field_path="root",
                error_type="type_error",
                expected="object/dict",
                actual=str(type(data).__name__),
                message="Response must be a JSON object"
            ))
            return False
        
        if "next_play" not in data:
            self.errors.append(ValidationError(
                field_path="root",
                error_type="missing_field",
                expected="next_play field",
                actual="missing",
                message="Response must contain 'next_play' field"
            ))
            return False
        
        return True
    
    def _validate_next_play_structure(self, next_play: Any) -> bool:
        """Validate the next_play object structure and types."""
        if not isinstance(next_play, dict):
            self.errors.append(ValidationError(
                field_path="next_play",
                error_type="type_error",
                expected="object/dict",
                actual=str(type(next_play).__name__),
                message="next_play must be an object"
            ))
            return False
        
        # Required fields with expected types
        required_fields = {
            "quarter": int,
            "time_remaining": str,
            "score": str,
            "players_on_court": list,
            "player": (str, type(None)),
            "description": str,
            "shot_details": dict
        }
        
        for field, expected_type in required_fields.items():
            if field not in next_play:
                self.errors.append(ValidationError(
                    field_path=f"next_play.{field}",
                    error_type="missing_field",
                    expected=field,
                    actual="missing",
                    message=f"next_play must contain '{field}' field"
                ))
                continue
            
            value = next_play[field]
            if not isinstance(value, expected_type):
                self.errors.append(ValidationError(
                    field_path=f"next_play.{field}",
                    error_type="type_error",
                    expected=str(expected_type),
                    actual=str(type(value).__name__),
                    message=f"next_play.{field} must be {expected_type}"
                ))
        
        # Validate specific field formats
        if "time_remaining" in next_play:
            if not self._validate_time_format(next_play["time_remaining"]):
                self.errors.append(ValidationError(
                    field_path="next_play.time_remaining",
                    error_type="format_error",
                    expected="MM:SS format",
                    actual=next_play["time_remaining"],
                    message="time_remaining must be in MM:SS format"
                ))
        
        if "score" in next_play:
            if not self._validate_score_format(next_play["score"]):
                self.errors.append(ValidationError(
                    field_path="next_play.score",
                    error_type="format_error",
                    expected="TEAM1 XX - TEAM2 YY format",
                    actual=next_play["score"],
                    message="score must be in 'TEAM1 XX - TEAM2 YY' format"
                ))
        
        if "quarter" in next_play:
            if not isinstance(next_play["quarter"], int) or next_play["quarter"] < 1 or next_play["quarter"] > 4:
                self.errors.append(ValidationError(
                    field_path="next_play.quarter",
                    error_type="value_error",
                    expected="1, 2, 3, or 4",
                    actual=str(next_play["quarter"]),
                    message="quarter must be an integer between 1 and 4"
                ))
        
        # Validate players_on_court structure
        if "players_on_court" in next_play and isinstance(next_play["players_on_court"], list):
            if not self._validate_players_on_court(next_play["players_on_court"]):
                return False
        
        # Validate shot_details structure
        if "shot_details" in next_play and isinstance(next_play["shot_details"], dict):
            if not self._validate_shot_details(next_play["shot_details"]):
                return False
        
        return len(self.errors) == 0
    
    def _validate_players_on_court(self, players_on_court: List) -> bool:
        """Validate players_on_court structure."""
        if len(players_on_court) != 2:
            self.errors.append(ValidationError(
                field_path="next_play.players_on_court",
                error_type="length_error",
                expected="2 teams",
                actual=str(len(players_on_court)),
                message="players_on_court must contain exactly 2 teams"
            ))
            return False
        
        for i, team_data in enumerate(players_on_court):
            if not isinstance(team_data, dict):
                self.errors.append(ValidationError(
                    field_path=f"next_play.players_on_court[{i}]",
                    error_type="type_error",
                    expected="object/dict",
                    actual=str(type(team_data).__name__),
                    message=f"players_on_court[{i}] must be an object"
                ))
                continue
            
            if "team" not in team_data or "players" not in team_data:
                self.errors.append(ValidationError(
                    field_path=f"next_play.players_on_court[{i}]",
                    error_type="missing_field",
                    expected="team and players fields",
                    actual="missing",
                    message=f"players_on_court[{i}] must have 'team' and 'players' fields"
                ))
                continue
            
            if not isinstance(team_data["team"], str):
                self.errors.append(ValidationError(
                    field_path=f"next_play.players_on_court[{i}].team",
                    error_type="type_error",
                    expected="string",
                    actual=str(type(team_data["team"]).__name__),
                    message="team must be a string"
                ))
            
            if not isinstance(team_data["players"], list):
                self.errors.append(ValidationError(
                    field_path=f"next_play.players_on_court[{i}].players",
                    error_type="type_error",
                    expected="array/list",
                    actual=str(type(team_data["players"]).__name__),
                    message="players must be an array"
                ))
            elif len(team_data["players"]) != 5:
                self.errors.append(ValidationError(
                    field_path=f"next_play.players_on_court[{i}].players",
                    error_type="length_error",
                    expected="5 players",
                    actual=str(len(team_data["players"])),
                    message="players must contain exactly 5 player names"
                ))
        
        return len(self.errors) == 0
    
    def _validate_shot_details(self, shot_details: Dict) -> bool:
        """Validate shot_details structure and types."""
        expected_fields = {
            "team": (str, type(None)),
            "points": (int, type(None)),
            "x_coord": (float, int, type(None)),
            "y_coord": (float, int, type(None))
        }
        
        for field, expected_type in expected_fields.items():
            if field not in shot_details:
                self.errors.append(ValidationError(
                    field_path=f"next_play.shot_details.{field}",
                    error_type="missing_field",
                    expected=field,
                    actual="missing",
                    message=f"shot_details must contain '{field}' field"
                ))
                continue
            
            value = shot_details[field]
            if not isinstance(value, expected_type):
                self.errors.append(ValidationError(
                    field_path=f"next_play.shot_details.{field}",
                    error_type="type_error",
                    expected=str(expected_type),
                    actual=str(type(value).__name__),
                    message=f"shot_details.{field} must be {expected_type}"
                ))
        
        # Validate points range
        if "points" in shot_details and shot_details["points"] is not None:
            if not isinstance(shot_details["points"], int) or shot_details["points"] < 0 or shot_details["points"] > 3:
                self.errors.append(ValidationError(
                    field_path="next_play.shot_details.points",
                    error_type="value_error",
                    expected="0, 1, 2, or 3",
                    actual=str(shot_details["points"]),
                    message="points must be 0, 1, 2, or 3"
                ))
        
        return len(self.errors) == 0
    
    def _validate_content_consistency(self, next_play: Dict, context: Dict[str, Any]) -> bool:
        """Validate content consistency with input context."""
        # Extract valid team names and players
        away_team = context.get("away_team", {})
        home_team = context.get("home_team", {})
        
        valid_teams = {away_team.get("name"), home_team.get("name")}
        valid_teams.discard(None)
        
        valid_players = set()
        for team in [away_team, home_team]:
            for player in team.get("players", []):
                valid_players.add(player.get("name"))
        valid_players.discard(None)
        
        # Validate team names in score
        if "score" in next_play:
            score_teams = self._extract_teams_from_score(next_play["score"])
            for team in score_teams:
                if team not in valid_teams:
                    self.errors.append(ValidationError(
                        field_path="next_play.score",
                        error_type="content_error",
                        expected=f"teams from {valid_teams}",
                        actual=team,
                        message=f"Team '{team}' in score not found in context"
                    ))
        
        # Validate team names in players_on_court
        if "players_on_court" in next_play:
            for i, team_data in enumerate(next_play["players_on_court"]):
                if isinstance(team_data, dict) and "team" in team_data:
                    if team_data["team"] not in valid_teams:
                        self.errors.append(ValidationError(
                            field_path=f"next_play.players_on_court[{i}].team",
                            error_type="content_error",
                            expected=f"teams from {valid_teams}",
                            actual=team_data["team"],
                            message=f"Team '{team_data['team']}' not found in context"
                        ))
                
                # Validate player names
                if isinstance(team_data, dict) and "players" in team_data:
                    for j, player in enumerate(team_data["players"]):
                        if isinstance(player, str) and player not in valid_players:
                            self.errors.append(ValidationError(
                                field_path=f"next_play.players_on_court[{i}].players[{j}]",
                                error_type="content_error",
                                expected="valid player name from context",
                                actual=player,
                                message=f"Player '{player}' not found in context"
                            ))
        
        # Validate player in main play
        if "player" in next_play and next_play["player"] is not None:
            if next_play["player"] not in valid_players:
                self.errors.append(ValidationError(
                    field_path="next_play.player",
                    error_type="content_error",
                    expected="valid player name from context",
                    actual=next_play["player"],
                    message=f"Player '{next_play['player']}' not found in context"
                ))
        
        # Validate team in shot_details
        if "shot_details" in next_play and isinstance(next_play["shot_details"], dict):
            shot_team = next_play["shot_details"].get("team")
            if shot_team is not None and shot_team not in valid_teams:
                self.errors.append(ValidationError(
                    field_path="next_play.shot_details.team",
                    error_type="content_error",
                    expected=f"teams from {valid_teams}",
                    actual=shot_team,
                    message=f"Team '{shot_team}' in shot_details not found in context"
                ))
        
        return len(self.errors) == 0
    
    def _validate_time_format(self, time_str: str) -> bool:
        """Validate time format (MM:SS)."""
        pattern = r'^\d{1,2}:\d{2}$'
        if not re.match(pattern, time_str):
            return False
        
        parts = time_str.split(':')
        try:
            minutes = int(parts[0])
            seconds = int(parts[1])
            return 0 <= minutes <= 12 and 0 <= seconds <= 59
        except ValueError:
            return False
    
    def _validate_score_format(self, score_str: str) -> bool:
        """Validate score format (TEAM1 XX - TEAM2 YY)."""
        pattern = r'^[A-Z]{2,4} \d+ - [A-Z]{2,4} \d+$'
        return bool(re.match(pattern, score_str))
    
    def _extract_teams_from_score(self, score_str: str) -> List[str]:
        """Extract team names from score string."""
        pattern = r'^([A-Z]{2,4}) \d+ - ([A-Z]{2,4}) \d+$'
        match = re.match(pattern, score_str)
        if match:
            return [match.group(1), match.group(2)]
        return []
    
    def get_error_summary(self) -> str:
        """Get a formatted summary of all validation errors."""
        if not self.errors:
            return "No validation errors"
        
        summary = f"Found {len(self.errors)} validation errors:\n"
        for i, error in enumerate(self.errors, 1):
            summary += f"  {i}. {error.field_path}: {error.message}\n"
        
        return summary
