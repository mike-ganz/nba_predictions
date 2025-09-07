"""
Response validator for NBA prediction model outputs.
Ensures all model responses adhere to the expected JSON format and data constraints.
"""

import json
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum


@dataclass
class ValidationError:
    """Represents a validation error with context."""
    field_path: str
    error_type: str
    expected: str
    actual: str
    message: str


@dataclass
class ValidationTermination:
    """Detailed information about why a validation caused simulation termination."""
    termination_type: str  # "game_ended", "rollback_time", "excessive_retries", etc.
    reason: str  # Human-readable reason
    trigger_condition: str  # Specific condition that triggered termination
    game_state: Dict[str, Any]  # Game state when termination occurred
    validation_context: Dict[str, Any]  # Validation-specific context
    termination_timestamp: str  # When termination was decided
    consecutive_count: int = 0  # For consecutive condition tracking
    total_attempts: int = 0  # Total attempts before termination


class ValidationResult(Enum):
    """Possible validation results."""
    VALID = "valid"
    RETRY = "retry"
    END_GAME = "end_game"
    ROLLBACK_TIME = "rollback_time"


class NBAResponseValidator:
    """Validates NBA prediction model responses for format consistency."""
    
    def __init__(self):
        print("🔧 Initializing NBAResponseValidator")
        self.errors: List[ValidationError] = []
        
        # State tracking for advanced validations
        self.response_history: List[str] = []  # Track raw responses for duplicate detection
        self.consecutive_subs: int = 0  # Track consecutive substitution responses
        self.consecutive_endgame: int = 0  # Track consecutive end-game scenarios
        self.consecutive_same_time: int = 0  # Track consecutive same time_remaining values
        self.last_time_remaining: Optional[str] = None  # Track the last time_remaining value
        self.max_history_size: int = 10  # Limit history size for memory management
        
        # Enhanced timestamp rollback functionality
        self.rollback_recent_plays_snapshot: Optional[List[Dict[str, Any]]] = None  # Snapshot of recent_plays before problematic timestamp
        self.problematic_timestamp: Optional[str] = None  # The timestamp that's causing issues
        
        # Termination tracking
        self.last_termination: Optional[ValidationTermination] = None  # Details about the most recent termination
        self.total_validation_attempts: int = 0  # Total validation attempts in this session
        self.retry_count: int = 0  # Count of retries for current validation sequence
        
        print("🔧 NBAResponseValidator initialized with clean state and termination tracking")
    
    def validate_response(self, response_text: str, context: Dict[str, Any]) -> Tuple[ValidationResult, List[ValidationError], str]:
        """
        Validate a model response against expected format with advanced game flow logic.
        
        Args:
            response_text: Raw response text from the model
            context: Original game context for validation reference
            
        Returns:
            (validation_result, validation_errors, reason)
        """
        self.errors = []
        self.total_validation_attempts += 1
        
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
            return ValidationResult.RETRY, self.errors, "JSON parse error"
        
        # Step 2: Validate top-level structure
        if not self._validate_top_level_structure(response_data):
            return ValidationResult.RETRY, self.errors, "Invalid top-level structure"
        
        next_play = response_data["next_play"]
        
        # Step 3: Validate next_play structure and types
        if not self._validate_next_play_structure(next_play):
            return ValidationResult.RETRY, self.errors, "Invalid next_play structure"
        
        # Step 4: Validate content against context
        if not self._validate_content_consistency(next_play, context):
            return ValidationResult.RETRY, self.errors, "Content inconsistency with context"
        
        # Step 5: Advanced game flow validations
        if len(self.errors) == 0:  # Only proceed if basic validation passed
            advanced_result, reason = self._validate_advanced_game_flow(response_text, next_play, context)
            if advanced_result != ValidationResult.VALID:
                # Create termination record if this is a terminating result
                if advanced_result in [ValidationResult.END_GAME, ValidationResult.ROLLBACK_TIME]:
                    self._create_termination_record(advanced_result, reason, next_play, context)
                elif advanced_result == ValidationResult.RETRY:
                    self.retry_count += 1
                return advanced_result, self.errors, reason
        
        # If we get here, all validations passed
        self.retry_count = 0  # Reset retry count on successful validation
        return ValidationResult.VALID, self.errors, "Response is valid"
    
    def _create_termination_record(self, result_type: ValidationResult, reason: str, next_play: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Create a detailed termination record for logging and analysis."""
        termination_type_map = {
            ValidationResult.END_GAME: "game_ended",
            ValidationResult.ROLLBACK_TIME: "rollback_time"
        }
        
        # Extract key game state information
        game_state = {
            "quarter": next_play.get("quarter"),
            "time_remaining": next_play.get("time_remaining"),
            "score": next_play.get("score"),
            "description": next_play.get("description", "")[:100],  # Truncate for readability
        }
        
        # Create validation-specific context
        validation_context = {
            "consecutive_same_time": self.consecutive_same_time,
            "consecutive_subs": self.consecutive_subs,
            "consecutive_endgame": self.consecutive_endgame,
            "last_time_remaining": self.last_time_remaining,
            "total_attempts": self.total_validation_attempts,
            "retry_count": self.retry_count,
            "response_history_length": len(self.response_history),
            "has_rollback_snapshot": self.rollback_recent_plays_snapshot is not None,
            "problematic_timestamp": self.problematic_timestamp
        }
        
        # Determine specific trigger condition
        trigger_condition = self._determine_trigger_condition(result_type)
        
        # Create the termination record
        self.last_termination = ValidationTermination(
            termination_type=termination_type_map.get(result_type, "unknown"),
            reason=reason,
            trigger_condition=trigger_condition,
            game_state=game_state,
            validation_context=validation_context,
            termination_timestamp=datetime.now().isoformat(),
            consecutive_count=max(self.consecutive_same_time, self.consecutive_subs, self.consecutive_endgame),
            total_attempts=self.total_validation_attempts
        )
        
        print(f"🛑 TERMINATION RECORD CREATED: {self.last_termination.termination_type}")
        print(f"   Reason: {reason}")
        print(f"   Trigger: {trigger_condition}")
        print(f"   Game State: Q{game_state.get('quarter')} {game_state.get('time_remaining')} - {game_state.get('score')}")
        print(f"   Validation Context: {self.consecutive_same_time} same_time, {self.consecutive_subs} subs, {self.total_validation_attempts} total attempts")
    
    def _determine_trigger_condition(self, result_type: ValidationResult) -> str:
        """Determine the specific condition that triggered the termination."""
        if result_type == ValidationResult.END_GAME:
            if self.consecutive_endgame >= 5:
                return f"consecutive_endgame_scenarios_limit_reached ({self.consecutive_endgame})"
            else:
                return "quarter_4_time_expired"
                
        elif result_type == ValidationResult.ROLLBACK_TIME:
            return f"consecutive_same_timestamp_limit_reached ({self.consecutive_same_time} times at '{self.last_time_remaining}')"
            
        return "unknown_trigger"
    
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
            # "x_coord": (float, int, type(None)),
            # "y_coord": (float, int, type(None))
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
    
    def _validate_advanced_game_flow(self, response_text: str, next_play: Dict[str, Any], context: Dict[str, Any]) -> Tuple[ValidationResult, str]:
        """
        Advanced game flow validations based on response patterns and game state.
        
        Args:
            response_text: Raw response text for duplicate detection
            next_play: Parsed next_play object
            context: Original game context for validation reference
            
        Returns:
            (validation_result, reason)
        """
        # Debug state information
        current_time = next_play.get("time_remaining", "N/A")
        description = next_play.get("description", "N/A")
        print(f"🔍 Validator state: time_count={self.consecutive_same_time}, last_time='{self.last_time_remaining}', current_time='{current_time}', subs_count={self.consecutive_subs}")
        print(f"🔍 Current play: {description[:60]}...")
        
        # Update response history
        self._update_response_history(response_text)
        
        # Check for duplicate responses (2nd time getting same response)
        if self._check_duplicate_response(response_text):
            return ValidationResult.RETRY, "Received same response 2 times consecutively - requesting new response"
        
        # Check for consecutive same time_remaining with rollback capability
        rollback_result = self._check_consecutive_same_time(next_play, context.get('recent_plays', []))
        if rollback_result == ValidationResult.ROLLBACK_TIME:
            return ValidationResult.ROLLBACK_TIME, "Too many plays with same timestamp - rolling back to previous game state"
        elif rollback_result == ValidationResult.RETRY:
            return ValidationResult.RETRY, "Same time_remaining for multiple consecutive responses - requesting time progression"
        
        # Check for excessive substitutions
        if self._check_excessive_substitutions(next_play):
            return ValidationResult.RETRY, "Too many consecutive substitutions - requesting different play type"
        
        # Check for game ending conditions
        game_end_result = self._check_game_ending_conditions(next_play)
        if game_end_result != ValidationResult.VALID:
            return game_end_result, "Game ending condition met"
        
        print(f"✅ Advanced validations passed for: {description[:60]}...")
        return ValidationResult.VALID, "Advanced validations passed"
    
    def _update_response_history(self, response_text: str) -> None:
        """Update the response history with size management."""
        self.response_history.append(response_text.strip())
        if len(self.response_history) > self.max_history_size:
            self.response_history.pop(0)
    
    def _check_duplicate_response(self, response_text: str) -> bool:
        """Check if we've received the same response 2 times consecutively."""
        if len(self.response_history) < 2:
            return False
        
        # Check if the last 2 responses are identical
        recent_responses = self.response_history[-2:]
        return all(r == response_text.strip() for r in recent_responses)
    
    def _check_consecutive_same_time(self, next_play: Dict[str, Any], current_recent_plays: List[Dict[str, Any]]) -> ValidationResult:
        """
        Check for consecutive responses with the same time_remaining.
        Returns ROLLBACK_TIME if too many consecutive same timestamps detected.
        """
        consecutive_responses_limit = 10
        current_time = next_play.get("time_remaining")
        
        # Skip validation if time_remaining is not a string (invalid format)
        if not isinstance(current_time, str):
            return ValidationResult.VALID
        
        if self.last_time_remaining == current_time:
            self.consecutive_same_time += 1
            print(f"🕒 Same time '{current_time}' count: {self.consecutive_same_time}/{consecutive_responses_limit}")
            
            # If this is the first time we're seeing a repeat of this timestamp, save a snapshot
            if self.consecutive_same_time == 2 and self.rollback_recent_plays_snapshot is None:
                # Find the last play with a different timestamp to roll back to
                rollback_plays = []
                for play in reversed(current_recent_plays):
                    if play.get("time_remaining") != current_time:
                        # Found a play with different timestamp - this is our rollback point
                        # Include this play and all plays after it (but before current problematic sequence)
                        rollback_index = current_recent_plays.index(play)
                        rollback_plays = current_recent_plays[:rollback_index + 1]
                        break
                
                self.rollback_recent_plays_snapshot = rollback_plays.copy()
                self.problematic_timestamp = current_time
                print(f"📸 Snapshot saved: {len(self.rollback_recent_plays_snapshot)} plays before timestamp '{current_time}' sequence")
                if rollback_plays:
                    last_good_time = rollback_plays[-1].get("time_remaining", "N/A")
                    print(f"🔄 Rollback point: Most recent play at time '{last_good_time}'")
            
            if self.consecutive_same_time >= consecutive_responses_limit:
                print(f"🚨 Time rollback validation triggered! Same time '{current_time}' for {self.consecutive_same_time} consecutive responses")
                print(f"🔄 Rolling back to snapshot with {len(self.rollback_recent_plays_snapshot or [])} plays")
                # Don't reset state here - let the caller handle the rollback
                return ValidationResult.ROLLBACK_TIME
                
        else:
            # Time has changed - reset all tracking
            if self.consecutive_same_time > 1:
                print(f"🕒 Time changed: '{self.last_time_remaining}' → '{current_time}' (clearing rollback tracking)")
                self._clear_rollback_state()
            else:
                print(f"🕒 Time changed: '{self.last_time_remaining}' → '{current_time}' (normal progression)")
            
            self.consecutive_same_time = 1  # First occurrence of new time
            self.last_time_remaining = current_time
        
        return ValidationResult.VALID
    
    def _check_excessive_substitutions(self, next_play: Dict[str, Any]) -> bool:
        """Check for 3 consecutive substitution plays."""
        description = next_play.get("description", "").upper()
        
        if "SUB" in description:
            self.consecutive_subs += 1
            print(f"🔄 Substitution detected: '{description[:50]}...' count: {self.consecutive_subs}/3")
            if self.consecutive_subs >= 3:
                print(f"🚨 Substitution validation triggered! {self.consecutive_subs} consecutive substitutions")
                # Reset counter and return retry
                self.consecutive_subs = 0
                return True
        else:
            # Reset counter if not a substitution
            if self.consecutive_subs > 0:
                print(f"🔄 Non-substitution play: '{description[:50]}...' (resetting sub counter from {self.consecutive_subs} to 0)")
            self.consecutive_subs = 0
        
        return False
    
    def _check_game_ending_conditions(self, next_play: Dict[str, Any]) -> ValidationResult:
        """Check for game ending conditions."""
        quarter = next_play.get("quarter")
        time_remaining = next_play.get("time_remaining", "")
        
        # Immediate end: Quarter 4 with 00:00 or negative time
        if quarter == 4 and self._is_time_expired(time_remaining):
            return ValidationResult.END_GAME
        
        # Track consecutive end-game scenarios
        if quarter == 4 and self._is_time_very_low(time_remaining):
            self.consecutive_endgame += 1
            if self.consecutive_endgame >= 5:
                return ValidationResult.END_GAME
        else:
            # Reset counter if not in end-game scenario
            self.consecutive_endgame = 0
        
        return ValidationResult.VALID
    
    def _is_time_expired(self, time_remaining: str) -> bool:
        """Check if time is 00:00 or less."""
        if not isinstance(time_remaining, str):
            return False
        
        try:
            parts = time_remaining.split(':')
            if len(parts) != 2:
                return False
            
            minutes = int(parts[0])
            seconds = int(parts[1])
            
            return minutes <= 0 and seconds <= 0
        except (ValueError, IndexError):
            return False
    
    def _is_time_very_low(self, time_remaining: str) -> bool:
        """Check if time is less than 00:05."""
        if not isinstance(time_remaining, str):
            return False
        
        try:
            parts = time_remaining.split(':')
            if len(parts) != 2:
                return False
            
            minutes = int(parts[0])
            seconds = int(parts[1])
            
            # Less than 5 seconds total
            total_seconds = minutes * 60 + seconds
            return total_seconds < 5
        except (ValueError, IndexError):
            return False
    
    def reset_state(self) -> None:
        """Reset all state tracking (useful for new games)."""
        self.response_history.clear()
        self.consecutive_subs = 0
        self.consecutive_endgame = 0
        self.consecutive_same_time = 0
        self.last_time_remaining = None
        self._clear_rollback_state()
        
        # Reset termination tracking
        self.last_termination = None
        self.total_validation_attempts = 0
        self.retry_count = 0
    
    def _clear_rollback_state(self) -> None:
        """Clear rollback-related state tracking."""
        self.rollback_recent_plays_snapshot = None
        self.problematic_timestamp = None
    
    def get_rollback_snapshot(self) -> Optional[List[Dict[str, Any]]]:
        """Get the saved rollback snapshot and clear rollback state."""
        snapshot = self.rollback_recent_plays_snapshot
        self._clear_rollback_state()
        return snapshot
    
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
    
    def get_termination_info(self) -> Optional[ValidationTermination]:
        """Get the most recent termination information."""
        return self.last_termination
    
    def has_termination_record(self) -> bool:
        """Check if there's a termination record available."""
        return self.last_termination is not None
    
    def get_termination_summary(self) -> str:
        """Get a formatted summary of the termination information."""
        if not self.last_termination:
            return "No termination record available"
        
        term = self.last_termination
        
        summary = f"🛑 VALIDATION TERMINATION SUMMARY\n"
        summary += f"{'='*50}\n"
        summary += f"Termination Type: {term.termination_type}\n"
        summary += f"Reason: {term.reason}\n"
        summary += f"Trigger Condition: {term.trigger_condition}\n"
        summary += f"Timestamp: {term.termination_timestamp}\n"
        
        summary += f"\nGame State at Termination:\n"
        summary += f"  Quarter: {term.game_state.get('quarter', 'N/A')}\n"
        summary += f"  Time: {term.game_state.get('time_remaining', 'N/A')}\n"
        summary += f"  Score: {term.game_state.get('score', 'N/A')}\n"
        summary += f"  Description: {term.game_state.get('description', 'N/A')}\n"
        
        summary += f"\nValidation Context:\n"
        vc = term.validation_context
        summary += f"  Consecutive Same Time: {vc.get('consecutive_same_time', 0)}\n"
        summary += f"  Consecutive Substitutions: {vc.get('consecutive_subs', 0)}\n"
        summary += f"  Consecutive Endgame: {vc.get('consecutive_endgame', 0)}\n"
        summary += f"  Total Validation Attempts: {vc.get('total_attempts', 0)}\n"
        summary += f"  Retry Count: {vc.get('retry_count', 0)}\n"
        summary += f"  Response History Length: {vc.get('response_history_length', 0)}\n"
        summary += f"  Had Rollback Snapshot: {vc.get('has_rollback_snapshot', False)}\n"
        summary += f"  Problematic Timestamp: {vc.get('problematic_timestamp', 'None')}\n"
        
        summary += f"\nStatistics:\n"
        summary += f"  Consecutive Count: {term.consecutive_count}\n"
        summary += f"  Total Attempts: {term.total_attempts}\n"
        
        summary += f"{'='*50}\n"
        
        return summary
    
    def get_termination_for_database(self) -> Dict[str, Any]:
        """Get termination information formatted for database storage."""
        if not self.last_termination:
            return {}
        
        term = self.last_termination
        
        return {
            "validation_termination_type": term.termination_type,
            "validation_termination_reason": term.reason,
            "validation_trigger_condition": term.trigger_condition,
            "validation_termination_timestamp": term.termination_timestamp,
            "validation_consecutive_count": term.consecutive_count,
            "validation_total_attempts": term.total_attempts,
            "validation_game_state_quarter": term.game_state.get("quarter"),
            "validation_game_state_time": term.game_state.get("time_remaining"),
            "validation_game_state_score": term.game_state.get("score"),
            "validation_context_json": json.dumps(term.validation_context)
        }
