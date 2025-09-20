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

# Pre-compile regex patterns for performance (avoid recompilation)
TIME_FORMAT_REGEX = re.compile(r'^\d{1,2}:\d{2}$')
SCORE_PATTERNS = [
    re.compile(r'^[A-Z]{2,15} \d+ - [A-Z]{2,15} \d+$'),           # LAL 108 - BOS 102
    re.compile(r'^[A-Z]{2,15} \d+, [A-Z]{2,15} \d+$'),            # LAL 108, BOS 102
    re.compile(r'^[A-Z]{2,15}: \d+ [A-Z]{2,15}: \d+$'),           # LAL: 108 BOS: 102
    re.compile(r'^[A-Z]{2,15} \d+ [A-Z]{2,15} \d+$'),             # LAL 108 BOS 102
    re.compile(r'^[A-Z]{2,15}\s*\d+\s*[-,]\s*[A-Z]{2,15}\s*\d+$') # Flexible spacing
]

# Fast lookups for common validations
GAME_END_TIMES = frozenset(["0:00", "00:00", "0", "00"])
QUARTER_4 = 4


@dataclass(slots=True)  # Use slots for memory efficiency
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
    QUARTER_TRANSITION = "quarter_transition"


class NBAResponseValidator:
    """Validates NBA prediction model responses for format consistency."""
    
    def __init__(self, validation_mode: str = "fast"):
        """
        Initialize validator with configurable validation mode.
        
        Args:
            validation_mode: "fast" (essential only), "normal" (balanced), "strict" (all checks)
        """
        self.validation_mode = validation_mode
        # Initializing NBAResponseValidator
        
        self.errors: List[ValidationError] = []
        
        # Always track essential state regardless of mode
        self.total_validation_attempts: int = 0
        self.retry_count: int = 0
        self.last_termination: Optional[ValidationTermination] = None
        
        # Initialize advanced state tracking only for normal/strict modes
        if validation_mode in ["normal", "strict"]:
            self._init_advanced_tracking()
        else:
            self._init_minimal_tracking()
        
        # Performance optimization caches
        self._init_performance_caches()
        
        # NBAResponseValidator initialized
    
    def _convert_compact_response_for_validation(self, compact_response: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert compact format response to verbose format for validation.
        
        Args:
            compact_response: Response with "y" field containing play tuple(s)
            context: Game context for reference
            
        Returns:
            Dict with "next_play" field in verbose format
        """
        try:
            y_data = compact_response.get("y")
            if not y_data:
                return {"error": "No 'y' field in compact response"}
            
            # Handle both single play tuple and array of play tuples
            if isinstance(y_data, list) and len(y_data) > 0:
                if isinstance(y_data[0], list):
                    # Multiple plays (e.g., offensive foul + turnover)
                    primary_play = y_data[0]
                else:
                    # Single play tuple
                    primary_play = y_data
            else:
                return {"error": "Invalid 'y' data structure"}
            
            # Convert play tuple to verbose format
            if len(primary_play) < 6:
                return {"error": f"Play tuple too short: {len(primary_play)} elements"}
            
            quarter = primary_play[0]
            time_seconds = primary_play[1]
            score_array = primary_play[2] if len(primary_play[2]) >= 2 else [0, 0]
            actor = primary_play[3]
            event_code = primary_play[4]
            
            # Handle both scoring and non-scoring formats
            if len(primary_play) == 7:  # Scoring: [q, t, score, actor, event, pts, lineup_id]
                points = primary_play[5]
                lineup_id = primary_play[6]
            else:  # Non-scoring: [q, t, score, actor, event, lineup_id]
                points = None
                lineup_id = primary_play[5]
            
            # Convert time back to MM:SS format
            minutes = time_seconds // 60
            seconds = time_seconds % 60
            time_remaining = f"{minutes:02d}:{seconds:02d}"
            
            # Create verbose format next_play
            next_play = {
                "quarter": quarter,
                "time_remaining": time_remaining,
                "description": f"Predicted {event_code}",
                "score": f"AWAY {score_array[0]} - HOME {score_array[1]}",
                "shot_details": {
                    "team": actor[0] if points else None,
                    "points": points
                }
            }
            
            return {"next_play": next_play}
            
        except Exception as e:
            return {"error": f"Conversion failed: {e}"}
    
    def _init_minimal_tracking(self):
        """Initialize minimal state tracking for fast mode."""
        # Only track absolute essentials for fast validation
        self.consecutive_same_time: int = 0
        self.last_time_remaining: Optional[str] = None
        self.consecutive_endgame: int = 0
        
        # Essential rollback functionality (required even in fast mode)
        self.rollback_recent_plays_snapshot: Optional[List[Dict[str, Any]]] = None  # Snapshot of recent_plays before problematic timestamp
        self.problematic_timestamp: Optional[str] = None  # The timestamp that's causing issues
        
    def _init_advanced_tracking(self):
        """Initialize full state tracking for normal/strict modes."""
        # State tracking for advanced validations
        self.response_history: List[str] = []  # Track raw responses for duplicate detection
        self.consecutive_subs: int = 0  # Track consecutive substitution responses
        self.consecutive_endgame: int = 0  # Track consecutive end-game scenarios
        self.consecutive_same_time: int = 0  # Track consecutive same time_remaining values
        self.last_time_remaining: Optional[str] = None  # Track the last time_remaining value
        self.max_history_size: int = 10  # Limit history size for memory management
        
        # Time progression tracking
        self.last_quarter: Optional[int] = None  # Track the last quarter
        self.last_time_seconds: Optional[int] = None  # Track last time in seconds for comparison
        self.time_progression_violations: int = 0  # Track backward time jumps
        
        # Score progression tracking  
        self.last_team_scores: Optional[Dict[str, int]] = None  # Track team scores {team_name: score}
        self.score_progression_violations: int = 0  # Track score decreases
        
        # Quarter transition tracking
        self.quarter_transition_violations: int = 0  # Track improper quarter transitions
        
        # Game situation tracking
        self.situation_violations: int = 0  # Track unrealistic game situation responses
        
        # Description consistency tracking
        self.description_consistency_violations: int = 0  # Track description-data mismatches
        
        # Enhanced duplicate detection tracking
        self.response_patterns: Dict[str, int] = {}  # Track response patterns and counts
        self.near_duplicate_count: int = 0  # Track near-duplicate responses
        
        # Enhanced timestamp rollback functionality
        self.rollback_recent_plays_snapshot: Optional[List[Dict[str, Any]]] = None  # Snapshot of recent_plays before problematic timestamp
        self.problematic_timestamp: Optional[str] = None  # The timestamp that's causing issues
    
    def _init_performance_caches(self):
        """Initialize performance optimization caches."""
        # Cache for repeated string operations
        self._score_validation_cache: Dict[str, bool] = {}  # Cache score format validation results
        self._time_validation_cache: Dict[str, bool] = {}   # Cache time format validation results  
        self._cache_max_size = 100  # Limit cache growth
        
        # Pre-compile common error messages to avoid string operations
        self._error_messages = {
            'dict_required': "next_play must be a dictionary",
            'description_missing': "next_play must contain 'description' field", 
            'description_empty': "description must be a non-empty string",
            'json_object_required': "Response must be a JSON object",
            'next_play_missing': "Response must contain 'next_play' field"
        }
    
    def validate_response(self, response_text: str, context: Dict[str, Any]) -> Tuple[ValidationResult, List[ValidationError], str]:
        """
        Validate a model response against expected format with configurable validation depth.
        Supports both verbose and compact schema formats.
        
        Args:
            response_text: Raw response text from the model
            context: Original game context for validation reference
            
        Returns:
            (validation_result, validation_errors, reason)
        """
        self.errors = []
        self.total_validation_attempts += 1
        
        # DEBUG: Log validation entry
        current_time = context.get('recent_plays', [{}])[-1].get('time_remaining', 'N/A') if context.get('recent_plays') else 'N/A'
        
        # Step 1: Optimized JSON parsing
        try:
            # Fast path: try ujson if available, fallback to standard json
            stripped_text = response_text.strip()  # Cache stripped version
            try:
                import ujson
                response_data = ujson.loads(stripped_text)
            except (ImportError, AttributeError):
                response_data = json.loads(stripped_text)
        except (json.JSONDecodeError, ValueError) as e:
            # Optimized error creation (avoid f-strings in hot path)
            error_msg = "Response is not valid JSON: " + str(e)
            self.errors.append(ValidationError(
                field_path="root",
                error_type="json_parse_error",
                expected="valid JSON",
                actual="parse_error",  # Avoid expensive string formatting
                message=error_msg
            ))
            # Debug: print raw response when in debug mode
            try:
                import os
                dbg = int(os.getenv("PREDICTION_LOG_LEVEL", "1")) >= 3 or os.getenv("VALIDATION_DEBUG", "0").lower() in ("1","true","yes","on")
                if dbg:
                    print("=== DEBUG RAW (JSON parse error) ===")
                    try:
                        print(stripped_text)
                    except Exception:
                        print("<non-printable response content>")
                    print("=== END DEBUG RAW ===")
            except Exception:
                pass
            return ValidationResult.RETRY, self.errors, "JSON parse error"
        
        # Step 1.5: Check if this is raw compact tuple or compact format and convert if needed
        if isinstance(response_data, list) and len(response_data) >= 6:
            # This looks like a raw compact tuple [q, t, score, actor, event, ...], wrap it
            try:
                response_data = {"y": response_data}
            except Exception as e:
                self.errors.append(ValidationError(
                    field_path="root",
                    error_type="compact_format_error", 
                    expected="valid compact play tuple",
                    actual="raw_tuple_wrap_failed",
                    message=f"Failed to wrap raw compact tuple: {e}"
                ))
                return ValidationResult.RETRY, self.errors, "Raw compact tuple wrap error"
        
        if "y" in response_data and "next_play" not in response_data:
            # This looks like compact format, convert to verbose for validation
            try:
                response_data = self._convert_compact_response_for_validation(response_data, context)
                if "error" in response_data:
                    self.errors.append(ValidationError(
                        field_path="y",
                        error_type="compact_format_error",
                        expected="valid compact play tuple",
                        actual="conversion_failed",
                        message=response_data["error"]
                    ))
                    # Debug: show offending compact payload
                    try:
                        import os
                        dbg = int(os.getenv("PREDICTION_LOG_LEVEL", "1")) >= 3 or os.getenv("VALIDATION_DEBUG", "0").lower() in ("1","true","yes","on")
                        if dbg:
                            print("=== DEBUG COMPACT PAYLOAD (conversion_failed) ===")
                            print(response_text)
                            print("=== END COMPACT PAYLOAD ===")
                    except Exception:
                        pass
                    return ValidationResult.RETRY, self.errors, "Compact format conversion error"
            except Exception as e:
                self.errors.append(ValidationError(
                    field_path="y",
                    error_type="compact_format_error",
                    expected="valid compact format",
                    actual="conversion_exception",
                    message=f"Failed to convert compact format: {e}"
                ))
                try:
                    import os
                    dbg = int(os.getenv("PREDICTION_LOG_LEVEL", "1")) >= 3 or os.getenv("VALIDATION_DEBUG", "0").lower() in ("1","true","yes","on")
                    if dbg:
                        print("=== DEBUG COMPACT PAYLOAD (conversion_exception) ===")
                        print(response_text)
                        print("=== END COMPACT PAYLOAD ===")
                except Exception:
                    pass
                return ValidationResult.RETRY, self.errors, "Compact format exception"
        
        # Step 2: Validate top-level structure (always required)
        if not self._validate_top_level_structure(response_data):
            return ValidationResult.RETRY, self.errors, "Invalid top-level structure"
        
        next_play = response_data["next_play"]
        
        # Step 3: Validate next_play structure (always required)
        if self.validation_mode == "fast":
            if not self._validate_next_play_structure_fast(next_play):
                return ValidationResult.RETRY, self.errors, "Invalid next_play structure"
        else:
            if not self._validate_next_play_structure(next_play):
                return ValidationResult.RETRY, self.errors, "Invalid next_play structure"
        
        # Fast mode: Skip expensive validations and only do essential checks
        if self.validation_mode == "fast":
            return self._validate_fast_path(response_text, next_play, context)
        
        # Normal/Strict modes: Continue with full validation
        # Step 4: Validate content against context
        if self.validation_mode == "strict" and not self._validate_content_consistency(next_play, context):
            return ValidationResult.RETRY, self.errors, "Content inconsistency with context"
        
        # Step 5: Advanced game flow validations
        if len(self.errors) == 0:  # Only proceed if basic validation passed
            advanced_result, reason = self._validate_advanced_game_flow(response_text, next_play, context)
            if advanced_result != ValidationResult.VALID:
                # Create termination record if this is a terminating result
                if advanced_result in [ValidationResult.END_GAME, ValidationResult.ROLLBACK_TIME, ValidationResult.QUARTER_TRANSITION]:
                    self._create_termination_record(advanced_result, reason, next_play, context)
                elif advanced_result == ValidationResult.RETRY:
                    self.retry_count += 1
                return advanced_result, self.errors, reason
        
        # If we get here, all validations passed
        self.retry_count = 0  # Reset retry count on successful validation
        return ValidationResult.VALID, self.errors, "Response is valid"
    
    def _create_termination_record(self, result_type: ValidationResult, reason: str, next_play: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Create a detailed termination record for logging and analysis."""
        print(f"DEBUG: _create_termination_record called with result_type={result_type}")
        
        termination_type_map = {
            ValidationResult.END_GAME: "game_ended",
            ValidationResult.ROLLBACK_TIME: "rollback_time",
            ValidationResult.QUARTER_TRANSITION: "quarter_transition"
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
        
        print(f"TERMINATION RECORD CREATED: {self.last_termination.termination_type}")
        print(f"   Reason: {reason}")
        print(f"   Trigger: {trigger_condition}")
        print(f"   Game State: Q{game_state.get('quarter')} {game_state.get('time_remaining')} - {game_state.get('score')}")
        print(f"   Validation Context: {self.consecutive_same_time} same_time, {self.consecutive_subs} subs, {self.total_validation_attempts} total attempts")
        
        print(f"DEBUG: Termination record stored, has_termination_record()={self.has_termination_record()}")
        print(f"DEBUG: get_termination_for_database()={len(str(self.get_termination_for_database()))} chars")
    
    def _determine_trigger_condition(self, result_type: ValidationResult) -> str:
        """Determine the specific condition that triggered the termination."""
        if result_type == ValidationResult.END_GAME:
            if self.consecutive_endgame >= 5:
                return f"consecutive_endgame_scenarios_limit_reached ({self.consecutive_endgame})"
            else:
                return "quarter_4_time_expired"
                
        elif result_type == ValidationResult.ROLLBACK_TIME:
            return f"consecutive_same_timestamp_limit_reached ({self.consecutive_same_time} times at '{self.last_time_remaining}')"
            
        elif result_type == ValidationResult.QUARTER_TRANSITION:
            quarter = getattr(self, 'quarter_transition_target', 'unknown')
            return f"end_of_quarter_administrative_pattern_detected (transitioning to Q{quarter})"
            
        return "unknown_trigger"
    
    def _validate_top_level_structure(self, data: Any) -> bool:
        """Validate top-level response structure (optimized with caching)."""
        # Fast type check with minimal string operations
        if not isinstance(data, dict):
            self.errors.append(ValidationError(
                field_path="root",
                error_type="type_error",
                expected="dict",
                actual=type(data).__name__,  # Avoid str() call
                message=self._error_messages['json_object_required']  # Pre-compiled message
            ))
            return False
        
        # Direct membership test (faster than .get())
        if "next_play" not in data:
            self.errors.append(ValidationError(
                field_path="root",
                error_type="missing_field",
                expected="next_play",
                actual="missing",
                message=self._error_messages['next_play_missing']  # Pre-compiled message
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
        # Time format validation (relaxed - allow various formats)
        if "time_remaining" in next_play:
            time_val = next_play["time_remaining"]
            # Allow any string format for now - games might use different time representations
            if not isinstance(time_val, str) or not time_val.strip():
                self.errors.append(ValidationError(
                    field_path="next_play.time_remaining",
                    error_type="format_error",
                    expected="non-empty time string",
                    actual=time_val,
                    message="time_remaining must be a non-empty string"
                ))
        
        if "score" in next_play:
            if not self._validate_score_format(next_play["score"]):
                self.errors.append(ValidationError(
                    field_path="next_play.score",
                    error_type="format_error",
                    expected="team score formats like 'LAL 108 - BOS 102' or 'Lakers 108, Celtics 102'",
                    actual=next_play["score"],
                    message="score must be in team-score format (various formats accepted)"
                ))
        
        if "quarter" in next_play:
            if not isinstance(next_play["quarter"], int) or next_play["quarter"] < 1 or next_play["quarter"] > 10:
                self.errors.append(ValidationError(
                    field_path="next_play.quarter",
                    error_type="value_error",
                    expected="1-4 (regulation) or 5-10 (overtime)",
                    actual=str(next_play["quarter"]),
                    message="quarter must be an integer between 1 and 10 (1-4 for regulation, 5+ for overtime)"
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
        
        # Validate team names in score (disabled - too restrictive)
        # Team names in scores can have different abbreviations/formats
        pass
        
        # Validate team names in players_on_court (disabled - too restrictive)
        # Team names can have abbreviations, different formats, etc.
        # This was causing too many false positives
        pass
        
        # Validate player in main play (disabled - too restrictive for name variations)
        # Player names can have many variations, abbreviations, nicknames
        # This validation was causing too many false positives
        pass
        
        # Validate team in shot_details (disabled - too restrictive)
        # Team names can have different formats/abbreviations
        pass
        
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
        print(f"Validator state: time_count={self.consecutive_same_time}, last_time='{self.last_time_remaining}', current_time='{current_time}', subs_count={self.consecutive_subs}")
        print(f"Current play: {description[:60]}...")
        
        # Update response history
        self._update_response_history(response_text)
        
        # Check for duplicate and near-duplicate responses
        duplicate_result, duplicate_reason = self._check_enhanced_duplicates(response_text, next_play)
        if duplicate_result != ValidationResult.VALID:
            return duplicate_result, duplicate_reason
        
        # Check for consecutive same time_remaining with rollback capability
        rollback_result = self._check_consecutive_same_time(next_play, context.get('recent_plays', []))
        if rollback_result == ValidationResult.ROLLBACK_TIME:
            return ValidationResult.ROLLBACK_TIME, "Too many plays with same timestamp - rolling back to previous game state"
        elif rollback_result == ValidationResult.QUARTER_TRANSITION:
            return ValidationResult.QUARTER_TRANSITION, "Period detected - transitioning to next quarter"
        elif rollback_result == ValidationResult.RETRY:
            return ValidationResult.RETRY, "Same time_remaining for multiple consecutive responses - requesting time progression"
        
        # Check for excessive substitutions
        if self._check_excessive_substitutions(next_play):
            return ValidationResult.RETRY, "Too many consecutive substitutions - requesting different play type"
        
        # Check for game ending conditions
        game_end_result = self._check_game_ending_conditions(next_play)
        if game_end_result != ValidationResult.VALID:
            return game_end_result, "Game ending condition met"
        
        # Check time progression within quarters
        time_progression_result = self._check_time_progression(next_play)
        if time_progression_result != ValidationResult.VALID:
            return time_progression_result, "Invalid time progression within quarter"
        
        # Check score progression (scores should not decrease)
        score_progression_result = self._check_score_progression(next_play)
        if score_progression_result != ValidationResult.VALID:
            return score_progression_result, "Invalid score progression - scores decreased"
        
        # Check score-shot consistency (score changes should match shot_details.points)
        shot_consistency_result = self._check_score_shot_consistency(next_play)
        if shot_consistency_result != ValidationResult.VALID:
            return shot_consistency_result, "Score change inconsistent with shot points"
        
        # Check for statistical impossibilities
        impossibility_result = self._check_statistical_impossibilities(next_play)
        if impossibility_result != ValidationResult.VALID:
            return impossibility_result, "Statistically impossible scenario detected"
        
        # Check quarter transition logic
        quarter_transition_result = self._check_quarter_transitions(next_play)
        if quarter_transition_result != ValidationResult.VALID:
            return quarter_transition_result, "Invalid quarter transition logic"
        
        # Check game situation awareness
        situation_result = self._check_game_situation_awareness(next_play)
        if situation_result != ValidationResult.VALID:
            return situation_result, "Play doesn't match game situation context"
        
        # Check description consistency with data
        description_result = self._check_description_consistency(next_play)
        if description_result != ValidationResult.VALID:
            return description_result, "Description inconsistent with play data"
        
        print(f"Advanced validations passed for: {description[:60]}...")
        return ValidationResult.VALID, "Advanced validations passed"
    
    def _update_response_history(self, response_text: str) -> None:
        """Update the response history with size management."""
        self.response_history.append(response_text.strip())
        if len(self.response_history) > self.max_history_size:
            self.response_history.pop(0)
    
    def _check_enhanced_duplicates(self, response_text: str, next_play: Dict[str, Any]) -> Tuple[ValidationResult, str]:
        """Enhanced duplicate detection including exact, near, and pattern-based duplicates."""
        
        # 1. Check exact duplicates (original logic)
        if len(self.response_history) >= 3:
            recent_responses = self.response_history[-3:]
            if all(r == response_text.strip() for r in recent_responses):
                return ValidationResult.RETRY, "Received exact same response 3 times consecutively"
                
        # 2. Check near-duplicates (similar but not identical)
        normalized_response = self._normalize_response_for_comparison(response_text)
        near_duplicate_count = 0
        
        for historical_response in self.response_history[-5:]:  # Check last 5 responses
            normalized_historical = self._normalize_response_for_comparison(historical_response)
            similarity = self._calculate_response_similarity(normalized_response, normalized_historical)
            
            if similarity > 0.85:  # 85% similarity threshold
                near_duplicate_count += 1
        
        if near_duplicate_count >= 3:  # Require more near-duplicates
            self.near_duplicate_count += 1
            print(f"Near-duplicate response #{self.near_duplicate_count}: {similarity:.1%} similar to recent responses")
            
            if self.near_duplicate_count >= 5:  # Be much more lenient
                print(f"🚨 Near-duplicate validation triggered! {self.near_duplicate_count} near-duplicates")
                self.near_duplicate_count = 0
                return ValidationResult.RETRY, "Too many near-duplicate responses detected"
        
        # 3. Check semantic patterns (same key data, different descriptions)
        description = next_play.get("description", "")
        play_pattern = self._extract_play_pattern(next_play)
        
        if play_pattern in self.response_patterns:
            self.response_patterns[play_pattern] += 1
            pattern_count = self.response_patterns[play_pattern]
            
            if pattern_count >= 8:  # Same pattern repeated 8+ times (more lenient)
                print(f"Pattern repetition: '{play_pattern}' seen {pattern_count} times")
                
                if pattern_count >= 12:  # Much more lenient threshold
                    print(f"🚨 Pattern repetition validation triggered! Pattern '{play_pattern}' repeated {pattern_count} times")
                    self.response_patterns[play_pattern] = 0  # Reset counter
                    return ValidationResult.RETRY, f"Excessive pattern repetition: {play_pattern}"
        else:
            self.response_patterns[play_pattern] = 1
        
        # Cleanup old patterns to prevent memory buildup
        if len(self.response_patterns) > 50:
            # Remove patterns with count 1 (seen only once)
            self.response_patterns = {k: v for k, v in self.response_patterns.items() if v > 1}
        
        return ValidationResult.VALID, "No duplicate issues detected"
    
    def _normalize_response_for_comparison(self, response_text: str) -> str:
        """Normalize response text for similarity comparison."""
        import re
        
        # Remove timestamps, scores, and other varying elements
        normalized = response_text.upper()
        normalized = re.sub(r'\d+:\d+', 'TIME', normalized)  # Replace times
        normalized = re.sub(r'\d+ - \d+', 'SCORE', normalized)  # Replace scores  
        normalized = re.sub(r'\d+', 'NUM', normalized)  # Replace other numbers
        normalized = re.sub(r'\s+', ' ', normalized)  # Normalize whitespace
        
        return normalized.strip()
    
    def _calculate_response_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two normalized response texts."""
        if not text1 or not text2:
            return 0.0
        
        # Simple character-based similarity
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_play_pattern(self, next_play: Dict[str, Any]) -> str:
        """Extract a pattern signature from play data for repetition detection."""
        player = next_play.get("player", "")
        shot_details = next_play.get("shot_details", {})
        shot_points = shot_details.get("points") if isinstance(shot_details, dict) else None
        description = next_play.get("description", "").upper()
        
        # Create pattern based on key elements
        pattern_elements = []
        
        # Add player info (normalize)
        if player:
            pattern_elements.append(f"PLAYER:{player[:10]}")  # First 10 chars of player name
        
        # Add shot info
        if isinstance(shot_points, int) and shot_points > 0:
            pattern_elements.append(f"SHOT:{shot_points}PT")
        
        # Add description keywords
        key_desc_words = []
        for word in ["MAKE", "MISS", "FOUL", "SUB", "TIMEOUT", "REBOUND", "STEAL", "TURNOVER", "ASSIST"]:
            if word in description:
                key_desc_words.append(word)
        
        if key_desc_words:
            pattern_elements.append(f"DESC:{'-'.join(key_desc_words[:3])}")  # Max 3 keywords
        
        return "|".join(pattern_elements) if pattern_elements else "GENERIC_PLAY"
    
    def _check_consecutive_same_time(self, next_play: Dict[str, Any], current_recent_plays: List[Dict[str, Any]]) -> ValidationResult:
        """
        Check for consecutive responses with the same time_remaining.
        Smart detection for end-of-quarter scenarios with automatic quarter transition.
        """
        consecutive_responses_limit = 8
        current_time = next_play.get("time_remaining")
        
        # Skip validation if time_remaining is not a string (invalid format)
        if not isinstance(current_time, str):
            return ValidationResult.VALID
        
        if self.last_time_remaining == current_time:
            self.consecutive_same_time += 1
            print(f"🕒 Same time '{current_time}' count: {self.consecutive_same_time}/{consecutive_responses_limit}")
            
            # DIRECT PERIOD DETECTION - Much simpler and more reliable!
            if current_time == "00:00":
                # Check for "period" play in current play OR recent plays
                current_play_desc = next_play.get("description", "").lower()
                current_event_code = next_play.get("event_code", "")  # For compact format
                
                # Check current play first
                period_detected = ("period" in current_play_desc or current_event_code == "period")
                print(f"DEBUG: Period detection - current play: '{current_play_desc}', event_code: '{current_event_code}', detected: {period_detected}")
                
                # If not found in current play, check recent plays at 00:00
                if not period_detected:
                    print(f"DEBUG: Checking recent plays for period signal...")
                    period_plays_found = []
                    for i, play in enumerate(current_recent_plays):
                        if play.get("time_remaining") == "00:00":
                            play_desc = play.get("description", "").lower()
                            play_event = play.get("event_code", "")
                            period_in_desc = "period" in play_desc
                            period_in_event = play_event == "period"
                            period_plays_found.append(f"Play{i}: '{play_desc}' (event: '{play_event}') -> period_in_desc: {period_in_desc}, period_in_event: {period_in_event}")
                            if period_in_desc or period_in_event:
                                period_detected = True
                                break
                    print(f"DEBUG: Recent plays analysis: {period_plays_found}")
                    print(f"DEBUG: Final period_detected: {period_detected}")
                
                if period_detected:
                    current_quarter = next_play.get("quarter", 1)
                    print(f"PERIOD DETECTED: End of Q{current_quarter} (period play found in context)")
                    
                    # For quarters 1-3, transition to next quarter. For Q4, let normal end-game logic handle it.
                    if current_quarter < 4:
                        print(f"QUARTER TRANSITION: Q{current_quarter} → Q{current_quarter + 1} (period signal)")
                        self._prepare_quarter_transition(current_quarter + 1)
                        return ValidationResult.QUARTER_TRANSITION
                    else:
                        print(f"🏁 Q4 period detected - allowing normal end-game processing")
                
                # Fallback: Multiple robust detection methods (if no direct period signal)
                else:
                    current_quarter = next_play.get("quarter", 1)
                    
                    # Try multiple fallback methods in order of preference
                    
                    # Method 1: Administrative pattern detection (original logic)
                    if self.consecutive_same_time >= 4:
                        administrative_pattern = self._detect_administrative_pattern(next_play, current_recent_plays)
                        
                        if administrative_pattern:
                            print(f"FALLBACK 1: Administrative pattern detected at Q{current_quarter} 00:00")
                            print(f"Pattern: {administrative_pattern}")
                            
                            if current_quarter < 4:
                                print(f"QUARTER TRANSITION: Q{current_quarter} → Q{current_quarter + 1} (administrative pattern)")
                                self._prepare_quarter_transition(current_quarter + 1)
                                return ValidationResult.QUARTER_TRANSITION
                            else:
                                print(f"🏁 Q4 end detected - allowing normal end-game processing")
                                return ValidationResult.VALID
                    
                    # Method 2: Aggressive substitution detection (new fallback)
                    if self.consecutive_same_time >= 3:
                        substitution_count = self._count_recent_substitutions(next_play, current_recent_plays)
                        
                        if substitution_count >= 3:  # 3+ substitutions at 00:00
                            print(f"FALLBACK 2: Multiple substitutions at Q{current_quarter} 00:00 ({substitution_count} substitutions)")
                            
                            if current_quarter < 4:
                                print(f"QUARTER TRANSITION: Q{current_quarter} → Q{current_quarter + 1} (substitution pattern)")
                                self._prepare_quarter_transition(current_quarter + 1)
                                return ValidationResult.QUARTER_TRANSITION
                    
                    # Method 3: Time-based aggressive fallback (last resort)
                    if self.consecutive_same_time >= 6:  # Been stuck for a while
                        print(f"FALLBACK 3: Extended 00:00 stall detected at Q{current_quarter} ({self.consecutive_same_time} consecutive)")
                        
                        if current_quarter < 4:
                            print(f"QUARTER TRANSITION: Q{current_quarter} → Q{current_quarter + 1} (time-based fallback)")
                            self._prepare_quarter_transition(current_quarter + 1)
                            return ValidationResult.QUARTER_TRANSITION
            
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
                    print(f"Rollback point: Most recent play at time '{last_good_time}'")
            
            if self.consecutive_same_time >= consecutive_responses_limit:
                print(f"🚨 Time rollback validation triggered! Same time '{current_time}' for {self.consecutive_same_time} consecutive responses")
                print(f"Rolling back to snapshot with {len(self.rollback_recent_plays_snapshot or [])} plays")
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
    
    def _detect_administrative_pattern(self, next_play: Dict[str, Any], current_recent_plays: List[Dict[str, Any]]) -> str:
        """
        Detect if we're stuck in an end-of-quarter administrative pattern.
        
        Returns:
            str: Description of detected pattern, or empty string if no pattern
        """
        # Get the description of the current play
        current_desc = next_play.get("description", "").lower().strip()
        
        # Administrative play keywords
        admin_keywords = ["substitution", "timeout", "unknown", "period", "end of", "technical"]
        
        # Check if current play is administrative
        current_is_admin = any(keyword in current_desc for keyword in admin_keywords)
        
        if not current_is_admin:
            return ""  # Current play is not administrative
        
        # Count recent administrative plays at 00:00
        admin_count = 0
        admin_types = set()
        
        # Include current play
        if current_is_admin:
            admin_count += 1
            admin_types.add(current_desc[:20])  # First 20 chars for type identification
        
        # Check recent plays
        for play in reversed(current_recent_plays[-5:]):  # Check last 5 plays
            play_time = play.get("time_remaining", "")
            play_desc = play.get("description", "").lower().strip()
            
            if play_time == "00:00" and any(keyword in play_desc for keyword in admin_keywords):
                admin_count += 1
                admin_types.add(play_desc[:20])  # First 20 chars for type identification
        
        # Pattern detected if 3+ administrative plays at 00:00
        if admin_count >= 3:
            pattern_desc = f"{admin_count} administrative plays: {', '.join(list(admin_types)[:3])}"
            return pattern_desc
        
        return ""
    
    def _count_recent_substitutions(self, next_play: Dict[str, Any], current_recent_plays: List[Dict[str, Any]]) -> int:
        """
        Count substitutions in recent plays at 00:00.
        
        Returns:
            int: Number of substitution plays found
        """
        substitution_count = 0
        
        # Check current play
        current_desc = next_play.get("description", "").lower()
        if "substitution" in current_desc or "sub" in current_desc:
            substitution_count += 1
        
        # Check recent plays at 00:00
        for play in reversed(current_recent_plays[-6:]):  # Check last 6 plays
            play_time = play.get("time_remaining", "")
            play_desc = play.get("description", "").lower()
            
            if play_time == "00:00" and ("substitution" in play_desc or "sub" in play_desc):
                substitution_count += 1
        
        return substitution_count
    
    def _prepare_quarter_transition(self, next_quarter: int) -> None:
        """
        Prepare validator state for quarter transition.
        
        Args:
            next_quarter: The quarter we're transitioning to (2, 3, or 4)
        """
        # Store quarter transition information for the prediction pipeline to use
        self.quarter_transition_target = next_quarter
        
        # Clear end-of-quarter tracking since we're moving to next quarter
        self.consecutive_same_time = 0
        self.last_time_remaining = None
        self._clear_rollback_state()
        
        print(f"Validator prepared for transition to Q{next_quarter}")
    
    def get_quarter_transition_target(self) -> int:
        """Get the target quarter for transition and clear it."""
        target = getattr(self, 'quarter_transition_target', None)
        if hasattr(self, 'quarter_transition_target'):
            delattr(self, 'quarter_transition_target')
        return target or 2  # Default to Q2 if somehow missing
    
    def _check_excessive_substitutions(self, next_play: Dict[str, Any]) -> bool:
        """Check for 6 consecutive substitution plays (allowing for strategic substitution sequences)."""
        description = next_play.get("description", "").upper()
        
        if "SUB" in description:
            self.consecutive_subs += 1
            print(f"Substitution detected: '{description[:50]}...' count: {self.consecutive_subs}/10")
            if self.consecutive_subs >= 10:  # Much more lenient - timeouts can have many subs
                print(f"🚨 Substitution validation triggered! {self.consecutive_subs} consecutive substitutions")
                # Reset counter and return retry
                self.consecutive_subs = 0
                return True
        else:
            # Reset counter if not a substitution
            if self.consecutive_subs > 0:
                print(f"Non-substitution play: '{description[:50]}...' (resetting sub counter from {self.consecutive_subs} to 0)")
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
    
    def _check_time_progression(self, next_play: Dict[str, Any]) -> ValidationResult:
        """
        Validate that time progresses logically within quarters.
        Time should generally decrease within the same quarter.
        """
        current_quarter = next_play.get("quarter")
        current_time = next_play.get("time_remaining")
        
        # Skip if essential data is missing
        if not isinstance(current_quarter, int) or not isinstance(current_time, str):
            return ValidationResult.VALID
        
        # Convert current time to seconds for comparison
        current_seconds = self._time_to_seconds(current_time)
        if current_seconds is None:
            return ValidationResult.VALID  # Invalid time format handled elsewhere
        
        # If we have previous data and we're in the same quarter
        if self.last_quarter == current_quarter and self.last_time_seconds is not None:
            # Allow small increases (up to 10 seconds) for timeouts, reviews, etc.
            time_diff = current_seconds - self.last_time_seconds
            
            # Only flag significant time increases (more than 10 seconds forward in same quarter)
            if time_diff > 10:  # Time jumped forward by more than 10 seconds
                self.time_progression_violations += 1
                print(f"Time progression violation: {self.last_time_remaining} → {current_time} (+{time_diff}s) in Q{current_quarter}")
                
                if self.time_progression_violations >= 5:  # More lenient - allow more violations
                    print(f"🚨 Time progression validation triggered! {self.time_progression_violations} violations")
                    self.time_progression_violations = 0  # Reset counter
                    return ValidationResult.RETRY
        
        elif self.last_quarter != current_quarter:
            # Quarter changed - reset violation counter
            self.time_progression_violations = 0
            print(f"Quarter changed: Q{self.last_quarter} → Q{current_quarter} (resetting time progression tracking)")
        
        # Update tracking state
        self.last_quarter = current_quarter
        self.last_time_seconds = current_seconds
        
        return ValidationResult.VALID
    
    def _time_to_seconds(self, time_str: str) -> Optional[int]:
        """Convert time string (MM:SS) to total seconds."""
        try:
            parts = time_str.split(':')
            if len(parts) != 2:
                return None
            
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes * 60 + seconds
        except (ValueError, IndexError):
            return None
    
    def _check_score_progression(self, next_play: Dict[str, Any]) -> ValidationResult:
        """
        Validate that scores progress logically (never decrease).
        Scores should only increase or stay the same.
        """
        current_score = next_play.get("score")
        if not isinstance(current_score, str):
            return ValidationResult.VALID
        
        # Parse current scores
        current_teams_scores = self._parse_score_to_dict(current_score)
        if not current_teams_scores:
            return ValidationResult.VALID  # Invalid score format handled elsewhere
        
        # If we have previous scores to compare
        if self.last_team_scores:
            for team, current_score_val in current_teams_scores.items():
                if team in self.last_team_scores:
                    last_score = self.last_team_scores[team]
                    
                    if current_score_val < last_score:  # Score decreased
                        decrease = last_score - current_score_val
                        self.score_progression_violations += 1
                        print(f"Score progression violation: {team} {last_score} → {current_score_val} (decreased by {decrease})")
                        
                        # Only trigger on significant decreases or repeated violations
                        # Allow minor decreases (1-2 points) which might be score corrections
                        if decrease > 3 or self.score_progression_violations >= 5:
                            print(f"🚨 Score progression validation triggered! {self.score_progression_violations} violations")
                            self.score_progression_violations = 0  # Reset counter
                            return ValidationResult.RETRY
        
        # Update tracking state
        self.last_team_scores = current_teams_scores
        return ValidationResult.VALID
    
    def _parse_score_to_dict(self, score_str: str) -> Dict[str, int]:
        """Parse score string to dictionary of team scores."""
        try:
            teams = self._extract_teams_from_score(score_str)
            if len(teams) != 2:
                return {}
            
            # Remove extra whitespace and normalize
            normalized = re.sub(r'\s+', ' ', score_str.strip().upper())
            
            # Extract scores using flexible patterns
            score_patterns = [
                r'^[A-Z]{2,15} (\d+) - [A-Z]{2,15} (\d+)$',        # LAL 108 - BOS 102
                r'^[A-Z]{2,15} (\d+), [A-Z]{2,15} (\d+)$',         # LAL 108, BOS 102  
                r'^[A-Z]{2,15}: (\d+) [A-Z]{2,15}: (\d+)$',        # LAL: 108 BOS: 102
                r'^[A-Z]{2,15} (\d+) [A-Z]{2,15} (\d+)$',          # LAL 108 BOS 102
                r'^[A-Z]{2,15}\s*(\d+)\s*[-,]\s*[A-Z]{2,15}\s*(\d+)$'  # Flexible spacing
            ]
            
            for pattern in score_patterns:
                match = re.match(pattern, normalized)
                if match:
                    score1 = int(match.group(1))
                    score2 = int(match.group(2))
                    return {teams[0]: score1, teams[1]: score2}
            
            return {}
        except (ValueError, IndexError):
            return {}
    
    def _check_score_shot_consistency(self, next_play: Dict[str, Any]) -> ValidationResult:
        """
        Validate that score changes match shot_details.points.
        If shot_details.points > 0, one team's score should increase by that amount.
        """
        shot_details = next_play.get("shot_details", {})
        if not isinstance(shot_details, dict):
            return ValidationResult.VALID
        
        shot_points = shot_details.get("points")
        shot_team = shot_details.get("team")
        
        # Skip if no shot points or invalid data
        if not isinstance(shot_points, int) or shot_points <= 0 or not shot_team:
            return ValidationResult.VALID
        
        # Parse current scores
        current_score = next_play.get("score")
        if not isinstance(current_score, str):
            return ValidationResult.VALID
        
        current_teams_scores = self._parse_score_to_dict(current_score)
        if not current_teams_scores or shot_team not in current_teams_scores:
            return ValidationResult.VALID  # Can't validate without proper score data
        
        # If we have previous scores to compare
        if self.last_team_scores and shot_team in self.last_team_scores:
            expected_score = self.last_team_scores[shot_team] + shot_points
            actual_score = current_teams_scores[shot_team]
            
            # Allow for more flexibility in score tracking
            score_diff = actual_score - expected_score
            
            if score_diff != 0:
                print(f"Score-shot tracking: {shot_team} scored {shot_points} points")
                print(f"   Expected: {self.last_team_scores[shot_team]} + {shot_points} = {expected_score}")
                print(f"   Actual: {actual_score}")
                print(f"   Difference: {score_diff}")
                
                # Only flag major inconsistencies (more than 5 points off)
                # This might indicate free throws, technical fouls, or other scoring
                if abs(score_diff) > 5:
                    print(f"Major score inconsistency detected")
                    return ValidationResult.RETRY
                elif abs(score_diff) > 2:
                    print(f"Minor score discrepancy - possibly additional free throws or scoring")
        
        return ValidationResult.VALID
    
    def _check_statistical_impossibilities(self, next_play: Dict[str, Any]) -> ValidationResult:
        """
        Check for statistically impossible or extremely unlikely scenarios.
        """
        # Check for impossible shot point values
        shot_details = next_play.get("shot_details", {})
        if isinstance(shot_details, dict):
            shot_points = shot_details.get("points")
            if isinstance(shot_points, int) and shot_points > 4:  # NBA max is 4-point play (3+foul)
                print(f"🚨 Statistical impossibility: {shot_points}-point shot detected")
                return ValidationResult.RETRY
        
        # Check for unrealistic score jumps
        current_score = next_play.get("score")
        if isinstance(current_score, str) and self.last_team_scores:
            current_teams_scores = self._parse_score_to_dict(current_score)
            for team, current_score_val in current_teams_scores.items():
                if team in self.last_team_scores:
                    score_increase = current_score_val - self.last_team_scores[team]
                    if score_increase > 6:  # Allow for technical fouls, flagrant fouls + shots, etc.
                        print(f"🚨 Statistical impossibility: {team} scored {score_increase} points in single play")
                        return ValidationResult.RETRY
        
        # Check for impossible time jumps (more than a full quarter)
        current_time = next_play.get("time_remaining")
        if isinstance(current_time, str) and self.last_time_remaining and self.last_quarter == next_play.get("quarter"):
            current_seconds = self._time_to_seconds(current_time)
            last_seconds = self._time_to_seconds(self.last_time_remaining)
            
            if current_seconds is not None and last_seconds is not None:
                time_jump = current_seconds - last_seconds
                if time_jump > 720:  # More than 12 minutes (full quarter)
                    print(f"🚨 Statistical impossibility: Time jumped by {time_jump}s ({time_jump/60:.1f} minutes)")
                    return ValidationResult.RETRY
        
        # Check for unrealistic quarter progression
        current_quarter = next_play.get("quarter")
        if isinstance(current_quarter, int) and self.last_quarter is not None:
            quarter_jump = current_quarter - self.last_quarter
            if quarter_jump > 3:  # Only block massive quarter jumps
                print(f"🚨 Statistical impossibility: Quarter jumped from {self.last_quarter} to {current_quarter}")
                return ValidationResult.RETRY
        
        # Check for unrealistic team scores (NBA record is 186 points in regulation)
        if isinstance(current_score, str):
            current_teams_scores = self._parse_score_to_dict(current_score)
            for team, score in current_teams_scores.items():
                if score > 300:  # Allow very high scores - some games can be unusual
                    print(f"🚨 Statistical impossibility: {team} has {score} points (exceeds realistic NBA limits)")
                    return ValidationResult.RETRY
        
        # Check for duplicate player names in players_on_court
        players_on_court = next_play.get("players_on_court", [])
        if isinstance(players_on_court, list):
            all_players = []
            for team_data in players_on_court:
                if isinstance(team_data, dict) and "players" in team_data:
                    all_players.extend(team_data["players"])
            
            # Check for duplicate player names
            if len(all_players) != len(set(all_players)):
                print(f"🚨 Statistical impossibility: Duplicate player names detected in players_on_court")
                return ValidationResult.RETRY
        
        return ValidationResult.VALID
    
    def _check_quarter_transitions(self, next_play: Dict[str, Any]) -> ValidationResult:
        """
        Validate proper quarter transitions and time resets.
        """
        current_quarter = next_play.get("quarter")
        current_time = next_play.get("time_remaining")
        
        if not isinstance(current_quarter, int) or not isinstance(current_time, str):
            return ValidationResult.VALID
        
        # If we have previous quarter data
        if self.last_quarter is not None:
            # Check for proper quarter progression
            quarter_change = current_quarter - self.last_quarter
            
            # Allow staying in same quarter or advancing by 1
            if quarter_change == 0:
                return ValidationResult.VALID  # Same quarter is fine
            elif quarter_change == 1:
                # Quarter advanced - check if time reset appropriately
                current_seconds = self._time_to_seconds(current_time)
                
                if current_seconds is not None:
                    # For quarters 1-4, time should be around 12:00 (720 seconds)
                    # For overtime (5+), time should be around 5:00 (300 seconds)
                    # But be very lenient - games can have different timing patterns
                    if current_quarter <= 4:
                        expected_start_time = 720  # 12:00
                        tolerance = 300  # 5 minute tolerance (very lenient)
                    else:
                        expected_start_time = 300  # 5:00 for overtime
                        tolerance = 180  # 3 minute tolerance (very lenient)
                    
                    time_diff = abs(current_seconds - expected_start_time)
                    if time_diff > tolerance:
                        self.quarter_transition_violations += 1
                        print(f"Quarter transition issue: Q{self.last_quarter}→Q{current_quarter} but time is {current_time}")
                        print(f"   Expected ~{expected_start_time//60}:{expected_start_time%60:02d}, got {current_time} (diff: {time_diff}s)")
                        
                        if self.quarter_transition_violations >= 5:  # More violations allowed
                            print(f"🚨 Quarter transition validation triggered! {self.quarter_transition_violations} violations")
                            self.quarter_transition_violations = 0
                            return ValidationResult.RETRY
            else:
                # Allow quarter progression - games should continue 
                # Only block truly impossible jumps (more than 2 quarters)
                if quarter_change > 2:  # Allow single quarter skips
                    print(f"🚨 Major quarter skip: Q{self.last_quarter} → Q{current_quarter} (skipped {quarter_change-1} quarters)")
                    return ValidationResult.RETRY
                elif quarter_change < 0:
                    print(f"Quarter regression: Q{self.last_quarter} → Q{current_quarter} (allowing to continue)")
                    # Allow backwards progression - might be model correction
        
        return ValidationResult.VALID
    
    def _check_game_situation_awareness(self, next_play: Dict[str, Any]) -> ValidationResult:
        """
        Validate that plays make sense given the game context and situation.
        """
        quarter = next_play.get("quarter")
        time_remaining = next_play.get("time_remaining")
        current_score = next_play.get("score")
        description = next_play.get("description", "").upper()
        
        if not all([isinstance(quarter, int), isinstance(time_remaining, str), isinstance(current_score, str)]):
            return ValidationResult.VALID
        
        # Parse time and scores for situation analysis
        current_seconds = self._time_to_seconds(time_remaining)
        current_teams_scores = self._parse_score_to_dict(current_score)
        
        if current_seconds is None or not current_teams_scores:
            return ValidationResult.VALID
        
        # Calculate score differential
        team_scores = list(current_teams_scores.values())
        if len(team_scores) == 2:
            score_diff = abs(team_scores[0] - team_scores[1])
        else:
            return ValidationResult.VALID
        
        # Analyze game situation
        is_close_game = score_diff <= 10
        is_very_close = score_diff <= 3
        is_final_minutes = current_seconds < 120  # Less than 2 minutes
        is_final_seconds = current_seconds < 30   # Less than 30 seconds
        is_fourth_quarter_or_ot = quarter >= 4
        is_blowout = score_diff > 20
        
        # Check for unrealistic situations
        violations = []
        
        # 1. Check timeout usage in unrealistic situations
        if "TIMEOUT" in description:
            if is_blowout and not is_final_minutes:
                violations.append("Timeout called in blowout game with time remaining")
        
        # 2. Check fouling strategy awareness (be much more lenient)
        if "FOUL" in description:
            # Only flag fouls in very specific inappropriate situations
            # Most fouls are normal basketball plays, not strategic
            if "INTENTIONAL" in description and not is_fourth_quarter_or_ot:
                # Only flag specifically mentioned intentional fouls outside of end-game
                if not is_final_minutes and score_diff > 15:
                    violations.append("Intentional fouling when far ahead early in game")
            # Otherwise, let most fouls pass - they're normal basketball
        
        # 3. Check for unrealistic play pace
        if is_final_seconds and is_very_close:
            # In final seconds of close games, every second matters
            if "SLOW" in description or "DELIBERATE" in description:
                if score_diff > 1:  # Trailing team shouldn't slow down
                    violations.append("Slow play when trailing in final seconds")
        
        # 4. Check shot selection awareness
        shot_details = next_play.get("shot_details", {})
        if isinstance(shot_details, dict):
            shot_points = shot_details.get("points")
            if isinstance(shot_points, int):
                # Three-point attempts should make sense situationally
                if shot_points == 3:
                    if is_final_minutes and is_fourth_quarter_or_ot:
                        # 3-pointers make sense when trailing by more than 3, or in very close games
                        if not (score_diff > 3 or is_very_close):
                            # Don't be too strict on this one
                            pass
                
                # Free throw situations
                if shot_points == 1:
                    if not ("FOUL" in description or "FREE" in description):
                        violations.append("1-point score without foul context")
        
        # 5. Check substitution timing
        if "SUB" in description:
            # Mass substitutions inappropriate in close, final moments
            if is_very_close and is_final_seconds:
                violations.append("Substitution in critical final seconds of close game")
        
        # Only trigger if we have clear violations and they're repeated
        if violations:
            self.situation_violations += 1
            print(f"Game situation violation #{self.situation_violations}: {violations[0]}")
            print(f"   Context: Q{quarter} {time_remaining}, Score diff: {score_diff}, Description: {description[:50]}...")
            
            # Be much more lenient - only trigger after many clear violations
            if self.situation_violations >= 8:  # Increased from 3 to 8
                print(f"🚨 Game situation awareness validation triggered! {self.situation_violations} violations")
                self.situation_violations = 0
                return ValidationResult.RETRY
        
        return ValidationResult.VALID
    
    def _check_description_consistency(self, next_play: Dict[str, Any]) -> ValidationResult:
        """
        Validate that the description field matches the actual data in the play.
        """
        description = next_play.get("description", "").upper()
        shot_details = next_play.get("shot_details", {})
        players_on_court = next_play.get("players_on_court", [])
        player = next_play.get("player", "")
        
        if not description:
            return ValidationResult.VALID
        
        inconsistencies = []
        
        # 1. Check shot point consistency with description
        if isinstance(shot_details, dict):
            shot_points = shot_details.get("points")
            if isinstance(shot_points, int):
                if shot_points == 3 and "3" not in description and "THREE" not in description and "POINTER" not in description:
                    if "SHOT" in description or "MAKE" in description or "SCORE" in description:
                        inconsistencies.append(f"Description suggests scoring but missing '3-pointer' reference for {shot_points}-point shot")
                
                if shot_points == 2 and "3" in description and ("THREE" in description or "POINTER" in description):
                    inconsistencies.append(f"Description mentions '3-pointer' but shot_details.points = {shot_points}")
                
                if shot_points == 1 and not ("FOUL" in description or "FREE" in description or "TECHNICAL" in description):
                    inconsistencies.append(f"1-point shot without foul context in description")
        
        # 2. Check player name consistency
        if isinstance(player, str) and player:
            # Check if mentioned player is actually on court
            all_court_players = []
            for team_data in players_on_court:
                if isinstance(team_data, dict) and "players" in team_data:
                    all_court_players.extend(team_data["players"])
            
            if player not in all_court_players and player.upper() not in description:
                # If the player field is set but not mentioned in description, that's suspicious
                if len(all_court_players) > 0:  # Only if we have valid player data
                    inconsistencies.append(f"Player '{player}' not mentioned in description but set in player field")
        
        # 3. Check for score mentions vs actual scoring
        current_score = next_play.get("score")
        if isinstance(current_score, str) and self.last_team_scores:
            current_teams_scores = self._parse_score_to_dict(current_score)
            
            # Check if description mentions scoring
            score_keywords = ["SCORE", "MAKE", "BASKET", "SHOT", "POINT"]
            mentions_scoring = any(keyword in description for keyword in score_keywords)
            
            # Check if any team's score actually increased
            actual_scoring = False
            for team, current_score_val in current_teams_scores.items():
                if team in self.last_team_scores and current_score_val > self.last_team_scores[team]:
                    actual_scoring = True
                    break
            
            # Flag mismatches
            if mentions_scoring and not actual_scoring:
                # Allow some flexibility for missed shots
                if not ("MISS" in description or "BLOCK" in description or "REBOUND" in description):
                    inconsistencies.append("Description suggests scoring but no score increase detected")
            
            elif actual_scoring and not mentions_scoring:
                # This is more serious - score increased but description doesn't reflect it
                if not ("SUB" in description or "TIMEOUT" in description or "FOUL" in description):
                    inconsistencies.append("Score increased but description doesn't mention scoring play")
        
        # 4. Check substitution consistency
        if "SUB" in description:
            # Should mention player names
            if not any(char.isalpha() for char in description.replace("SUB", "")):
                inconsistencies.append("Substitution mentioned but no player names in description")
        
        # 5. Check time/quarter mentions
        quarter = next_play.get("quarter")
        time_remaining = next_play.get("time_remaining")
        
        if isinstance(quarter, int) and quarter == 4:
            if "FOURTH" in description or "4TH" in description:
                pass  # Consistent
            elif "FIRST" in description or "1ST" in description:
                inconsistencies.append("Description mentions first quarter but quarter = 4")
        
        # Only trigger on significant inconsistencies
        if inconsistencies:
            self.description_consistency_violations += 1
            print(f"Description consistency violation #{self.description_consistency_violations}:")
            for inconsistency in inconsistencies[:2]:  # Show first 2
                print(f"   {inconsistency}")
            print(f"   Description: {description[:60]}...")
            
            # Be very lenient - only trigger after many clear violations
            if self.description_consistency_violations >= 6:  # Increased from 3 to 6
                print(f"🚨 Description consistency validation triggered! {self.description_consistency_violations} violations")
                self.description_consistency_violations = 0
                return ValidationResult.RETRY
        
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
        
        # Reset time progression tracking
        self.last_quarter = None
        self.last_time_seconds = None
        self.time_progression_violations = 0
        
        # Reset score progression tracking
        self.last_team_scores = None
        self.score_progression_violations = 0
        
        # Reset quarter transition tracking
        self.quarter_transition_violations = 0
        
        # Reset game situation tracking
        self.situation_violations = 0
        
        # Reset description consistency tracking
        self.description_consistency_violations = 0
        
        # Reset enhanced duplicate detection tracking
        self.response_patterns.clear()
        self.near_duplicate_count = 0
        
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
        """Validate time format (MM:SS) with caching and optimized parsing."""
        # Fast length check first
        if not time_str or len(time_str) < 3 or len(time_str) > 5:
            return False
        
        # Check cache first (time formats are often repeated)
        if time_str in self._time_validation_cache:
            return self._time_validation_cache[time_str]
            
        # Use pre-compiled regex for speed
        if not TIME_FORMAT_REGEX.match(time_str):
            if len(self._time_validation_cache) < self._cache_max_size:
                self._time_validation_cache[time_str] = False
            return False
        
        # Optimized parsing (avoid exception handling in happy path)
        colon_pos = time_str.find(':')
        if colon_pos == -1:
            if len(self._time_validation_cache) < self._cache_max_size:
                self._time_validation_cache[time_str] = False
            return False
            
        try:
            minutes = int(time_str[:colon_pos])
            seconds = int(time_str[colon_pos + 1:])
            # Fast bounds check
            is_valid = 0 <= minutes <= 12 and 0 <= seconds <= 59
            
            # Cache result
            if len(self._time_validation_cache) < self._cache_max_size:
                self._time_validation_cache[time_str] = is_valid
            
            return is_valid
        except ValueError:
            if len(self._time_validation_cache) < self._cache_max_size:
                self._time_validation_cache[time_str] = False
            return False
    
    def _validate_score_format(self, score_str: str) -> bool:
        """Validate score format with caching and pre-compiled patterns (optimized)."""
        # Fast path: empty or too short strings
        if not score_str or len(score_str) < 7:  # Minimum: "A 0-B 0"
            return False
        
        # Check cache first (avoid expensive regex operations)
        if score_str in self._score_validation_cache:
            return self._score_validation_cache[score_str]
        
        # Remove extra whitespace and normalize (vectorized operation)
        normalized = re.sub(r'\s+', ' ', score_str.strip().upper())
        
        # Use pre-compiled patterns for speed
        is_valid = any(pattern.match(normalized) for pattern in SCORE_PATTERNS)
        
        # Cache result (with size limit)
        if len(self._score_validation_cache) < self._cache_max_size:
            self._score_validation_cache[score_str] = is_valid
        
        return is_valid
    
    def _extract_teams_from_score(self, score_str: str) -> List[str]:
        """Extract team names from score string with flexible parsing."""
        # Remove extra whitespace and normalize
        normalized = re.sub(r'\s+', ' ', score_str.strip().upper())
        
        # Multiple patterns for team extraction
        extraction_patterns = [
            r'^([A-Z]{2,15}) \d+ - ([A-Z]{2,15}) \d+$',      # LAL 108 - BOS 102
            r'^([A-Z]{2,15}) \d+, ([A-Z]{2,15}) \d+$',       # LAL 108, BOS 102
            r'^([A-Z]{2,15}): \d+ ([A-Z]{2,15}): \d+$',      # LAL: 108 BOS: 102
            r'^([A-Z]{2,15}) \d+ ([A-Z]{2,15}) \d+$',        # LAL 108 BOS 102
            r'^([A-Z]{2,15})\s*\d+\s*[-,]\s*([A-Z]{2,15})\s*\d+$'  # Flexible spacing
        ]
        
        for pattern in extraction_patterns:
            match = re.match(pattern, normalized)
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
        
        summary = f"VALIDATION TERMINATION SUMMARY\n"
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

    # Fast validation methods for performance optimization
    def _validate_next_play_structure_fast(self, next_play: Any) -> bool:
        """Ultra-fast validation with minimal essential next_play structure checks."""
        # Fast type check without expensive string operations
        if not isinstance(next_play, dict):
            self.errors.append(ValidationError(
                field_path="next_play",
                error_type="type_error", 
                expected="dict",
                actual=type(next_play).__name__,  # Avoid str() call
                message=self._error_messages['dict_required']  # Pre-compiled message
            ))
            return False
        
        # Ultra-fast essential field check (single field, direct access)
        description = next_play.get("description")
        if description is None:
            self.errors.append(ValidationError(
                field_path="next_play.description",
                error_type="missing_field",
                expected="description",
                actual="missing",
                message=self._error_messages['description_missing']  # Pre-compiled message
            ))
            return False
        
        # Fast string validation (avoid strip() if possible)
        if not isinstance(description, str) or len(description) == 0:
            self.errors.append(ValidationError(
                field_path="next_play.description",
                error_type="value_error",
                expected="non-empty string",
                actual=str(description) if len(str(description)) < 50 else "long_value",  # Avoid expensive string conversion for long values
                message=self._error_messages['description_empty']  # Pre-compiled message
            ))
            return False
        
        # Fast path success - no expensive operations
        return True

    def _validate_fast_path(self, response_text: str, next_play: Dict[str, Any], context: Dict[str, Any]) -> Tuple[ValidationResult, List[ValidationError], str]:
        """Ultra-fast validation with optimized critical checks."""
        
        # Pre-fetch values once for efficiency (avoid multiple dict lookups)
        quarter = next_play.get("quarter")
        time_remaining = next_play.get("time_remaining", "")
        
        # 1. Optimized game ending check using pre-compiled sets
        if quarter == QUARTER_4 and time_remaining in GAME_END_TIMES:
            self.consecutive_endgame += 1
            if self.consecutive_endgame >= 3:  # More lenient threshold
                return ValidationResult.END_GAME, self.errors, "Game ended - quarter 4, time expired"
        else:
            self.consecutive_endgame = 0
        
        # 2. FAST PERIOD/ADMIN DETECTION - Check immediately for period at 00:00 (don't wait for consecutive times)
        if time_remaining == "00:00":
            # Check current play for period
            current_desc = next_play.get("description", "").lower()
            current_event = next_play.get("event_code", "")
            
            period_detected = ("period" in current_desc or current_event == "period")
            
            # Check recent plays for period if not found in current
            if not period_detected and context and "recent_plays" in context:
                for play in context["recent_plays"]:
                    if play.get("time_remaining") == "00:00":
                        play_desc = play.get("description", "").lower()
                        play_event = play.get("event_code", "")
                        if "period" in play_desc or play_event == "period":
                            period_detected = True
                            break
            
            # Trigger quarter transition if period detected
            if period_detected and quarter and quarter < 4:
                self._prepare_quarter_transition(quarter + 1)
                return ValidationResult.QUARTER_TRANSITION, self.errors, f"Period detected - transitioning Q{quarter} → Q{quarter + 1}"
            
            # NEW: Administrative/stall fallback at 00:00 without explicit 'period'
            # Works for both verbose (recent_plays) and compact (p) contexts.
            if quarter and quarter < 4 and context:
                admin_count = 0
                zero_time_count = 0
                # Verbose context
                if "recent_plays" in context and isinstance(context["recent_plays"], list):
                    admin_keywords = ("sub", "substitution", "timeout", "jump", "technical", "unknown")
                    zero_time_recent = [p for p in context["recent_plays"] if p.get("time_remaining") == "00:00"]
                    zero_time_count += len(zero_time_recent)
                    for p in zero_time_recent[-8:]:  # last few plays at 00:00
                        desc = str(p.get("description", "")).lower()
                        if any(k in desc for k in admin_keywords):
                            admin_count += 1
                # Compact context
                if "p" in context and isinstance(context["p"], list):
                    zero_time_compact = [pl for pl in context["p"][-12:] if isinstance(pl, list) and len(pl) >= 5 and pl[1] == 0]
                    zero_time_count += len(zero_time_compact)
                    for pl in zero_time_compact[-8:]:
                        ev = str(pl[4]).lower()
                        if ev in ("sub", "jumpball", "timeout", "unknown") or ev.startswith("sub") or ev.startswith("jump"):
                            admin_count += 1
                # Consider our own consecutive_same_time counter as well
                too_many_zero_time = (self.consecutive_same_time >= 4) or (zero_time_count >= 4)
                if admin_count >= 2 and too_many_zero_time:
                    print(f"FAST FALLBACK: {admin_count} admin plays at 00:00, zero_time_count={zero_time_count}, consecutive_same_time={self.consecutive_same_time}")
                    self._prepare_quarter_transition(quarter + 1)
                    return ValidationResult.QUARTER_TRANSITION, self.errors, f"Administrative pattern at 00:00 - transitioning Q{quarter} → Q{quarter + 1}"
        
        # 3. Time progression check for stuck scenarios
        if time_remaining and (
            self.last_time_remaining is time_remaining or 
            self.last_time_remaining == time_remaining
        ):
            self.consecutive_same_time += 1
            
            # Fallback: Basic rollback after more attempts
            if self.consecutive_same_time >= 8:
                return ValidationResult.ROLLBACK_TIME, self.errors, "Time progression stuck"
        else:
            self.consecutive_same_time = 0
            self.last_time_remaining = time_remaining
        
        # Ultra-fast success path
        return ValidationResult.VALID, self.errors, "Fast validation passed"
