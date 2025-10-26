"""
Base prediction client interface for multi-platform NBA play prediction.

This module provides the abstract base class for prediction clients,
enabling support for multiple platforms (OpenAI, Gemini, etc.) with a
consistent interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union
import json
import os
import time
import threading
import random
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dotenv import load_dotenv
from .response_validator import NBAResponseValidator, ValidationResult


# =====================
# Global rate limiter
# =====================

_GLOBAL_LIMITER = None
_GLOBAL_LIMITER_LOCK = threading.Lock()


class _RequestLimiter:
    """Process-wide limiter for concurrent calls and pacing.

    - Limits max in-flight requests (semaphore)
    - Enforces a minimum interval between request starts (global pacing)
    """

    def __init__(self, max_concurrent: int, min_interval_seconds: float, jitter: float = 0.0):
        self._semaphore = threading.Semaphore(max(1, int(max_concurrent)))
        self._min_interval = max(0.0, float(min_interval_seconds))
        self._jitter = max(0.0, float(jitter))
        self._lock = threading.Lock()
        self._last_start_ts = 0.0

    def acquire(self):
        # Limit concurrent in-flight requests
        self._semaphore.acquire()

        # Pacing: ensure at least min_interval between request starts
        if self._min_interval > 0.0:
            with self._lock:
                now = time.time()
                earliest = self._last_start_ts + self._min_interval
                if now < earliest:
                    delay = earliest - now
                    # Add small jitter to avoid sync thundering herd
                    delay += random.random() * self._jitter
                    time.sleep(delay)
                    now = time.time()
                # Mark start
                self._last_start_ts = now

    def release(self):
        try:
            self._semaphore.release()
        except Exception:
            pass


def _get_global_request_limiter() -> Optional[_RequestLimiter]:
    """Create or return a singleton limiter if enabled via env vars.

    Enable with GENAI_ENABLE_LIMITER=1
    Controls:
      - GENAI_MAX_CONCURRENT (default 2)
      - GENAI_TARGET_RPS (default 2.0) → min interval = 1/RPS
      - GENAI_MIN_DELAY_BETWEEN_CALLS (overrides RPS if set)
      - GENAI_LIMITER_JITTER (default 0.1s)
    """
    enabled = os.getenv("GENAI_ENABLE_LIMITER", "0").lower() in ("1", "true", "yes", "on")
    if not enabled:
        return None

    global _GLOBAL_LIMITER
    with _GLOBAL_LIMITER_LOCK:
        if _GLOBAL_LIMITER is None:
            try:
                max_concurrent = int(os.getenv("GENAI_MAX_CONCURRENT", "2"))
            except Exception:
                max_concurrent = 2
            # Determine pacing interval
            min_delay_env = os.getenv("GENAI_MIN_DELAY_BETWEEN_CALLS")
            if min_delay_env is not None:
                try:
                    min_interval = float(min_delay_env)
                except Exception:
                    min_interval = 0.0
            else:
                try:
                    target_rps = float(os.getenv("GENAI_TARGET_RPS", "2.0"))
                    min_interval = 1.0 / target_rps if target_rps > 0 else 0.0
                except Exception:
                    min_interval = 0.0
            try:
                jitter = float(os.getenv("GENAI_LIMITER_JITTER", "0.1"))
            except Exception:
                jitter = 0.1

            _GLOBAL_LIMITER = _RequestLimiter(max_concurrent=max_concurrent,
                                              min_interval_seconds=min_interval,
                                              jitter=jitter)
        return _GLOBAL_LIMITER

class BasePredictionClient(ABC):
    """Abstract base class for prediction clients."""
    
    def __init__(self, validation_mode: str = "fast"):
        """Initialize the prediction client."""
        self._initialize_client()
        self.validator = NBAResponseValidator(validation_mode=validation_mode)
        self.last_validation_failure = None  # Store detailed validation failure info
        # Debug/verbose response logging toggle (enabled when PREDICTION_LOG_LEVEL >= 3 or VALIDATION_DEBUG=1)
        try:
            _lvl = int(os.getenv("PREDICTION_LOG_LEVEL", "1"))
        except Exception:
            _lvl = 1
        self._debug_log = _lvl >= 3
        self._log_stage_responses = os.getenv("VALIDATION_DEBUG", "0").lower() in ("1", "true", "yes", "on")
        # Initialize global rate limiter (no-op if not configured)
        self._rate_limiter = _get_global_request_limiter()
    
    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return the name of the platform this client targets."""
        pass
    
    @abstractmethod
    def _initialize_client(self):
        """Initialize the platform-specific client."""
        pass
    
    @abstractmethod
    def predict(self, context: Dict[str, Any], model_id: str, 
                max_tokens: int = 1500, temperature: float = 0.1) -> Tuple[str, Dict[str, Any]]:
        """
        Make a prediction using the specified model.
        
        Args:
            context: Game context dictionary
            model_id: Model identifier for the platform
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            
        Returns:
            tuple: (response_content, usage_stats)
        """
        pass
    
    def predict_from_json(self, context_json: str, model_id: str, 
                         max_tokens: int = 1500, temperature: float = 0.1) -> Tuple[str, Dict[str, Any]]:
        """
        Make a prediction using pre-serialized JSON context (performance optimization).
        
        Args:
            context_json: Pre-serialized JSON context string
            model_id: Model identifier for the platform
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            
        Returns:
            tuple: (response_content, usage_stats)
        """
        # Default implementation: parse JSON and call regular predict
        # Subclasses can override this for better performance
        try:
            context = json.loads(context_json)
            # Default implementation ignores instruction; subclasses may use it
            return self.predict(context, model_id, max_tokens, temperature)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON context: {e}")
    
    def get_model_config(self) -> Dict[str, str]:
        """
        Get platform-specific model configuration.
        
        Returns:
            dict: Dictionary with 'model_1_id' and 'model_2_id' keys
        """
        return self._get_default_models()
    
    def predict_with_validation(self, context: Union[Dict[str, Any], str], model_id: str, 
                               max_tokens: int = 1500, temperature: float = 0.1,
                               max_retries: int = 3, stage1_mode: bool = False) -> Tuple[str, Dict[str, Any], bool, bool, Optional[Dict[str, Any]]]:
        """
        Make a prediction with validation and retry logic.
        """
        context_json: Optional[str] = None
        if isinstance(context, str):
            context_json = context
        else:
            context_json = json.dumps(context, separators=(',', ':'))
        validation_context = context if isinstance(context, dict) else json.loads(context_json)
        
        validation_attempts = []
        current_max_tokens = max_tokens
        
        for attempt in range(max_retries + 1):
            try:
                # Debug logging: Show input context being sent to model
                if self._debug_log or self._log_stage_responses:
                    stage_label = "Stage 1" if stage1_mode else "Stage 2"
                    print(f"\n{'='*80}")
                    print(f"🔵 DEBUG {stage_label} INPUT CONTEXT (attempt {attempt + 1})")
                    print(f"{'='*80}")
                    print(f"Model: {model_id}")
                    print(f"Max tokens: {current_max_tokens}, Temperature: {temperature}")
                    print(f"Context size: {len(context_json)} characters")
                    
                    # Show condensed context structure
                    try:
                        ctx = validation_context
                        is_compact = 'A' in ctx and 'H' in ctx
                        
                        if is_compact:
                            print(f"Format: COMPACT")
                            print(f"Teams: {ctx.get('A', 'N/A')} vs {ctx.get('H', 'N/A')}")
                            print(f"Away players: {ctx.get('ap_count', len(ctx.get('ap', [])))} players")
                            print(f"Home players: {ctx.get('hp_count', len(ctx.get('hp', [])))} players")
                            
                            # Show if recent plays exist
                            if 'p' in ctx:
                                p_len = len(ctx['p']) if isinstance(ctx['p'], list) else 'N/A'
                                print(f"Recent plays: {p_len} plays")
                                # Show first play structure to verify format
                                if isinstance(ctx['p'], list) and len(ctx['p']) > 0:
                                    first_play = ctx['p'][0]
                                    print(f"  First play structure: {len(first_play)} elements")
                                    if len(first_play) >= 10:
                                        print(f"    Away lineup: {first_play[8]}")
                                        print(f"    Home lineup: {first_play[9]}")
                            else:
                                print(f"Recent plays: None (first_N_plays mode)")
                            
                            # Show team stats snippet
                            if 'as' in ctx:
                                as_arr = ctx['as']
                                print(f"Away team stats: [{as_arr[0]:.2f}, {as_arr[1]:.2f}, {as_arr[2]:.2f}, ... {len(as_arr)} values]")
                            if 'hs' in ctx:
                                hs_arr = ctx['hs']
                                print(f"Home team stats: [{hs_arr[0]:.2f}, {hs_arr[1]:.2f}, {hs_arr[2]:.2f}, ... {len(hs_arr)} values]")
                            
                            # Show first player from each team
                            if 'ap' in ctx and len(ctx['ap']) > 0:
                                first_player = ctx['ap'][0]
                                if isinstance(first_player, list):
                                    print(f"Away P0: {first_player[0]} (MPG: {first_player[1]}, Usage: {first_player[2]:.3f})")
                            if 'hp' in ctx and len(ctx['hp']) > 0:
                                first_player = ctx['hp'][0]
                                if isinstance(first_player, list):
                                    print(f"Home P0: {first_player[0]} (MPG: {first_player[1]}, Usage: {first_player[2]:.3f})")
                        else:
                            print(f"Format: VERBOSE")
                            print(f"Away team: {ctx.get('away_team', {}).get('name', 'N/A')}")
                            print(f"Home team: {ctx.get('home_team', {}).get('name', 'N/A')}")
                            if 'recent_plays' in ctx:
                                rp_len = len(ctx['recent_plays']) if isinstance(ctx['recent_plays'], list) else 'N/A'
                                print(f"Recent plays: {rp_len} plays")
                        
                        # Validate JSON structure
                        test_json = json.dumps(ctx, separators=(',', ':'))
                        print(f"✓ Context is valid JSON ({len(test_json)} chars)")
                        
                        # Raw input payload as sent to the model
                        print("\n--- RAW REQUEST INPUT (context_json) ---")
                        try:
                            print(context_json)
                        except Exception:
                            print("<non-printable context_json>")
                        print("--- END RAW REQUEST INPUT ---\n")
                        
                    except Exception as e:
                        print(f"⚠️ Error analyzing context: {e}")
                        print(f"Raw context (first 500 chars): {context_json[:500]}")
                    
                    print(f"{'='*80}\n")
                
                limiter = getattr(self, "_rate_limiter", None)
                if limiter is not None:
                    limiter.acquire()
                try:
                    if context_json is not None and hasattr(self, "predict_from_json"):
                        response_content, usage_stats = self.predict_from_json(context_json, model_id, current_max_tokens, temperature)
                    else:
                        response_content, usage_stats = self.predict(validation_context, model_id, current_max_tokens, temperature)
                finally:
                    if limiter is not None:
                        limiter.release()
                
                # Debug logging: Show output response from model
                if self._debug_log or self._log_stage_responses:
                    stage_label = "Stage 1" if stage1_mode else "Stage 2"
                    print(f"\n{'='*80}")
                    print(f"🟢 DEBUG {stage_label} RAW RESPONSE (attempt {attempt + 1})")
                    print(f"{'='*80}")
                    try:
                        print(response_content)
                    except Exception:
                        # Ensure logging never breaks the run
                        print("<non-printable response content>")
                    print(f"{'='*80}\n")
                
                # Sanitize and normalize JSON before validation
                processed_content = self._sanitize_json_like_text(response_content)
                processed_content = self._normalize_compact_response_if_applicable(processed_content)
                
                # Validate response (different logic for Stage 1 vs Stage 2)
                if stage1_mode:
                    validation_result, validation_errors, reason = self._validate_stage1_response(processed_content, validation_context)
                else:
                    validation_result, validation_errors, reason = self.validator.validate_response(processed_content, validation_context)
                
                if validation_result == ValidationResult.VALID:
                    if attempt > 0:
                        print(f"Validation successful on attempt {attempt + 1}")
                    return processed_content, usage_stats, False, False, None
                
                elif validation_result == ValidationResult.END_GAME:
                    print(f"🏁 Game ending condition detected: {reason}")
                    print(f"Returning final play and ending game loop")
                    
                    # Capture detailed termination information
                    termination_info = None
                    if not stage1_mode and self.validator.has_termination_record():
                        termination_info = self.validator.get_termination_for_database()
                        print(f"📊 Captured termination details: {termination_info.get('validation_termination_type', 'unknown')}")
                        
                        # Also print summary to console
                        if self.validator.get_termination_info():
                            print(f"TERMINATION SUMMARY:")
                            print(f"   Type: {termination_info.get('validation_termination_type', 'N/A')}")
                            print(f"   Trigger: {termination_info.get('validation_trigger_condition', 'N/A')}")
                            print(f"   Game State: Q{termination_info.get('validation_game_state_quarter', '?')} {termination_info.get('validation_game_state_time', 'N/A')}")
                    
                    return response_content, usage_stats, True, False, termination_info
                
                elif validation_result == ValidationResult.ROLLBACK_TIME:
                    print(f"Timestamp rollback required: {reason}")
                    print(f"Returning rollback signal to main loop")
                    
                    # Capture detailed rollback termination information
                    termination_info = None
                    if not stage1_mode and self.validator.has_termination_record():
                        termination_info = self.validator.get_termination_for_database()
                        print(f"📊 Captured rollback details: {termination_info.get('validation_termination_type', 'unknown')}")
                        
                        # Also print summary to console
                        if self.validator.get_termination_info():
                            print(f"ROLLBACK SUMMARY:")
                            print(f"   Trigger: {termination_info.get('validation_trigger_condition', 'N/A')}")
                            print(f"   Consecutive Count: {termination_info.get('validation_consecutive_count', 0)}")
                            print(f"   Total Attempts: {termination_info.get('validation_total_attempts', 0)}")
                    
                    # Note: response_content may be invalid, but needs_rollback=True signals the main loop to handle this
                    return response_content, usage_stats, False, True, termination_info
                
                elif validation_result == ValidationResult.QUARTER_TRANSITION:
                    print(f"Quarter transition required: {reason}")
                    print(f"Returning quarter transition signal to main loop")
                    
                    # DEBUG: Check validator state
                    print(f"DEBUG: stage1_mode={stage1_mode}")
                    print(f"DEBUG: validator.has_termination_record()={self.validator.has_termination_record()}")
                    if hasattr(self.validator, 'quarter_transition_target'):
                        print(f"DEBUG: quarter_transition_target={self.validator.quarter_transition_target}")
                    else:
                        print(f"DEBUG: No quarter_transition_target in validator")
                    
                    # Capture quarter transition termination information
                    termination_info = None
                    if not stage1_mode and self.validator.has_termination_record():
                        termination_info = self.validator.get_termination_for_database()
                        print(f"📊 Captured quarter transition details: {termination_info.get('validation_termination_type', 'unknown')}")
                        
                        # Also print summary to console
                        if self.validator.get_termination_info():
                            print(f"QUARTER TRANSITION SUMMARY:")
                            print(f"   Trigger: {termination_info.get('validation_trigger_condition', 'N/A')}")
                            # Don't consume the target quarter here - let the main loop get it
                            target_for_display = getattr(self.validator, 'quarter_transition_target', 'Unknown')
                            print(f"   Target Quarter: Q{target_for_display}")
                    else:
                        print(f"DEBUG: Not capturing termination info - stage1_mode={stage1_mode}, has_termination_record={self.validator.has_termination_record()}")
                    
                    print(f"DEBUG: Final termination_info={termination_info}")
                    
                    # Return with quarter_transition flag (we'll use needs_rollback=True but with different termination info)
                    return response_content, usage_stats, False, True, termination_info
                
                elif validation_result == ValidationResult.RETRY:
                    # Log validation retry reason
                    validation_attempts.append({
                        'attempt': attempt + 1,
                        'reason': reason,
                        'errors': validation_errors,
                        'response_preview': response_content[:200] + "..." if len(response_content) > 200 else response_content
                    })
                    
                    print(f"Validation requires retry on attempt {attempt + 1}/{max_retries + 1}")
                    print(f"   Reason: {reason}")
                    if validation_errors:
                        print(f"   Additional errors: {len(validation_errors)} validation issues")
                        # Detailed error dump in debug mode
                        if self._debug_log or self._log_stage_responses:
                            for err in validation_errors[:10]:
                                try:
                                    print(f"     - [{getattr(err, 'field_path', '?')}] {getattr(err, 'error_type', '?')}: {getattr(err, 'message', '')}")
                                except Exception:
                                    pass
                            if len(validation_errors) > 10:
                                print(f"     ... and {len(validation_errors) - 10} more")
                    
                    if attempt < max_retries:
                        print(f"Retrying prediction...")
                    else:
                        print(f"💥 Max retries ({max_retries}) exceeded")
                
            except Exception as e:
                # Handle rate limiting / quota errors with exponential backoff + jitter
                if self._is_rate_limit_error(e):
                    delay = self._compute_backoff_seconds(attempt)
                    print(f"Rate limited or quota exhausted (attempt {attempt + 1}). Backing off {delay:.2f}s...")
                    time.sleep(delay)
                    continue
                # Handle Together token limit error by reducing max tokens and retrying
                new_allowed = self._extract_together_allowed_max_tokens(e)
                if new_allowed is not None:
                    if new_allowed < current_max_tokens:
                        print(f"Together token limit hit. Reducing max_tokens from {current_max_tokens} to {new_allowed} and retrying...")
                        current_max_tokens = max(64, new_allowed)
                        continue
                print(f"Prediction attempt {attempt + 1} failed with error: {str(e)}")
                if attempt == max_retries:
                    raise e
        
        # All attempts failed - create detailed error message and validation failure info
        error_details = []
        for attempt_info in validation_attempts:
            error_details.append(f"\nAttempt {attempt_info['attempt']}:")
            error_details.append(f"  Response preview: {attempt_info['response_preview']}")
            error_details.append(f"  Retry reason: {attempt_info['reason']}")
            if attempt_info['errors']:
                error_details.append(f"  Additional validation errors ({len(attempt_info['errors'])}):")
                for error in attempt_info['errors'][:3]:  # Show first 3 errors
                    error_details.append(f"    - {error.field_path}: {error.message}")
                if len(attempt_info['errors']) > 3:
                    error_details.append(f"    - ... and {len(attempt_info['errors']) - 3} more errors")
        
        full_error_message = f"Response validation failed after {max_retries + 1} attempts:{''.join(error_details)}"
        
        # Create comprehensive validation failure info for database storage
        validation_failure_info = self._create_validation_failure_info(validation_attempts, max_retries + 1)
        
        # Store validation failure info for potential capture by orchestrator
        self.last_validation_failure = validation_failure_info
        
        raise ValueError(full_error_message)
    
    def _create_validation_failure_info(self, validation_attempts: List[Dict[str, Any]], total_attempts: int) -> Dict[str, Any]:
        """Create comprehensive validation failure information for database storage."""
        from datetime import datetime
        
        if not validation_attempts:
            return {}
        
        # Analyze failure patterns
        failure_reasons = {}
        error_types = {}
        error_fields = {}
        
        for attempt in validation_attempts:
            reason = attempt['reason']
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            
            # Analyze validation errors
            for error in attempt.get('errors', []):
                error_type = error.error_type
                error_field = error.field_path
                
                error_types[error_type] = error_types.get(error_type, 0) + 1
                error_fields[error_field] = error_fields.get(error_field, 0) + 1
        
        # Get most common issues
        most_common_reason = max(failure_reasons.items(), key=lambda x: x[1]) if failure_reasons else ("unknown", 0)
        most_common_error_type = max(error_types.items(), key=lambda x: x[1]) if error_types else ("none", 0)
        most_common_field = max(error_fields.items(), key=lambda x: x[1]) if error_fields else ("none", 0)
        
        # Get examples of failed responses (first and last)
        response_examples = []
        if len(validation_attempts) > 0:
            response_examples.append(validation_attempts[0]['response_preview'])
            if len(validation_attempts) > 1:
                response_examples.append(validation_attempts[-1]['response_preview'])
        
        return {
            'validation_failure_timestamp': datetime.now().isoformat(),
            'validation_total_failed_attempts': total_attempts,
            'validation_most_common_reason': most_common_reason[0],
            'validation_most_common_reason_count': most_common_reason[1],
            'validation_most_common_error_type': most_common_error_type[0],
            'validation_most_common_error_type_count': most_common_error_type[1],
            'validation_most_common_field': most_common_field[0],
            'validation_most_common_field_count': most_common_field[1],
            'validation_unique_reasons': len(failure_reasons),
            'validation_unique_error_types': len(error_types),
            'validation_unique_fields': len(error_fields),
            'validation_failure_summary': json.dumps({
                'reasons': failure_reasons,
                'error_types': error_types,
                'error_fields': error_fields
            }),
            'validation_response_examples': json.dumps(response_examples)
        }
    
    def get_last_validation_failure_info(self) -> Optional[Dict[str, Any]]:
        """Get the most recent validation failure information."""
        return self.last_validation_failure
    
    # ==== Rate limit detection and backoff helpers ====
    def _is_rate_limit_error(self, err: Exception) -> bool:
        """Heuristically detect 429/RESOURCE_EXHAUSTED/rate limit style errors."""
        try:
            msg = str(err).lower()
        except Exception:
            return False
        indicators = [
            "429",
            "resource exhausted",
            "quota",
            "rate limit",
            "retry after",
            "too many requests",
        ]
        return any(tok in msg for tok in indicators)
    
    def _compute_backoff_seconds(self, attempt_index: int) -> float:
        """Exponential backoff with jitter, configurable via env vars.
        GENAI_BACKOFF_BASE (default 0.5), GENAI_BACKOFF_CAP (default 10), GENAI_BACKOFF_JITTER (default 0.25)
        """
        try:
            base = float(os.getenv("GENAI_BACKOFF_BASE", "0.5"))
        except Exception:
            base = 0.5
        try:
            cap = float(os.getenv("GENAI_BACKOFF_CAP", "10"))
        except Exception:
            cap = 10.0
        try:
            jitter = float(os.getenv("GENAI_BACKOFF_JITTER", "0.25"))
        except Exception:
            jitter = 0.25
        delay = min(cap, base * (2 ** attempt_index))
        return delay + random.random() * jitter
    
    def _validate_stage1_response(self, response_text: str, context: Dict[str, Any]) -> Tuple[ValidationResult, list, str]:
        """
        Validate Stage 1 response which should return multiple plays.
        Expected formats: 
        - Compact wrapped: {"y": [[play_tuple1], [play_tuple2], ..., [play_tupleN]]}
        - Compact raw array: [[play_tuple1], [play_tuple2], ..., [play_tupleN]]
        - Verbose: {"next_plays": [play1, play2, ..., playN]}
        """
        try:
            response_data = json.loads(response_text.strip())
        except json.JSONDecodeError as e:
            return ValidationResult.RETRY, [], f"JSON parse error: {str(e)}"
        
        # Check for plays array - handle multiple formats
        next_plays = None
        
        # NEW: Handle raw array of play tuples (model trained on compact format may return this)
        if isinstance(response_data, list):
            # Raw array format - wrap it as if it were {"y": [...]}
            play_tuples = response_data
            if not play_tuples:
                return ValidationResult.RETRY, [], "Empty play tuples array"
            
            next_plays = []
            for i, play_tuple in enumerate(play_tuples):
                if not isinstance(play_tuple, list) or len(play_tuple) not in (10, 9, 7):
                    return ValidationResult.RETRY, [], f"Play tuple {i+1} must have 10 elements (current), 9 (legacy with lineup_id), or 7 (legacy), got {len(play_tuple) if isinstance(play_tuple, list) else 'non-list'}"
                
                # Convert tuple to minimal play object for validation
                if len(play_tuple) == 10:
                    # Current format: [q, t, score, margin, actor, actor_fouls, event, shot_zone, away_lineup, home_lineup]
                    quarter = play_tuple[0]
                    time_seconds = play_tuple[1]
                    score_array = play_tuple[2] if len(play_tuple[2]) >= 2 else [0, 0]
                    actor = play_tuple[4]
                    event_code = play_tuple[6]
                elif len(play_tuple) == 9:
                    # Legacy format with lineup_id
                    quarter = play_tuple[0]
                    time_seconds = play_tuple[1]
                    score_array = play_tuple[2] if len(play_tuple[2]) >= 2 else [0, 0]
                    actor = play_tuple[4]
                    event_code = play_tuple[6]
                else:
                    # Legacy 7-element format
                    quarter = play_tuple[0]
                    time_seconds = play_tuple[1]
                    score_array = play_tuple[2] if len(play_tuple[2]) >= 2 else [0, 0]
                    actor = play_tuple[3]
                    event_code = play_tuple[4]
                
                # Convert time back to MM:SS format
                minutes = time_seconds // 60
                seconds = time_seconds % 60
                time_remaining = f"{minutes:02d}:{seconds:02d}"
                
                # Create minimal play object
                play = {
                    "quarter": quarter,
                    "time_remaining": time_remaining,
                    "description": f"Predicted {event_code}",
                    "score": f"AWAY {score_array[0]} - HOME {score_array[1]}",
                    "_compact_format": True
                }
                next_plays.append(play)
        
        elif "y" in response_data:
            # Compact wrapped format - convert play tuples to minimal verbose format for validation
            play_tuples = response_data["y"]
            if not isinstance(play_tuples, list):
                return ValidationResult.RETRY, [], "'y' field must be an array"
            
            next_plays = []
            for i, play_tuple in enumerate(play_tuples):
                if not isinstance(play_tuple, list) or len(play_tuple) not in (10, 9, 7):
                    return ValidationResult.RETRY, [], f"Play tuple {i+1} must have 10 elements (current), 9 (legacy with lineup_id), or 7 (legacy), got {len(play_tuple) if isinstance(play_tuple, list) else 'non-list'}"
                
                # Convert tuple to minimal play object for validation
                if len(play_tuple) == 10:
                    # Current format: [q, t, score, margin, actor, actor_fouls, event, shot_zone, away_lineup, home_lineup]
                    quarter = play_tuple[0]
                    time_seconds = play_tuple[1]
                    score_array = play_tuple[2] if len(play_tuple[2]) >= 2 else [0, 0]
                    actor = play_tuple[4]
                    event_code = play_tuple[6]
                elif len(play_tuple) == 9:
                    # Legacy format with lineup_id
                    quarter = play_tuple[0]
                    time_seconds = play_tuple[1]
                    score_array = play_tuple[2] if len(play_tuple[2]) >= 2 else [0, 0]
                    actor = play_tuple[4]
                    event_code = play_tuple[6]
                else:
                    # Legacy 7-element format
                    quarter = play_tuple[0]
                    time_seconds = play_tuple[1]
                    score_array = play_tuple[2] if len(play_tuple[2]) >= 2 else [0, 0]
                    actor = play_tuple[3]
                    event_code = play_tuple[4]
                
                # Convert time back to MM:SS format
                minutes = time_seconds // 60
                seconds = time_seconds % 60
                time_remaining = f"{minutes:02d}:{seconds:02d}"
                
                # Create minimal play object
                play = {
                    "quarter": quarter,
                    "time_remaining": time_remaining,
                    "description": f"Predicted {event_code}",  # Satisfies validation requirement
                    "score": f"AWAY {score_array[0]} - HOME {score_array[1]}",
                    "_compact_format": True
                }
                next_plays.append(play)
                
        elif "next_plays" in response_data:
            # Verbose format - use as-is
            next_plays = response_data["next_plays"]
        else:
            return ValidationResult.RETRY, [], "Missing 'next_plays' field (verbose) or 'y' field (compact)"
        
        # Validate it's an array
        if not isinstance(next_plays, list):
            if "y" in response_data:
                return ValidationResult.RETRY, [], "'y' field must be an array"
            else:
                return ValidationResult.RETRY, [], "'next_plays' must be an array"
        
        # Enforce expected count from env if provided
        expected_count = None
        try:
            from os import getenv as _getenv
            expected_str = _getenv("STAGE1_PLAY_COUNT")
            if expected_str:
                expected_count = int(expected_str)
        except Exception:
            expected_count = None
        if expected_count is not None and len(next_plays) != expected_count:
            return ValidationResult.RETRY, [], f"Stage 1 plays count mismatch: expected {expected_count}, got {len(next_plays)}"

        # Check array length (should have reasonable number of plays)
        if len(next_plays) == 0:
            if "y" in response_data:
                return ValidationResult.RETRY, [], "'y' array cannot be empty"
            else:
                return ValidationResult.RETRY, [], "'next_plays' array cannot be empty"
        
        if len(next_plays) > 50:  # Reasonable upper limit
            field_name = "'y'" if "y" in response_data else "'next_plays'"
            return ValidationResult.RETRY, [], f"{field_name} array too large ({len(next_plays)} plays)"
        
        # Basic validation of each play (less strict than Stage 2)
        for i, play in enumerate(next_plays):
            if not isinstance(play, dict):
                return ValidationResult.RETRY, [], f"Play {i+1} must be a dictionary"
                
            # Check for basic required fields
            required_fields = ["description"]  # Minimal requirement for Stage 1
            for field in required_fields:
                if field not in play:
                    return ValidationResult.RETRY, [], f"Play {i+1} missing required field: {field}"
        
        # Stage 1 validation passed
        return ValidationResult.VALID, [], "Valid Stage 1 response"
    
    def _extract_together_allowed_max_tokens(self, err: Exception) -> Optional[int]:
        """Parse Together 422 errors to compute allowed max tokens dynamically.
        Looks for: "`inputs` tokens + `max_new_tokens` must be <= <limit>. Given: <inputs> ... <max_new_tokens>"
        """
        try:
            msg = str(err)
        except Exception:
            return None
        if "Input validation error" not in msg:
            return None
        if "max_new_tokens" not in msg:
            return None
        # Extract limit
        limit_match = re.search(r"must be <=\s*(\d+)", msg)
        limit = int(limit_match.group(1)) if limit_match else 8192
        # Extract inputs tokens
        given_match = re.search(r"Given:\s*(\d+) `inputs` tokens and (\d+) `max_new_tokens`", msg)
        if not given_match:
            return None
        try:
            inputs_tokens = int(given_match.group(1))
        except Exception:
            return None
        # Compute allowed new tokens with a small safety cushion
        allowed = max(64, limit - inputs_tokens - 32)
        return allowed

    def _sanitize_json_like_text(self, text: str) -> str:
        """Trim to the first balanced {...} JSON object, strip trailing junk, return sanitized text.
        If no braces found, return original.
        """
        if not isinstance(text, str):
            return text
        s = text.strip()
        # Find first '{' and attempt to locate its matching '}'
        start = s.find('{')
        if start == -1:
            return s
        # Scan for matching closing, counting braces
        depth = 0
        end_index = None
        for i in range(start, len(s)):
            ch = s[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end_index = i
                    break
        if end_index is not None:
            candidate = s[start:end_index+1]
        else:
            candidate = s[start:]
        # Remove common trailing artifacts like comments or periods after object end
        return candidate.strip()

    def _normalize_compact_response_if_applicable(self, text: str) -> str:
        """If response is compact JSON with 'y', normalize shapes:
        - If y is a single tuple (list of values), wrap as [ tuple ]
        - If first tuple length is 6, append lineup=0 to make 7 elements
        Returns possibly re-serialized JSON string, else original text on failure.
        """
        try:
            data = json.loads(text)
        except Exception:
            return text
        try:
            if isinstance(data, dict) and 'y' in data:
                y = data['y']
                # If y is a single tuple (list of primitives), wrap it
                if isinstance(y, list) and y and not isinstance(y[0], list):
                    y = [y]
                # If y is list of tuples, ensure first one has 7 elements
                if isinstance(y, list) and y:
                    first = y[0]
                    if isinstance(first, list) and len(first) == 6:
                        # Append lineup=0 to reach 7 elements
                        first = first + [0]
                        y[0] = first
                data['y'] = y
                return json.dumps(data, separators=(',', ':'))
        except Exception:
            return text
        return text

    @abstractmethod
    def _get_default_models(self) -> Dict[str, str]:
        """Get default model IDs for this platform."""
        pass


class OpenAIPredictionClient(BasePredictionClient):
    """OpenAI prediction client implementation."""
    
    @property
    def platform_name(self) -> str:
        return "openai"
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI library not found. Install with: pip install openai")
        
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not found. "
                "Please set your OpenAI API key as an environment variable."
            )
        
        self.client = OpenAI(api_key=api_key)
    
    def predict(self, context: Dict[str, Any], model_id: str, 
                max_tokens: int = 1500, temperature: float = 0.1) -> Tuple[str, Dict[str, Any]]:
        """Make prediction using OpenAI API."""
        context_json = json.dumps(context, separators=(',', ':'))
        return self.predict_from_json(context_json, model_id, max_tokens, temperature)
    
    def predict_from_json(self, context_json: str, model_id: str, 
                         max_tokens: int = 1500, temperature: float = 0.1) -> Tuple[str, Dict[str, Any]]:
        """Make prediction using pre-serialized JSON (optimized)."""
        messages = [{"role": "user", "content": context_json}]
        response = self.client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        content = response.choices[0].message.content
        usage_stats = {
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        return content, usage_stats
    
    def _get_default_models(self) -> Dict[str, str]:
        """Get default OpenAI model IDs."""
        return {
            "model_1_id": "ft:gpt-4.1-nano-2025-04-14:personal:first-n-plays:CAieHHyW",
            "model_2_id": "ft:gpt-4.1-nano-2025-04-14:personal:part-1:CAoc9Uw6"
        }


class TogetherPredictionClient(BasePredictionClient):
    """Together.ai prediction client implementation (OpenAI-compatible API)."""
    
    @property
    def platform_name(self) -> str:
        return "together"
    
    def _initialize_client(self):
        """Initialize Together.ai client using official Together SDK."""
        try:
            from together import Together
        except ImportError:
            raise ImportError("Together library not found. Install with: pip install together")
        
        load_dotenv()
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError(
                "TOGETHER_API_KEY environment variable not found. "
                "Please set your Together.ai API key as an environment variable."
            )
        
        # Initialize official Together client (uses default base URL)
        # Note: We intentionally avoid passing unsupported kwargs like proxies
        self.client = Together(api_key=api_key)
    
    def predict(self, context: Dict[str, Any], model_id: str, 
                max_tokens: int = 1500, temperature: float = 0.1) -> Tuple[str, Dict[str, Any]]:
        """Make prediction using Together.ai chat completions.
        Adds a strict-format system prompt only for Stage 1 model (model_1_id) calls.
        """
        context_json = json.dumps(context, separators=(',', ':'))
        return self._predict_with_messages(context_json, model_id, max_tokens, temperature)
    
    def predict_from_json(self, context_json: str, model_id: str, 
                         max_tokens: int = 1500, temperature: float = 0.1) -> Tuple[str, Dict[str, Any]]:
        """Make prediction using pre-serialized JSON (optimized)."""
        # Include instruction in user/system messages alongside context
        return self._predict_with_messages(context_json, model_id, max_tokens, temperature)
    
    def _predict_with_messages(self, context_json: str, model_id: str,
                               max_tokens: int, temperature: float) -> Tuple[str, Dict[str, Any]]:
        """Call Together chat.completions with optional Stage-1-only system prompt."""
        # Detect Stage 1 vs 2 by comparing to configured model_1_id (if available)
        # We send the strict system message only for Stage 1 to enforce exact schema
        system_prompt = None
        try:
            model_cfg = self.get_model_config()
            if model_id == model_cfg.get("model_1_id"):
                # Stage 1 strict format
                system_prompt = (
                    "Return a single JSON object with key 'y' that maps to an array of tuples. "
                    "Each tuple must have exactly 9 elements: "
                    "[quarter:int, time_sec:int, score:[away:int,home:int], margin:int, actor:[\"A\"|\"H\", int], "
                    "actor_fouls:int, event:str, shot_zone:null|str, lineup:int]. No prose, no extra keys."
                )
            elif model_id == model_cfg.get("model_2_id"):
                # Stage 2 strict format and no trailing characters
                system_prompt = (
                    "Output a single strict JSON object only. No trailing characters, no comments, no prose. "
                    "The object must contain key 'y' mapping to an array of tuples, each exactly 9 elements: "
                    "[quarter:int, time_sec:int, score:[away:int,home:int], margin:int, actor:[\"A\"|\"H\", int], "
                    "actor_fouls:int, event:str, shot_zone:null|str, lineup:int]."
                )
        except Exception:
            pass

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": context_json})

        response = self.client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        content = response.choices[0].message.content
        # Together's response includes usage similar to OpenAI
        usage = getattr(response, "usage", None)
        usage_stats = {
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        }
        
        return content, usage_stats
    
    def _get_default_models(self) -> Dict[str, str]:
        """Get default Together model IDs (stage 1 and stage 2)."""
        # Allow multiple env names for convenience; leave placeholders if unset
        model_1 = (
            os.getenv("TOGETHER_MODEL_1_ID") or
            os.getenv("TOGETHER_INITIAL_MODEL_ID") or
            "YOUR_TOGETHER_MODEL_1_ID"
        )
        model_2 = (
            os.getenv("TOGETHER_MODEL_2_ID") or
            os.getenv("TOGETHER_ROLLING_MODEL_ID") or
            "YOUR_TOGETHER_MODEL_2_ID"
        )
        return {
            "model_1_id": model_1,
            "model_2_id": model_2,
        }


class GeminiPredictionClient(BasePredictionClient):
    """Gemini prediction client implementation."""
    
    @property
    def platform_name(self) -> str:
        return "gemini"
    
    def _initialize_client(self):
        """Initialize Gemini client."""
        try:
            from vertexai import init
            from vertexai.generative_models import GenerativeModel, GenerationConfig
        except ImportError:
            raise ImportError(
                "Vertex AI library not found. Install with: pip install google-cloud-aiplatform"
            )
        
        # Initialize Vertex AI with default project (assumes gcloud CLI is configured)
        try:
            # Try to get project from environment or gcloud config
            project_id = (os.getenv("GOOGLE_CLOUD_PROJECT") or 
                         os.getenv("GCP_PROJECT") or 
                         os.getenv("GOOGLE_PROJECT_ID"))
            if not project_id:
                # Try to get from gcloud config
                import subprocess
                try:
                    result = subprocess.run(
                        ["gcloud", "config", "get-value", "project"], 
                        capture_output=True, text=True, check=True
                    )
                    project_id = result.stdout.strip()
                except (subprocess.CalledProcessError, FileNotFoundError):
                    raise ValueError(
                        "Could not determine Google Cloud project. Please set GOOGLE_CLOUD_PROJECT "
                        "environment variable or configure gcloud CLI with: gcloud config set project PROJECT_ID"
                    )
            
            # Default to us-central1 location
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            
            init(project=project_id, location=location)
            self.project_id = project_id
            self.location = location
            
        except Exception as e:
            raise ValueError(f"Failed to initialize Vertex AI: {e}")
    
    def predict(self, context: Dict[str, Any], model_id: str, 
                max_tokens: int = 1500, temperature: float = 0.1) -> Tuple[str, Dict[str, Any]]:
        """Make prediction using Gemini API."""
        context_json = json.dumps(context, separators=(',', ':'))
        return self.predict_from_json(context_json, model_id, max_tokens, temperature)
    
    def predict_from_json(self, context_json: str, model_id: str, 
                         max_tokens: int = 1500, temperature: float = 0.1) -> Tuple[str, Dict[str, Any]]:
        """Make prediction using pre-serialized JSON (optimized)."""
        # Try the new Google GenAI SDK with proper thinking control first
        try:
            return self._predict_with_genai_sdk_json(context_json, model_id, max_tokens, temperature)
        except Exception as genai_error:
            # If it's a rate limit / quota error, propagate to outer retry loop for backoff
            if self._is_rate_limit_error(genai_error):
                raise
            print(f"Google GenAI SDK failed: {genai_error}")
            print("Falling back to Vertex AI approach...")
            return self._predict_with_vertexai_json(context_json, model_id, max_tokens, temperature)
    
    def _predict_with_genai_sdk(self, context: Dict[str, Any], model_id: str, 
                               max_tokens: int = 1500, temperature: float = 0.1) -> Tuple[str, Dict[str, Any]]:
        """Make prediction using the new Google GenAI SDK with thinking control."""
        # Add instruction to encourage direct responses
        enhanced_context = context.copy()
        enhanced_context["_instruction"] = "Provide direct JSON response immediately. No reasoning or explanation needed."
        context_json = json.dumps(enhanced_context, separators=(',', ':'))
        return self._predict_with_genai_sdk_json(context_json, model_id, max_tokens, temperature)
    
    def _predict_with_genai_sdk_json(self, context_json: str, model_id: str, 
                                    max_tokens: int = 1500, temperature: float = 0.1) -> Tuple[str, Dict[str, Any]]:
        """Make prediction using the new Google GenAI SDK with pre-serialized JSON (optimized)."""
        def _run_call():
            try:
                from google import genai
                from google.genai import types
            except ImportError:
                raise ImportError("Google GenAI library not found. Install with: pip install google-generativeai")
            
            # Initialize the client with Vertex AI backend
            client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.location,
            )
            
            # Prepare contents for the model
            contents = [types.Content(role="user", parts=[types.Part(text=context_json)])]
            
            # Configure generation with thinking disabled
            generate_content_config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0,  # This is the key - disables thinking!
                ),
            )
            
            # Generate content using streaming (but collect all chunks)
            response_text = ""
            token_count = 0
            
            for chunk in client.models.generate_content_stream(
                model=model_id,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.text:
                    response_text += chunk.text
                    token_count += len(chunk.text.split()) * 1.3  # Rough estimate
            
            # Prepare usage stats
            usage_stats = {
                "completion_tokens": token_count,
                "prompt_tokens": len(context_json.split()) * 1.3,  # Rough estimate
                "total_tokens": len(context_json.split()) * 1.3 + token_count
            }
            
            return response_text, usage_stats

        return self._call_with_timeout(_run_call)
    
    def _predict_with_vertexai(self, context: Dict[str, Any], model_id: str, 
                              max_tokens: int = 1500, temperature: float = 0.1) -> Tuple[str, Dict[str, Any]]:
        """Fallback prediction using Vertex AI SDK."""
        # Add instruction to encourage direct responses without extensive reasoning
        enhanced_context = context.copy()
        enhanced_context["_instruction"] = "Provide direct JSON response immediately. No reasoning or explanation needed."
        context_json = json.dumps(enhanced_context, separators=(',', ':'))
        return self._predict_with_vertexai_json(context_json, model_id, max_tokens, temperature)
    
    def _predict_with_vertexai_json(self, context_json: str, model_id: str, 
                                   max_tokens: int = 1500, temperature: float = 0.1) -> Tuple[str, Dict[str, Any]]:
        """Fallback prediction using Vertex AI SDK with pre-serialized JSON (optimized)."""
        def _run_call():
            try:
                from vertexai.generative_models import GenerativeModel, GenerationConfig
                # Try to import ThinkingConfig if available in newer versions
                try:
                    from vertexai.generative_models._generative_models import ThinkingConfig
                    thinking_config_available = True
                except ImportError:
                    thinking_config_available = False
            except ImportError:
                raise ImportError("Vertex AI library not found. Install with: pip install google-cloud-aiplatform")
            
            # Create model instance - model_id should be the full endpoint path
            # e.g., "projects/PROJECT_ID/locations/us-central1/endpoints/ENDPOINT_ID"
            model = GenerativeModel(model_id)
            
            # Create generation config - accommodate thinking tokens until we can disable them
            if thinking_config_available:
                try:
                    # Create ThinkingConfig to disable reasoning (Gemini 2.5+ models)
                    thinking_config = ThinkingConfig(thinking_budget=0)
                    generation_config = GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                        thinking_config=thinking_config
                    )
                    print("🔧 Thinking budget disabled (set to 0) for faster, direct responses")
                except Exception as e:
                    print(f"Could not disable thinking budget: {e}. Using expanded token config.")
                    # Increase tokens to accommodate thinking overhead
                    generation_config = GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens + 2000  # Extra tokens for thinking
                    )
            else:
                print("ThinkingConfig not available - increasing token limit to accommodate thinking overhead")
                # Since we can't disable thinking, give the model more tokens
                # The model used 1499 thinking tokens, so we need buffer space
                generation_config = GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens + 2000  # 3500 total: ~1500 thinking + 1500+ response
                )
            
            # Generate content
            response = model.generate_content(
                context_json,
                generation_config=generation_config
            )
            
            content = response.text
            
            # Gemini doesn't provide detailed token usage in the same way
            # We'll estimate based on content length
            usage_stats = {
                "completion_tokens": len(content.split()) * 1.3,  # Rough estimate
                "prompt_tokens": len(context_json.split()) * 1.3,  # Rough estimate
                "total_tokens": len(context_json.split()) * 1.3 + len(content.split()) * 1.3
            }
            
            return content, usage_stats

        return self._call_with_timeout(_run_call)
    
    def _get_default_models(self) -> Dict[str, str]:
        """Get default Gemini model endpoint IDs."""
        # Get endpoint IDs from environment variables or use placeholders
        # Support multiple variable name formats
        model_1_endpoint = (os.getenv("GEMINI_MODEL_1_ENDPOINT") or 
                          os.getenv("INITIAL_PREDICTION_ENDPOINT_ID") or 
                          "YOUR_MODEL_1_ENDPOINT_ID")
        model_2_endpoint = (os.getenv("GEMINI_MODEL_2_ENDPOINT") or 
                          os.getenv("ROLLING_PREDICTIONS_ENDPOINT_ID") or 
                          "YOUR_MODEL_2_ENDPOINT_ID")
        
        # Format: "projects/PROJECT_ID/locations/LOCATION/endpoints/ENDPOINT_ID"
        return {
            "model_1_id": f"projects/{self.project_id}/locations/{self.location}/endpoints/{model_1_endpoint}",
            "model_2_id": f"projects/{self.project_id}/locations/{self.location}/endpoints/{model_2_endpoint}"
        }

    # ==== Timeout utility ====
    def _get_request_timeout_seconds(self) -> float:
        try:
            return float(os.getenv("GENAI_REQUEST_TIMEOUT_SECONDS", "90"))
        except Exception:
            return 90.0

    def _call_with_timeout(self, func):
        """Run func() with a timeout; raise TimeoutError to trigger outer backoff."""
        timeout_s = self._get_request_timeout_seconds()
        if getattr(self, "_debug_log", False):
            print(f"⏱️  GENAI call start (timeout={timeout_s:.0f}s)")
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(func)
            try:
                result = fut.result(timeout=timeout_s)
                if getattr(self, "_debug_log", False):
                    print("✅ GENAI call finished")
                return result
            except FuturesTimeout:
                if getattr(self, "_debug_log", False):
                    print(f"⏰ GENAI request timed out after {timeout_s:.0f}s")
                raise TimeoutError(f"GENAI request timed out after {timeout_s:.0f}s")


class PredictionClientFactory:
    """Factory for creating platform-specific prediction clients."""
    
    _clients = {
        'openai': OpenAIPredictionClient,
        'gemini': GeminiPredictionClient,
        'together': TogetherPredictionClient
    }
    
    @classmethod
    def create_client(cls, platform_name: str, validation_mode: str = None) -> BasePredictionClient:
        """
        Create a prediction client instance for the specified platform.
        
        Args:
            platform_name: Name of the platform ('openai' or 'gemini')
            validation_mode: Validation mode ('fast', 'normal', 'strict'). 
                           If None, uses VALIDATION_MODE env var or defaults to 'fast'
            
        Returns:
            BasePredictionClient: Client instance for the platform
            
        Raises:
            ValueError: If platform is not supported
        """
        platform_key = platform_name.lower()
        if platform_key not in cls._clients:
            available = list(cls._clients.keys())
            raise ValueError(f"Unsupported platform '{platform_name}'. Available platforms: {available}")
        
        # Determine validation mode
        if validation_mode is None:
            validation_mode = os.getenv("VALIDATION_MODE", "fast")
        
        if validation_mode not in ["fast", "normal", "strict"]:
            print(f"Invalid validation mode '{validation_mode}', defaulting to 'fast'")
            validation_mode = "fast"
        
        return cls._clients[platform_key](validation_mode=validation_mode)
    
    @classmethod
    def get_available_platforms(cls) -> list:
        """
        Get list of available platform names.
        
        Returns:
            list: List of supported platform names
        """
        return list(cls._clients.keys())
    
    @classmethod
    def is_platform_supported(cls, platform_name: str) -> bool:
        """
        Check if a platform is supported.
        
        Args:
            platform_name: Name of the platform
            
        Returns:
            bool: True if platform is supported, False otherwise
        """
        return platform_name.lower() in cls._clients
