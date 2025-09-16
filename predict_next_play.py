#!/usr/bin/env python3
"""
NBA Play Prediction Script

Multi-platform script to test predictions using fine-tuned models.
Supports OpenAI and Gemini platforms.
Takes team and player data and predicts the next play using the trained model.

CONFIGURATION:
=============

Platform Selection:
- Set PREDICTION_PLATFORM environment variable to "openai" or "gemini"
- Defaults to "openai" if not specified

OpenAI Setup:
1. Set OPENAI_API_KEY environment variable
2. Default model IDs are preconfigured for the provided fine-tuned models

Gemini Setup:
1. Install: pip install google-cloud-aiplatform
2. Authenticate: gcloud auth login
3. Set project: gcloud config set project YOUR_PROJECT_ID
4. Set environment variables:
   - GEMINI_MODEL_1_ENDPOINT=your_model_1_endpoint_id
   - GEMINI_MODEL_2_ENDPOINT=your_model_2_endpoint_id
   - GOOGLE_CLOUD_PROJECT=your_project_id (optional)
   - GOOGLE_CLOUD_LOCATION=us-central1 (optional)

USAGE:
======
python predict_next_play.py

The script will automatically detect your platform configuration and run
the appropriate prediction client.
"""

import json
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from game.prediction_client import PredictionClientFactory, BasePredictionClient

# Try to import ujson for faster JSON operations, fallback to standard json
try:
    import ujson
    json_dumps = ujson.dumps
    json_loads = ujson.loads
    print("✅ Using ujson for faster JSON operations")
except ImportError:
    json_dumps = json.dumps
    json_loads = json.loads
    print("ℹ️ Using standard json library (consider installing ujson for better performance)")


class OptimizedGameContext:
    """Optimized context manager that caches JSON serialization to eliminate redundant operations."""
    
    def __init__(self, base_context: Dict[str, Any]):
        """Initialize with base context (teams, stats, etc. - everything except recent_plays)."""
        self.base_context = {k: v for k, v in base_context.items() if k != 'recent_plays'}
        
        # Cache the base context JSON (teams, stats, etc.) - this rarely changes
        self._base_json_cached = json_dumps(self.base_context, separators=(',', ':'))
        
        # Cache for recent_plays combinations to avoid re-serializing the same play sequences
        self._plays_cache: Dict[str, str] = {}
        
        # Current state
        self.current_recent_plays: list = base_context.get('recent_plays', [])
        self._current_full_json: Optional[str] = None
        self._current_plays_hash: Optional[str] = None
        
        print(f"🚀 OptimizedGameContext initialized - base context cached ({len(self._base_json_cached)} chars)")
    
    def _hash_plays(self, plays: list) -> str:
        """Create a hash key for play sequence for caching (optimized)."""
        if not plays:
            return "empty"
        
        # Fast hash using simplified approach for small sequences
        # This avoids expensive tuple creation and string operations
        play_count = len(plays)
        if play_count <= 3:  # For small sequences, use lightweight content hash
            # Use a simple but consistent hash for small sequences
            simple_hash = hash(play_count)
            for play in plays:
                desc = play.get('description', '')
                simple_hash ^= hash(desc[:20])  # First 20 chars only
            return f"small_{play_count}_{simple_hash}"
        
        # For larger sequences or when identity fails, use efficient content-based hash
        # Avoid creating intermediate tuple - use hash accumulation instead
        hash_acc = hash(play_count)  # Start with length
        for i, play in enumerate(plays):
            if i < 5:  # Only hash first 5 plays for speed (most variance is at the beginning)
                desc = play.get('description', '')
                time_r = play.get('time_remaining', '')
                hash_acc ^= hash(desc[:30])  # Only hash first 30 chars of description
                hash_acc ^= hash(time_r)
        
        return str(hash_acc)
    
    def update_recent_plays(self, new_plays: list) -> None:
        """Update recent plays and invalidate cache if changed."""
        new_hash = self._hash_plays(new_plays)
        if self._current_plays_hash != new_hash:
            self.current_recent_plays = new_plays
            self._current_plays_hash = new_hash
            self._current_full_json = None  # Invalidate cached JSON
    
    def get_json(self) -> str:
        """Get optimized JSON string for the current context."""
        if self._current_full_json is not None:
            return self._current_full_json
        
        # Check if we have this plays combination cached
        plays_hash = self._current_plays_hash or self._hash_plays(self.current_recent_plays)
        
        if plays_hash in self._plays_cache:
            plays_json = self._plays_cache[plays_hash]
        else:
            # Cache miss - serialize recent_plays
            plays_json = json_dumps(self.current_recent_plays, separators=(',', ':'))
            self._plays_cache[plays_hash] = plays_json
            
            # Optimized cache eviction - clear half the cache when full (more efficient than single item removal)
            if len(self._plays_cache) > 50:
                # Clear oldest half of entries (batch operation is faster)
                keys_to_remove = list(self._plays_cache.keys())[:25]  # Remove first 25
                for key in keys_to_remove:
                    del self._plays_cache[key]
        
        # Optimized JSON combination using join (faster than f-string for large strings)
        if self.current_recent_plays:
            self._current_full_json = ''.join([
                self._base_json_cached[:-1],  # Remove closing brace
                ',"recent_plays":',
                plays_json,
                '}}'
            ])
        else:
            # For empty plays, just add closing brace - avoid string slicing
            self._current_full_json = self._base_json_cached[:-1] + '}}'
        
        return self._current_full_json
    
    def get_context_dict(self) -> Dict[str, Any]:
        """Get the full context as a dictionary (for compatibility)."""
        result = self.base_context.copy()
        if self.current_recent_plays:
            result['recent_plays'] = self.current_recent_plays
        return result
    
    def add_play_and_slide(self, new_play: Dict[str, Any], max_plays: int = 20) -> None:
        """Add a new play and maintain sliding window (optimized)."""
        current_len = len(self.current_recent_plays)
        
        # Optimized: avoid copy when possible, modify in-place when beneficial
        if current_len < max_plays:
            # Simple append case - extend current list
            new_plays = self.current_recent_plays + [new_play]
        else:
            # Sliding window case - use slicing to avoid intermediate lists
            new_plays = self.current_recent_plays[1:] + [new_play]
        
        self.update_recent_plays(new_plays)
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get caching statistics for debugging."""
        return {
            "plays_cache_size": len(self._plays_cache),
            "base_json_length": len(self._base_json_cached),
            "current_plays_count": len(self.current_recent_plays)
        }
    
    def restore_recent_plays(self, restored_plays: list) -> None:
        """
        Restore recent_plays to a previous state (used for timestamp rollback).
        
        Args:
            restored_plays: List of plays to restore to
        """
        print(f"🔄 Restoring recent_plays: {len(self.current_recent_plays)} → {len(restored_plays)} plays")
        if restored_plays:
            last_play = restored_plays[-1]
            print(f"🕒 Rolling back to time: {last_play.get('time_remaining', 'N/A')}")
            print(f"📝 Last play: {last_play.get('description', 'N/A')[:60]}...")
        
        self.update_recent_plays(restored_plays)
        
        # Clear the plays cache since we've changed state
        self._plays_cache.clear()
        print(f"🧹 Cleared plays cache due to rollback")


class CompactGameContext:
    """Optimized context manager for compact schema format."""
    
    def __init__(self, compact_context: Dict[str, Any]):
        """Initialize with compact context format."""
        if isinstance(compact_context, str):
            compact_context = json_loads(compact_context)
        
        # Store the base compact context (everything except plays array)
        self.base_compact = {k: v for k, v in compact_context.items() if k != 'p'}
        
        # Cache the base compact JSON (rarely changes)
        self._base_json_cached = json_dumps(self.base_compact, separators=(',', ':'))
        
        # Cache for play combinations
        self._plays_cache: Dict[str, str] = {}
        
        # Current plays state
        self.current_plays: list = compact_context.get('p', [])
        self._current_full_json: Optional[str] = None
        self._current_plays_hash: Optional[str] = None
        
        print(f"🚀 CompactGameContext initialized - base context cached ({len(self._base_json_cached)} chars)")
    
    def _hash_plays(self, plays: list) -> str:
        """Create a hash key for play tuples."""
        if not plays:
            return "empty"
        
        # Hash based on play tuples content
        play_count = len(plays)
        if play_count <= 3:
            simple_hash = hash(play_count)
            for play in plays:
                if len(play) >= 5:  # Minimum play tuple length
                    simple_hash ^= hash(str(play[:5]))  # Hash first 5 elements
            return f"compact_{play_count}_{simple_hash}"
        
        # For larger sequences, hash more efficiently
        hash_acc = hash(play_count)
        for i, play in enumerate(plays):
            if i < 5 and len(play) >= 5:  # Only hash first 5 plays
                hash_acc ^= hash(str(play[:5]))  # Hash essential play elements
        
        return str(hash_acc)
    
    def update_plays(self, new_plays: list) -> None:
        """Update plays array and invalidate cache if changed."""
        new_hash = self._hash_plays(new_plays)
        if self._current_plays_hash != new_hash:
            self.current_plays = new_plays
            self._current_plays_hash = new_hash
            self._current_full_json = None  # Invalidate cached JSON
    
    def get_json(self) -> str:
        """Get optimized JSON string for the current compact context."""
        if self._current_full_json is not None:
            return self._current_full_json
        
        # Check if we have this plays combination cached
        plays_hash = self._current_plays_hash or self._hash_plays(self.current_plays)
        
        if plays_hash in self._plays_cache:
            plays_json = self._plays_cache[plays_hash]
        else:
            # Cache miss - serialize current plays
            plays_json = json_dumps(self.current_plays, separators=(',', ':'))
            self._plays_cache[plays_hash] = plays_json
            
            # Cache eviction
            if len(self._plays_cache) > 50:
                keys_to_remove = list(self._plays_cache.keys())[:25]
                for key in keys_to_remove:
                    del self._plays_cache[key]
        
        # Combine base context with plays array
        if self.current_plays:
            self._current_full_json = ''.join([
                self._base_json_cached[:-1],  # Remove closing brace
                ',"p":',
                plays_json,
                '}'
            ])
        else:
            # No plays, just close the base context
            self._current_full_json = self._base_json_cached
        
        return self._current_full_json
    
    def get_context_dict(self) -> Dict[str, Any]:
        """Get the full context as a dictionary."""
        result = self.base_compact.copy()
        if self.current_plays:
            result['p'] = self.current_plays
        return result
    
    def add_play_tuple(self, new_play_tuple: list, max_plays: int = 20) -> None:
        """Add a new play tuple and maintain sliding window."""
        current_len = len(self.current_plays)
        
        if current_len < max_plays:
            new_plays = self.current_plays + [new_play_tuple]
        else:
            # Sliding window - remove first, add last
            new_plays = self.current_plays[1:] + [new_play_tuple]
        
        self.update_plays(new_plays)
    
    def get_base_context_for_stage1(self) -> Dict[str, Any]:
        """Get base context without plays for Stage 1 (clean context)."""
        return self.base_compact.copy()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get caching statistics for debugging."""
        return {
            "plays_cache_size": len(self._plays_cache),
            "base_json_length": len(self._base_json_cached),
            "current_plays_count": len(self.current_plays)
        }


class LoggingConfig:
    """Centralized logging configuration for performance optimization."""
    
    def __init__(self):
        # Logging levels: 0=minimal, 1=normal, 2=verbose, 3=debug
        self.level = int(os.getenv("PREDICTION_LOG_LEVEL", "1"))
        self.show_json_dumps = self.level >= 3
        self.show_iteration_details = self.level >= 2
        self.show_context_stats = self.level >= 2
        self.show_validation_details = self.level >= 3
        
        if self.level == 0:
            print("🔇 Minimal logging mode - only essential messages")
        elif self.level == 1:
            print("📝 Normal logging mode")
        elif self.level == 2:
            print("📋 Verbose logging mode - detailed iteration info")
        else:
            print("🔍 Debug logging mode - full JSON dumps and validation details")
    
    def log_minimal(self, message: str) -> None:
        """Always shown - essential messages only."""
        print(message)
    
    def log_normal(self, message: str) -> None:
        """Shown in normal+ modes."""
        if self.level >= 1:
            print(message)
    
    def log_verbose(self, message: str) -> None:
        """Shown in verbose+ modes."""
        if self.level >= 2:
            print(message)
    
    def log_debug(self, message: str) -> None:
        """Shown in debug mode only."""
        if self.level >= 3:
            print(message)
    
    def should_show_json(self) -> bool:
        """Whether to show full JSON dumps."""
        return self.show_json_dumps
    
    def should_show_context_details(self) -> bool:
        """Whether to show context size and stats."""
        return self.show_context_stats


# Global logging configuration
log_config = LoggingConfig()


def get_prediction_platform() -> str:
    """Get the prediction platform from environment variable or default to OpenAI."""
    load_dotenv()
    platform = os.getenv("PREDICTION_PLATFORM", "gemini").lower()
    
    # Validate platform
    if not PredictionClientFactory.is_platform_supported(platform):
        available = PredictionClientFactory.get_available_platforms()
        print(f"⚠️  Warning: Unsupported platform '{platform}'. Using 'openai' instead.")
        print(f"   Available platforms: {available}")
        platform = "openai"
    
    return platform

def is_compact_format(context: Dict[str, Any]) -> bool:
    """
    Detect if the context is in compact format.
    Compact format has 'a', 'h' keys instead of 'away_team', 'home_team'.
    """
    if isinstance(context, str):
        try:
            context = json_loads(context)
        except:
            return False
    
    # Check for compact format indicators
    compact_keys = {'a', 'h', 'as', 'hs', 'ap', 'hp', 'L'}
    verbose_keys = {'away_team', 'home_team', 'recent_plays'}
    
    has_compact = any(key in context for key in compact_keys)
    has_verbose = any(key in context for key in verbose_keys)
    
    return has_compact and not has_verbose

def parse_compact_response(response_content: str) -> Dict[str, Any]:
    """
    Parse compact format response and extract play tuple(s).
    
    Args:
        response_content: JSON response from model
        
    Returns:
        Dict with parsed information
    """
    try:
        response_json = json_loads(response_content)
        
        if "y" in response_json:
            y_data = response_json["y"]
            
            # Handle both single play tuple and array of play tuples
            if isinstance(y_data, list) and len(y_data) > 0:
                if isinstance(y_data[0], list):
                    # Multiple plays (e.g., offensive foul + turnover)
                    plays = y_data
                else:
                    # Single play tuple
                    plays = [y_data]
                
                # Convert first/primary play to verbose-like format for compatibility
                primary_play = plays[0]
                if len(primary_play) >= 6:
                    quarter = primary_play[0]
                    time_seconds = primary_play[1]
                    score_array = primary_play[2]
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
                    
                    # Create verbose-compatible response
                    next_play = {
                        "quarter": quarter,
                        "time_remaining": time_remaining,
                        "description": f"Predicted {event_code}",
                        "score": f"AWAY {score_array[0]} - HOME {score_array[1]}",
                        "shot_details": {
                            "team": actor[0] if points else None,
                            "points": points
                        },
                        "_compact_event_code": event_code,
                        "_compact_actor": actor,
                        "_compact_lineup_id": lineup_id,
                        "_all_plays": plays  # Store all plays for offensive foul handling
                    }
                    
                    return {"next_play": next_play}
        
        return {"error": "No 'y' field found in compact response"}
        
    except Exception as e:
        return {"error": f"Failed to parse compact response: {e}"}

def convert_compact_play_to_tuple(next_play: Dict[str, Any], context: Dict[str, Any]) -> list:
    """
    Convert a predicted next_play back to compact tuple format for context updates.
    
    Args:
        next_play: Predicted play in verbose-like format
        context: Current context for reference
        
    Returns:
        list: Play tuple in compact format
    """
    try:
        quarter = next_play.get('quarter', 1)
        
        # Parse time back to seconds
        time_str = next_play.get('time_remaining', '12:00')
        time_parts = time_str.split(':')
        if len(time_parts) >= 2:
            time_seconds = int(time_parts[0]) * 60 + int(time_parts[1])
        else:
            time_seconds = 720  # Default 12:00
        
        # Parse score
        score_str = next_play.get('score', 'AWAY 0 - HOME 0')
        try:
            parts = score_str.split(' - ')
            away_score = int(parts[0].split()[-1])
            home_score = int(parts[1].split()[-1])
            score_array = [away_score, home_score]
        except:
            score_array = [0, 0]
        
        # Get compact-specific fields if available
        actor = next_play.get('_compact_actor', ['A', -1])
        event_code = next_play.get('_compact_event_code', 'unknown')
        lineup_id = next_play.get('_compact_lineup_id', 0)
        
        # Get points from shot_details
        shot_details = next_play.get('shot_details', {})
        points = shot_details.get('points')
        
        # Build tuple
        if points is not None:
            # Scoring play: [q, t, score, actor, event, pts, lineup_id]
            play_tuple = [quarter, time_seconds, score_array, actor, event_code, points, lineup_id]
        else:
            # Non-scoring play: [q, t, score, actor, event, lineup_id]
            play_tuple = [quarter, time_seconds, score_array, actor, event_code, lineup_id]
        
        return play_tuple
        
    except Exception as e:
        print(f"Warning: Failed to convert play to tuple: {e}")
        # Return minimal valid tuple
        return [1, 720, [0, 0], ['A', -1], 'unknown', 0]

def init_prediction_client() -> tuple[BasePredictionClient, Dict[str, str]]:
    """Initialize prediction client based on platform configuration."""
    platform = get_prediction_platform()
    validation_mode = os.getenv("VALIDATION_MODE", "fast")
    
    print(f"🤖 Initializing {platform.upper()} client...")
    print(f"🔧 Validation mode: {validation_mode.upper()} (set VALIDATION_MODE=fast/normal/strict to change)")
    
    try:
        client = PredictionClientFactory.create_client(platform, validation_mode)
        model_config = client.get_model_config()
        
        print(f"✅ {platform.upper()} client initialized successfully")
        print(f"📋 Model 1: {model_config['model_1_id']}")
        print(f"📋 Model 2: {model_config['model_2_id']}")
        
        # Show validation mode benefits
        if validation_mode == "fast":
            print(f"🚀 Fast validation mode: ~10-20% speed boost, essential checks only")
        elif validation_mode == "normal":
            print(f"⚖️ Normal validation mode: balanced speed and validation coverage")
        else:
            print(f"🔍 Strict validation mode: comprehensive checks, slower but thorough")
        
        return client, model_config
        
    except Exception as e:
        print(f"❌ Failed to initialize {platform.upper()} client: {e}")
        raise

def predict_rolling_sequence(game_context: Dict[str, Any], n_iterations: int = 5, skip_stage1: bool = False) -> Dict[str, Any]:
    """
    Rolling prediction pipeline with optimized JSON caching and configurable logging.
    Supports both verbose and compact schema formats.
    1. [Optional] Send game_context to MODEL_1 to get next_plays (initial sequence)
    2. [Optional] Add next_plays to game_context as recent_plays/plays
    3. Send to MODEL_2 to get next_play
    4. Add next_play to END of recent_plays/plays, remove FIRST play (sliding window)
    5. Repeat step 3-4 N times
    
    Args:
        game_context: Dictionary containing team and player data (verbose or compact format)
        n_iterations: Number of times to repeat the rolling prediction
        skip_stage1: If True, assumes game_context already has recent_plays/plays and skips Stage 1
        
    Returns:
        Dict containing all stages and iterations
    """
    client, model_config = init_prediction_client()
    
    # Detect format and initialize appropriate context manager
    is_compact = is_compact_format(game_context)
    log_config.log_normal(f"🔍 Detected format: {'Compact' if is_compact else 'Verbose'}")
    
    if is_compact:
        # Use compact context manager
        optimized_context = CompactGameContext(game_context)
    else:
        # Use verbose context manager
        optimized_context = OptimizedGameContext(game_context)
    
    # Results storage
    results = {
        "stage1_response": None,
        "stage2_responses": [],
        "iterations": []
    }
    
    if skip_stage1:
        log_config.log_normal("\n⚡ SKIPPING STAGE 1: Using pre-loaded plays...")
        
        # Validate that game_context has plays (different field names for different formats)
        if is_compact:
            if "p" not in game_context:
                return {"error": "skip_stage1=True but compact game_context has no 'p' (plays)"}
            plays_count = len(optimized_context.current_plays)
        else:
            if "recent_plays" not in game_context:
                return {"error": "skip_stage1=True but verbose game_context has no 'recent_plays'"}
            plays_count = len(optimized_context.current_recent_plays)
        
        log_config.log_normal(f"✅ Using {plays_count} existing plays")
        results["stage1_response"] = "SKIPPED - Stage 1 bypassed for testing"
        
    else:
        # === STAGE 1: Get initial predictions (first_n_plays mode) ===
        log_config.log_normal("\nSTAGE 1: Getting initial next_plays from first model...")
        log_config.log_normal("Using clean context (no recent_plays) - matching first_N_plays training mode")
        
        # Get clean base context (teams, players, stats only - no plays)
        log_config.log_normal(f"Sending to model: {model_config['model_1_id']}")
        log_config.log_verbose(f"Clean context size: {len(optimized_context._base_json_cached)} characters")
        
        if is_compact:
            log_config.log_verbose("Context contains: teams, players, stats (NO plays array)")
        else:
            log_config.log_verbose("Context contains: teams, players, stats (NO recent_plays)")
        
        try:
            # Get clean context for Stage 1
            if is_compact:
                stage1_context = optimized_context.get_base_context_for_stage1()
            else:
                stage1_context = optimized_context.base_context
            
            log_config.log_verbose(f"Stage 1 context keys: {list(stage1_context.keys())}")
            
            # Verify clean context for Stage 1
            if is_compact:
                if 'p' in stage1_context:
                    log_config.log_normal("WARNING: Stage 1 context contains 'p' (plays) - this should not happen!")
                else:
                    log_config.log_verbose("Stage 1 context is clean (no plays array)")
            else:
                if 'recent_plays' in stage1_context:
                    log_config.log_normal("WARNING: Stage 1 context contains recent_plays - this should not happen!")
                else:
                    log_config.log_verbose("Stage 1 context is clean (no recent_plays)")
            
            # Stage 1 API call with base context only (matching first_N_plays training mode)
            stage1_content, stage1_usage, stage1_game_ended, stage1_needs_rollback, stage1_termination_info = client.predict_with_validation(
                context=stage1_context,  # Base context without recent_plays
                model_id=model_config['model_1_id'],
                max_tokens=8000,  # Higher limit for Stage 1 (generates ~20 plays)
                temperature=1.01,
                max_retries=6,
                stage1_mode=True  # Use Stage 1 validation (expects next_plays array)
            )
            
            # Store Stage 1 termination information if present (shouldn't normally happen)
            if stage1_termination_info:
                log_config.log_normal(f"⚠️ Unexpected Stage 1 termination: {stage1_termination_info.get('validation_termination_type', 'unknown')}")
                results["stage1_validation_termination"] = stage1_termination_info
            
            # Note: Rollback shouldn't happen in Stage 1 since there are no recent_plays
            if stage1_needs_rollback:
                log_config.log_normal("⚠️ Unexpected rollback signal in Stage 1 - ignoring")
            
            if stage1_game_ended:
                log_config.log_minimal("Game ended during Stage 1 - terminating prediction sequence")
                results["termination_reason"] = "Game ended during Stage 1"
                return results
            
            log_config.log_verbose(f"Stage 1 tokens: {stage1_usage.get('completion_tokens', 'N/A')} / 5000")
            log_config.log_normal("Stage 1 completed!")
            
            results["stage1_response"] = stage1_content
            
            # Parse Stage 1 response
            try:
                stage1_json = json_loads(stage1_content)
            except (json.JSONDecodeError, ValueError) as e:
                log_config.log_minimal(f"❌ Failed to parse Stage 1 response as JSON: {e}")
                return {"error": f"Stage 1 JSON parse error: {e}"}
            
            # Update context with initial plays (format-dependent)
            if is_compact:
                # Compact format: expect plays array in response
                if "p" in stage1_json:
                    optimized_context.update_plays(stage1_json["p"])
                    log_config.log_normal(f"✅ Added {len(stage1_json['p'])} play tuples to context")
                elif "next_plays" in stage1_json:
                    # Fallback: convert verbose next_plays to compact format
                    # This is for backward compatibility during transition
                    log_config.log_normal("Converting verbose next_plays to compact format")
                    next_plays = stage1_json["next_plays"]
                    # For now, create minimal play tuples from verbose plays
                    play_tuples = []
                    for play in next_plays:
                        # Convert each verbose play to compact tuple
                        quarter = play.get('quarter', 1)
                        time_seconds = 720  # Default 12:00
                        score_array = [0, 0]  # Default scores
                        actor = ['A', -1]  # Default actor
                        event_code = 'unknown'  # Default event
                        lineup_id = 0  # Default lineup
                        
                        play_tuple = [quarter, time_seconds, score_array, actor, event_code, lineup_id]
                        play_tuples.append(play_tuple)
                    
                    optimized_context.update_plays(play_tuples)
                    log_config.log_normal(f"✅ Converted and added {len(play_tuples)} play tuples to context")
                else:
                    log_config.log_normal("⚠️ Warning: No 'p' or 'next_plays' found in Stage 1 response")
                    optimized_context.update_plays([])
            else:
                # Verbose format: expect next_plays array
                if "next_plays" in stage1_json:
                    optimized_context.update_recent_plays(stage1_json["next_plays"])
                    log_config.log_normal(f"✅ Added {len(stage1_json['next_plays'])} recent_plays to context")
                else:
                    log_config.log_normal("⚠️ Warning: No 'next_plays' found in Stage 1 response")
                    optimized_context.update_recent_plays([])
                
        except Exception as e:
            log_config.log_minimal(f"❌ Error in Stage 1: {e}")
            raise
    
    # === ROLLING ITERATIONS ===
    log_config.log_normal(f"\n🔄 Starting {n_iterations} rolling iterations...")
    
    try:
        for iteration in range(n_iterations):
            log_config.log_normal(f"\n--- ITERATION {iteration + 1}/{n_iterations} ---")
            
            # Get optimized JSON (cached where possible)
            iteration_json = optimized_context.get_json()
            log_config.log_normal(f"📡 Sending to model: {model_config['model_2_id']}")
            log_config.log_verbose(f"📊 Context size: {len(iteration_json)} characters")
            
            # Display plays count based on format
            if is_compact:
                plays_count = len(optimized_context.current_plays)
                log_config.log_verbose(f"📋 Current plays count: {plays_count}")
                current_plays = optimized_context.current_plays
            else:
                plays_count = len(optimized_context.current_recent_plays)
                log_config.log_verbose(f"📋 Current recent_plays count: {plays_count}")
                current_plays = optimized_context.current_recent_plays
            
            # Show essential game state info (format-dependent)
            if current_plays:
                if is_compact:
                    # Compact format: play tuples [q, t, score, actor, event, (pts), lineup_id]
                    latest_play = current_plays[-1]
                    if len(latest_play) >= 3:
                        current_quarter = latest_play[0]
                        time_seconds = latest_play[1]
                        score_array = latest_play[2] if len(latest_play[2]) >= 2 else [0, 0]
                        
                        # Convert time back to MM:SS
                        minutes = time_seconds // 60
                        seconds = time_seconds % 60
                        current_time = f"{minutes:02d}:{seconds:02d}"
                        current_score = f"AWAY {score_array[0]} - HOME {score_array[1]}"
                        
                        log_config.log_normal(f"🏀 Game State: Q{current_quarter} {current_time} | {current_score}")
                        
                        # Show last 5 play tuples for context
                        log_config.log_normal("📋 Recent play tuples:")
                        recent_to_show = current_plays[-5:]
                        for i, play_tuple in enumerate(recent_to_show, 1):
                            if len(play_tuple) >= 5:
                                q = play_tuple[0]
                                t_sec = play_tuple[1]
                                t_min = t_sec // 60
                                t_s = t_sec % 60
                                event = play_tuple[4] if len(play_tuple) > 4 else 'unknown'
                                log_config.log_normal(f"   {i}. Q{q} [{t_min:02d}:{t_s:02d}] {event}")
                else:
                    # Verbose format: play objects
                    latest_play = current_plays[-1]
                    current_score = latest_play.get('score', 'N/A')
                    current_quarter = latest_play.get('quarter', 'N/A')
                    current_time = latest_play.get('time_remaining', 'N/A')
                    
                    log_config.log_normal(f"🏀 Game State: Q{current_quarter} {current_time} | {current_score}")
                    
                    # Show last 5 plays for context
                    log_config.log_normal("📋 Recent plays context:")
                    recent_to_show = current_plays[-5:]
                    for i, play in enumerate(recent_to_show, 1):
                        play_desc = play.get('description', 'No description')[:60]
                        play_time = play.get('time_remaining', 'N/A')
                        log_config.log_normal(f"   {i}. [{play_time}] {play_desc}")
            
            # Show caching stats in debug mode
            if log_config.should_show_context_details():
                cache_stats = optimized_context.get_cache_stats()
                log_config.log_debug(f"🚀 Cache stats: {cache_stats}")
            
            # 🔍 LOG: Show the input context being sent to the model (debug mode only)
            if log_config.should_show_json():
                log_config.log_debug("\n" + "="*60)
                log_config.log_debug(f"📤 INPUT TO MODEL (Iteration {iteration + 1}):")
                log_config.log_debug("="*60)
                
                if is_compact:
                    log_config.log_debug("📋 PLAY TUPLES being sent:")
                    if current_plays:
                        for i, play_tuple in enumerate(current_plays):
                            if len(play_tuple) >= 5:
                                q, t_sec, score, actor, event = play_tuple[0], play_tuple[1], play_tuple[2], play_tuple[3], play_tuple[4]
                                t_min = t_sec // 60
                                t_s = t_sec % 60
                                log_config.log_debug(f"   {i+1:2d}. Q{q} [{t_min:02d}:{t_s:02d}] {event} by {actor}")
                    else:
                        log_config.log_debug("   ❌ NO play tuples in context!")
                else:
                    log_config.log_debug("📋 RECENT_PLAYS being sent:")
                    if current_plays:
                        for i, play in enumerate(current_plays):
                            play_desc = play.get('description', 'No description')
                            play_time = play.get('time_remaining', 'No time')
                            log_config.log_debug(f"   {i+1:2d}. [{play_time}] {play_desc}")
                    else:
                        log_config.log_debug("   ❌ NO recent_plays in context!")
                        
                log_config.log_debug("="*60)
                log_config.log_debug("📤 FULL INPUT JSON:")
                log_config.log_debug(iteration_json)
                log_config.log_debug("="*60 + "\n")
            
            # Stage 2 API call with validation and rollback handling
            stage2_content, stage2_usage, stage2_game_ended, stage2_needs_rollback, termination_info = client.predict_with_validation(
                context=optimized_context.get_context_dict(),
                model_id=model_config['model_2_id'],
                max_tokens=5000,
                temperature=1.01,
                max_retries=6
            )
            
            # Store termination information for later use
            if termination_info:
                log_config.log_normal(f"🛑 Validation termination info captured: {termination_info.get('validation_termination_type', 'unknown')}")
                # Store in the results for the orchestrator to access
                results["validation_termination"] = termination_info
            
            # Handle timestamp rollback scenario
            if stage2_needs_rollback:
                log_config.log_normal(f"🔄 Timestamp rollback triggered during iteration {iteration + 1}")
                
                # Get the rollback snapshot from the validator
                rollback_snapshot = client.validator.get_rollback_snapshot()
                if rollback_snapshot is not None:
                    # Restore the context to the rollback state
                    optimized_context.restore_recent_plays(rollback_snapshot)
                    log_config.log_normal(f"✅ Game state restored to {len(rollback_snapshot)} plays")
                    
                    # Skip to next iteration with restored state
                    continue
                else:
                    log_config.log_normal(f"⚠️ Rollback snapshot was None - continuing with current state")
            
            if stage2_game_ended:
                log_config.log_minimal(f"🏁 Game ended during iteration {iteration + 1} - terminating prediction sequence")
                results["termination_reason"] = f"Game ended at iteration {iteration + 1}"
                # Still process this final response before breaking
                # Continue to process the response below, then break after processing
                should_break_after_processing = True
            else:
                should_break_after_processing = False
            
            log_config.log_verbose(f"📊 Stage 2 tokens: {stage2_usage.get('completion_tokens', 'N/A')} / 5000")
            
            # 📝 LOG: Print full model response for debugging (debug mode only)
            if log_config.should_show_json():
                log_config.log_debug("\n" + "="*60)
                log_config.log_debug(f"🔍 FULL MODEL RESPONSE (Iteration {iteration + 1}):")
                log_config.log_debug("="*60)
                log_config.log_debug(stage2_content)
                log_config.log_debug("="*60 + "\n")
            
            # Store this iteration's response
            results["stage2_responses"].append(stage2_content)
            
            # Parse the response based on format
            try:
                stage2_json = json_loads(stage2_content)
            except (json.JSONDecodeError, ValueError) as e:
                log_config.log_minimal(f"❌ Failed to parse iteration {iteration + 1} response as JSON: {e}")
                results["iterations"].append({
                    "iteration": iteration + 1,
                    "error": f"JSON parse error: {e}",
                    "raw_response": stage2_content
                })
                continue
            
            # Extract and process the predicted play (format-dependent)
            next_play = None
            
            if is_compact:
                # Compact format: expect "y" field with play tuple(s)
                if "y" in stage2_json:
                    # Parse compact response
                    parsed_response = parse_compact_response(stage2_content)
                    if "next_play" in parsed_response:
                        next_play = parsed_response["next_play"]
                        
                        # Show new play
                        play_desc = next_play.get('description', 'Predicted play')
                        new_score = next_play.get('score', 'N/A')
                        new_quarter = next_play.get('quarter', 'N/A')
                        new_time = next_play.get('time_remaining', 'N/A')
                        event_code = next_play.get('_compact_event_code', 'unknown')
                        
                        log_config.log_normal(f"✅ NEW PLAY: {event_code} - {play_desc}")
                        log_config.log_normal(f"🏀 Updated State: Q{new_quarter} {new_time} | {new_score}")
                        
                        # Check for scoring information
                        shot_details = next_play.get('shot_details', {})
                        points = shot_details.get('points')
                        if points is not None and points > 0:
                            team = shot_details.get('team', 'Unknown')
                            log_config.log_normal(f"🎯 SCORING PLAY: {team} +{points} points!")
                        else:
                            log_config.log_verbose(f"📋 Non-scoring play ({event_code})")
                        
                        # Convert to compact tuple and update context
                        play_tuple = convert_compact_play_to_tuple(next_play, optimized_context.get_context_dict())
                        from config.settings import DEFAULT_N_TOTAL_PLAYS
                        optimized_context.add_play_tuple(play_tuple, DEFAULT_N_TOTAL_PLAYS)
                    else:
                        log_config.log_normal(f"⚠️ Warning: Failed to parse compact response: {parsed_response.get('error', 'Unknown error')}")
                else:
                    log_config.log_normal(f"⚠️ Warning: No 'y' field found in compact format response")
            else:
                # Verbose format: expect "next_play" field
                if "next_play" in stage2_json:
                    next_play = stage2_json["next_play"]
                    
                    # Show new play with essential game state
                    play_desc = next_play.get('description', 'No description')
                    new_score = next_play.get('score', 'N/A')
                    new_quarter = next_play.get('quarter', 'N/A')
                    new_time = next_play.get('time_remaining', 'N/A')
                    
                    log_config.log_normal(f"✅ NEW PLAY: {play_desc}")
                    log_config.log_normal(f"🏀 Updated State: Q{new_quarter} {new_time} | {new_score}")
                    
                    # Check for scoring information
                    if "shot_details" in next_play:
                        shot_details = next_play["shot_details"]
                        points = shot_details.get("points") if shot_details else None
                        if shot_details and points is not None and points > 0:
                            log_config.log_normal(f"🎯 SCORING PLAY: {shot_details.get('team', 'Unknown')} +{points} points!")
                        else:
                            points_display = points if points is not None else 'N/A'
                            log_config.log_verbose(f"📋 Non-scoring play (points: {points_display})")
                    else:
                        log_config.log_verbose("⚠️ WARNING: No 'shot_details' field found in next_play")
                    
                    # Update the sliding window using verbose context
                    from config.settings import DEFAULT_N_TOTAL_PLAYS
                    optimized_context.add_play_and_slide(next_play, DEFAULT_N_TOTAL_PLAYS)
                else:
                    log_config.log_normal(f"⚠️ Warning: No 'next_play' found in verbose format response")
            
            # Continue only if we successfully extracted a play
            if next_play is None:
                results["iterations"].append({
                    "iteration": iteration + 1,
                    "error": "No valid play found in response",
                    "raw_response": stage2_content
                })
                # Break if game ended (even with invalid response)
                if should_break_after_processing:
                    log_config.log_minimal("🏁 Breaking out of prediction loop - game ended")
                    break
                continue
            
            # Show sliding window info (if we successfully got a play)
            if next_play and log_config.should_show_context_details():
                if is_compact:
                    current_plays_count = len(optimized_context.current_plays)
                    if current_plays_count >= DEFAULT_N_TOTAL_PLAYS:
                        log_config.log_verbose("🔄 Sliding window: Added new play tuple, removed oldest")
                    else:
                        log_config.log_verbose(f"📈 Window growing: Now {current_plays_count} play tuples")
                else:
                    current_plays_count = len(optimized_context.current_recent_plays)
                    if current_plays_count >= DEFAULT_N_TOTAL_PLAYS:
                        log_config.log_verbose("🔄 Sliding window: Added new play, removed oldest play")
                    else:
                        log_config.log_verbose(f"📈 Window growing: Now {current_plays_count} plays")
            
            # Store results for this iteration
            if next_play:
                if is_compact:
                    plays_count = len(optimized_context.current_plays)
                else:
                    plays_count = len(optimized_context.current_recent_plays)
                    
                results["iterations"].append({
                    "iteration": iteration + 1,
                    "next_play": next_play,
                    "raw_response": stage2_content,
                    "recent_plays_count": plays_count
                })
                
                # Break if game ended
                if should_break_after_processing:
                    log_config.log_minimal("🏁 Breaking out of prediction loop - game ended")
                    break
        
        log_config.log_normal(f"\n✅ Rolling sequence completed! {n_iterations} iterations done.")
        
        # 📊 SCORING ANALYSIS SUMMARY
        scoring_plays = 0
        non_scoring_plays = 0
        for iteration_result in results["iterations"]:
            if "next_play" in iteration_result:
                next_play = iteration_result["next_play"]
                # Check shot_details for scoring information, handling None values
                if "shot_details" in next_play and next_play["shot_details"]:
                    points = next_play["shot_details"].get("points")
                    if points is not None and points > 0:
                        scoring_plays += 1
                    else:
                        non_scoring_plays += 1
                else:
                    non_scoring_plays += 1
        
        log_config.log_normal(f"\n📈 SCORING SUMMARY:")
        log_config.log_normal(f"   🏀 Scoring plays: {scoring_plays}/{scoring_plays + non_scoring_plays}")
        log_config.log_normal(f"   📋 Non-scoring plays: {non_scoring_plays}/{scoring_plays + non_scoring_plays}")
        if scoring_plays + non_scoring_plays > 0:
            scoring_rate = (scoring_plays / (scoring_plays + non_scoring_plays)) * 100
            log_config.log_normal(f"   📊 Scoring rate: {scoring_rate:.1f}%")
        
        # 🚀 PERFORMANCE SUMMARY (debug mode)
        if log_config.should_show_context_details():
            final_cache_stats = optimized_context.get_cache_stats()
            log_config.log_verbose(f"\n🚀 PERFORMANCE SUMMARY:")
            log_config.log_verbose(f"   📊 Final cache stats: {final_cache_stats}")
            cache_hit_ratio = (final_cache_stats['plays_cache_size'] / max(n_iterations, 1)) * 100
            log_config.log_verbose(f"   ⚡ Estimated JSON cache efficiency: {cache_hit_ratio:.1f}%")
        
        return results
        
    except Exception as e:
        log_config.log_minimal(f"❌ Error in rolling iterations: {e}")
        
        # Check if this was a validation failure and capture details
        if "Response validation failed" in str(e) and hasattr(client, 'get_last_validation_failure_info'):
            validation_failure_info = client.get_last_validation_failure_info()
            if validation_failure_info:
                log_config.log_normal(f"🔍 Captured validation failure details:")
                log_config.log_normal(f"   Most common reason: {validation_failure_info.get('validation_most_common_reason', 'unknown')}")
                log_config.log_normal(f"   Most common error type: {validation_failure_info.get('validation_most_common_error_type', 'unknown')}")
                log_config.log_normal(f"   Total failed attempts: {validation_failure_info.get('validation_total_failed_attempts', 0)}")
                
                # Store in results for orchestrator to access (even though we're about to raise)
                # We can store this in a global or modify the exception
                import builtins
                builtins._last_validation_failure = validation_failure_info
        
        raise

def main():
    """Main function to test the prediction system with performance optimizations."""
    
    print("🏀 NBA Multi-Platform Play Prediction Test (OPTIMIZED)")
    print("=" * 60)
    
    # Display platform information
    platform = get_prediction_platform()
    available_platforms = PredictionClientFactory.get_available_platforms()
    print(f"📱 Platform: {platform.upper()}")
    print(f"🔧 Available platforms: {', '.join(available_platforms)}")
    
    # Display optimization information
    validation_mode = os.getenv("VALIDATION_MODE", "fast")
    print(f"\n🚀 Performance Optimizations Active:")
    print(f"   • JSON Caching: ✅ Enabled")
    print(f"   • Logging Level: {log_config.level} (set PREDICTION_LOG_LEVEL=0-3)")
    print(f"   • Fast JSON Library: {'ujson' if 'ujson' in globals() else 'standard json'}")
    print(f"   • Double Serialization: ❌ Eliminated")
    print(f"   • Validation Mode: {validation_mode.upper()} (set VALIDATION_MODE=fast/normal/strict)")
    
    if validation_mode == "fast":
        print(f"   • Response Validation: ⚡ Fast mode - 10-20% speed boost")
    elif validation_mode == "normal":
        print(f"   • Response Validation: ⚖️ Normal mode - balanced performance")
    else:
        print(f"   • Response Validation: 🔍 Strict mode - comprehensive checks")
    
    if platform == "gemini":
        print("\n💡 Gemini Configuration Notes:")
        print("   • Ensure you've run: gcloud auth login")
        print("   • Set project: gcloud config set project YOUR_PROJECT_ID")
        print("   • Update model endpoint IDs in GEMINI_MODEL_1_ENDPOINT and GEMINI_MODEL_2_ENDPOINT")
        print("   • Or modify the _get_default_models() method in GeminiPredictionClient")
    elif platform == "openai":
        print("\n💡 OpenAI Configuration Notes:")
        print("   • Ensure OPENAI_API_KEY environment variable is set")
        print("   • Default model IDs are configured for the provided fine-tuned models")
    
    print("\n🎛️ Logging Levels:")
    print("   • 0: Minimal (essential messages only)")
    print("   • 1: Normal (default - standard progress)")
    print("   • 2: Verbose (detailed iteration info + cache stats)")
    print("   • 3: Debug (full JSON dumps + validation details)")
    print("   Set with: PREDICTION_LOG_LEVEL=0 (or 1,2,3)")
    
    print("\n" + "=" * 60)
    
    # ==================================================================================
    # 🎯 TESTING MODE SELECTION - Change this to switch between modes
    # ==================================================================================
    TEST_MODE = "full_pipeline"  # Options: "full_pipeline" or "skip_stage1"
    # ==================================================================================
    
    if TEST_MODE == "full_pipeline":
        print("🔄 Mode: FULL PIPELINE (Stage 1 + Rolling Iterations)")
        
        # Example team and player data (original context without recent_plays)
        game_context = {"away_team":{"name":"ORL","stats":{"OEFF":113.7,"DEFF":112.7,"PACE":96.5,"REST_DAYS":2},"players":[{"name":"Franz Wagner","profile":{"offense":2.34,"defense":1.42,"shot_selection":0.69,"efficiency":-0.67,"MPG":33,"usage":26}},{"name":"Paolo Banchero","profile":{"offense":4.14,"defense":1.02,"shot_selection":1.28,"efficiency":-1.04,"MPG":35,"usage":30}},{"name":"Jonathan Isaac","profile":{"offense":-0.71,"defense":1.02,"shot_selection":0.3,"efficiency":-0.5,"MPG":16,"usage":17}},{"name":"Gary Harris","profile":{"offense":-0.41,"defense":0.52,"shot_selection":-1.29,"efficiency":-0.33,"MPG":24,"usage":12}},{"name":"Jalen Suggs","profile":{"offense":1.44,"defense":2.5,"shot_selection":-0.34,"efficiency":-1.05,"MPG":27,"usage":20}},{"name":"Wendell Carter Jr.","profile":{"offense":0.42,"defense":0.76,"shot_selection":0.46,"efficiency":-0.96,"MPG":25,"usage":18}},{"name":"Joe Ingles","profile":{"offense":0.18,"defense":-0.52,"shot_selection":-1.26,"efficiency":-0.82,"MPG":17,"usage":11}},{"name":"Markelle Fultz","profile":{"offense":-0.09,"defense":0.42,"shot_selection":0.52,"efficiency":0.67,"MPG":21,"usage":19}},{"name":"Cole Anthony","profile":{"offense":1.16,"defense":0.87,"shot_selection":0.48,"efficiency":-0.2,"MPG":22,"usage":24}},{"name":"Moritz Wagner","profile":{"offense":0.68,"defense":0.16,"shot_selection":1.33,"efficiency":-1.71,"MPG":18,"usage":23}},{"name":"Caleb Houstan","profile":{"offense":-1.09,"defense":-1.29,"shot_selection":-2.17,"efficiency":-0.26,"MPG":14,"usage":12}}]},"home_team":{"name":"CLE","stats":{"OEFF":114.9,"DEFF":112.5,"PACE":97.2,"REST_DAYS":2},"players":[{"name":"Max Strus","profile":{"offense":1.48,"defense":1.24,"shot_selection":-1.23,"efficiency":-0.44,"MPG":32,"usage":17}},{"name":"Evan Mobley","profile":{"offense":1.49,"defense":2.77,"shot_selection":1.39,"efficiency":-1.13,"MPG":31,"usage":21}},{"name":"Jarrett Allen","profile":{"offense":1.82,"defense":1.41,"shot_selection":2.04,"efficiency":-1.77,"MPG":32,"usage":20}},{"name":"Donovan Mitchell","profile":{"offense":3.79,"defense":2.55,"shot_selection":0.2,"efficiency":-1.04,"MPG":35,"usage":31}},{"name":"Darius Garland","profile":{"offense":3.38,"defense":0.87,"shot_selection":-0.12,"efficiency":-0.88,"MPG":33,"usage":25}},{"name":"Caris LeVert","profile":{"offense":1.86,"defense":1.08,"shot_selection":0.15,"efficiency":-0.01,"MPG":29,"usage":23}},{"name":"Georges Niang","profile":{"offense":0.07,"defense":0.16,"shot_selection":-1.34,"efficiency":-0.52,"MPG":22,"usage":18}},{"name":"Isaac Okoro","profile":{"offense":0.17,"defense":0.95,"shot_selection":0.19,"efficiency":-0.68,"MPG":27,"usage":14}}]}}
        
        print("📋 Game Context Summary:")
        print(f"   Away Team: {game_context['away_team']['name']} ({len(game_context['away_team']['players'])} players)")
        print(f"   Home Team: {game_context['home_team']['name']} ({len(game_context['home_team']['players'])} players)")
        print()
        
    elif TEST_MODE == "skip_stage1":
        print("⚡ Mode: SKIP STAGE 1 (Direct to Rolling Iterations)")
        
        # Pre-built context with recent_plays already included (MIN vs POR game)
        # Load your JSON data and convert null to None
        import json
        json_data = """{"away_team":{"name":"ATL","stats":{"OEFF":115.6,"DEFF":117.8,"PACE":100.7,"REST_DAYS":3},"players":[{"name":"De'Andre Hunter","profile":{"offense":0.54,"defense":1.77,"shot_selection":0.21,"efficiency":-0.23,"MPG":31,"usage":18}},{"name":"Jalen Johnson","profile":{"offense":1.15,"defense":1.58,"shot_selection":0.4,"efficiency":-1.47,"MPG":30,"usage":18}},{"name":"Clint Capela","profile":{"offense":-0.17,"defense":2.14,"shot_selection":1.75,"efficiency":-0.7,"MPG":25,"usage":14}},{"name":"Bogdan Bogdanovic","profile":{"offense":0.79,"defense":1.11,"shot_selection":-1.72,"efficiency":-0.25,"MPG":26,"usage":21}},{"name":"Dejounte Murray","profile":{"offense":3.05,"defense":1.02,"shot_selection":0.28,"efficiency":-0.81,"MPG":35,"usage":25}},{"name":"Onyeka Okongwu","profile":{"offense":0.28,"defense":2.02,"shot_selection":1.71,"efficiency":-1.65,"MPG":22,"usage":13}},{"name":"AJ Griffin","profile":{"offense":-0.87,"defense":-1.7,"shot_selection":-1.74,"efficiency":-1.35,"MPG":9,"usage":14}},{"name":"Saddiq Bey","profile":{"offense":0.13,"defense":0.15,"shot_selection":-0.44,"efficiency":-0.45,"MPG":28,"usage":17}},{"name":"Trent Forrest","profile":{"offense":-0.51,"defense":-1.36,"shot_selection":0.79,"efficiency":1.22,"MPG":9,"usage":3}},{"name":"Garrison Mathews","profile":{"offense":-0.29,"defense":-0.51,"shot_selection":-1.47,"efficiency":-1.46,"MPG":3,"usage":15}}]},"home_team":{"name":"DET","stats":{"OEFF":109.0,"DEFF":115.9,"PACE":97.7,"REST_DAYS":2},"players":[{"name":"Ausar Thompson","profile":{"offense":1.04,"defense":3.99,"shot_selection":0.84,"efficiency":0.53,"MPG":32,"usage":19}},{"name":"Isaiah Stewart","profile":{"offense":0.62,"defense":0.79,"shot_selection":-0.12,"efficiency":-0.63,"MPG":33,"usage":16}},{"name":"Marvin Bagley III","profile":{"offense":0.45,"defense":-0.24,"shot_selection":1.74,"efficiency":-1.16,"MPG":20,"usage":22}},{"name":"Cade Cunningham","profile":{"offense":3.83,"defense":1.41,"shot_selection":0.23,"efficiency":0.2,"MPG":36,"usage":31}},{"name":"Killian Hayes","profile":{"offense":0.92,"defense":1.67,"shot_selection":0.08,"efficiency":0.58,"MPG":30,"usage":16}},{"name":"Alec Burks","profile":{"offense":0.94,"defense":0.05,"shot_selection":0.01,"efficiency":-0.91,"MPG":24,"usage":20}},{"name":"James Wiseman","profile":{"offense":-0.87,"defense":-0.11,"shot_selection":0.51,"efficiency":-0.6,"MPG":10,"usage":22}},{"name":"Kevin Knox II","profile":{"offense":-0.21,"defense":-0.34,"shot_selection":-0.75,"efficiency":-0.42,"MPG":26,"usage":14}},{"name":"Marcus Sasser","profile":{"offense":0.44,"defense":0.07,"shot_selection":-1.09,"efficiency":-0.7,"MPG":20,"usage":17}},{"name":"Jaden Ivey","profile":{"offense":0.68,"defense":0.96,"shot_selection":0.27,"efficiency":-0.75,"MPG":20,"usage":23}}]},"recent_plays":[{"quarter":1,"time_remaining":"06:15","score":"ATL 18 - DET 11","players_on_court":[{"team":"ATL","players":["Clint Capela","Bogdan Bogdanovic","Dejounte Murray","De'Andre Hunter","Jalen Johnson"]},{"team":"DET","players":["Marvin Bagley III","Ausar Thompson","Cade Cunningham","Alec Burks","Isaiah Stewart"]}],"player":"Dejounte Murray","description":"MURRAY DEF.REBOUND","shot_details":{"team":null,"points":null}},{"quarter":1,"time_remaining":"06:12","score":"ATL 21 - DET 11","players_on_court":[{"team":"ATL","players":["Clint Capela","Bogdan Bogdanovic","Dejounte Murray","De'Andre Hunter","Jalen Johnson"]},{"team":"DET","players":["Marvin Bagley III","Ausar Thompson","Cade Cunningham","Alec Burks","Isaiah Stewart"]}],"player":"Bogdan Bogdanovic","description":"Bogdanovic 26' 3PT Running Jump Shot","shot_details":{"team":"ATL","points":3}},{"quarter":1,"time_remaining":"05:57","score":"ATL 21 - DET 14","players_on_court":[{"team":"ATL","players":["Clint Capela","Bogdan Bogdanovic","Dejounte Murray","De'Andre Hunter","Jalen Johnson"]},{"team":"DET","players":["Marvin Bagley III","Ausar Thompson","Cade Cunningham","Alec Burks","Isaiah Stewart"]}],"player":"Isaiah Stewart","description":"Stewart 28' 3PT Jump Shot","shot_details":{"team":"DET","points":3}},{"quarter":1,"time_remaining":"05:41","score":"ATL 21 - DET 14","players_on_court":[{"team":"ATL","players":["Clint Capela","Bogdan Bogdanovic","Dejounte Murray","De'Andre Hunter","Jalen Johnson"]},{"team":"DET","players":["Marvin Bagley III","Ausar Thompson","Cade Cunningham","Alec Burks","Isaiah Stewart"]}],"player":"Bogdan Bogdanovic","description":"MISS Bogdanovic 26' 3PT Jump Shot","shot_details":{"team":"ATL","points":0}},{"quarter":1,"time_remaining":"05:39","score":"ATL 21 - DET 14","players_on_court":[{"team":"ATL","players":["Clint Capela","Bogdan Bogdanovic","Dejounte Murray","De'Andre Hunter","Jalen Johnson"]},{"team":"DET","players":["Marvin Bagley III","Ausar Thompson","Cade Cunningham","Alec Burks","Isaiah Stewart"]}],"player":"Marvin Bagley III","description":"BAGLEY III DEF.REBOUND","shot_details":{"team":null,"points":null}},{"quarter":1,"time_remaining":"05:26","score":"ATL 21 - DET 16","players_on_court":[{"team":"ATL","players":["Clint Capela","Bogdan Bogdanovic","Dejounte Murray","De'Andre Hunter","Jalen Johnson"]},{"team":"DET","players":["Marvin Bagley III","Ausar Thompson","Cade Cunningham","Alec Burks","Isaiah Stewart"]}],"player":"Ausar Thompson","description":"Thompson 6' Turnaround Jump Shot","shot_details":{"team":"DET","points":2}},{"quarter":1,"time_remaining":"05:06","score":"ATL 23 - DET 16","players_on_court":[{"team":"ATL","players":["Clint Capela","Bogdan Bogdanovic","Dejounte Murray","De'Andre Hunter","Jalen Johnson"]},{"team":"DET","players":["Marvin Bagley III","Ausar Thompson","Cade Cunningham","Alec Burks","Isaiah Stewart"]}],"player":"Dejounte Murray","description":"Murray 19' Pullup Jump Shot","shot_details":{"team":"ATL","points":2}},{"quarter":1,"time_remaining":"04:55","score":"ATL 23 - DET 16","players_on_court":[{"team":"ATL","players":["Clint Capela","Bogdan Bogdanovic","Dejounte Murray","De'Andre Hunter","Jalen Johnson"]},{"team":"DET","players":["Marvin Bagley III","Ausar Thompson","Cade Cunningham","Alec Burks","Isaiah Stewart"]}],"player":"Clint Capela","description":"Capela P.FOUL on Cade Cunningham","shot_details":{"team":null,"points":null}},{"quarter":1,"time_remaining":"04:55","score":"ATL 23 - DET 16","players_on_court":[{"team":"ATL","players":["Clint Capela","Bogdan Bogdanovic","Dejounte Murray","De'Andre Hunter","Jalen Johnson"]},{"team":"DET","players":["Marvin Bagley III","Ausar Thompson","Cade Cunningham","Alec Burks","James Wiseman"]}],"player":null,"description":"SUBS: Wiseman FOR Stewart, Knox II FOR Bagley III, Okongwu FOR Johnson, Griffin FOR Bogdanovic, Bey FOR Capela","shot_details":{"team":null,"points":null}},{"quarter":1,"time_remaining":"04:46","score":"ATL 23 - DET 16","players_on_court":[{"team":"ATL","players":["Saddiq Bey","AJ Griffin","Dejounte Murray","De'Andre Hunter","Onyeka Okongwu"]},{"team":"DET","players":["Kevin Knox II","Ausar Thompson","Cade Cunningham","Alec Burks","James Wiseman"]}],"player":"Alec Burks","description":"MISS Burks 26' 3PT Jump Shot","shot_details":{"team":"DET","points":0}},{"quarter":1,"time_remaining":"04:44","score":"ATL 23 - DET 16","players_on_court":[{"team":"ATL","players":["Saddiq Bey","AJ Griffin","Dejounte Murray","De'Andre Hunter","Onyeka Okongwu"]},{"team":"DET","players":["Kevin Knox II","Ausar Thompson","Cade Cunningham","Alec Burks","James Wiseman"]}],"player":"Onyeka Okongwu","description":"OKONGWU DEF.REBOUND","shot_details":{"team":null,"points":null}},{"quarter":1,"time_remaining":"04:36","score":"ATL 26 - DET 16","players_on_court":[{"team":"ATL","players":["Saddiq Bey","AJ Griffin","Dejounte Murray","De'Andre Hunter","Onyeka Okongwu"]},{"team":"DET","players":["Kevin Knox II","Ausar Thompson","Cade Cunningham","Alec Burks","James Wiseman"]}],"player":"AJ Griffin","description":"Griffin 26' 3PT Pullup Jump Shot","shot_details":{"team":"ATL","points":3}},{"quarter":1,"time_remaining":"04:24","score":"ATL 26 - DET 16","players_on_court":[{"team":"ATL","players":["Saddiq Bey","AJ Griffin","Dejounte Murray","De'Andre Hunter","Onyeka Okongwu"]},{"team":"DET","players":["Kevin Knox II","Ausar Thompson","Cade Cunningham","Alec Burks","James Wiseman"]}],"player":"Dejounte Murray","description":"Murray S.FOUL on James Wiseman","shot_details":{"team":null,"points":null}},{"quarter":1,"time_remaining":"04:24","score":"ATL 26 - DET 16","players_on_court":[{"team":"ATL","players":["Saddiq Bey","AJ Griffin","Dejounte Murray","De'Andre Hunter","Onyeka Okongwu"]},{"team":"DET","players":["Kevin Knox II","Ausar Thompson","Cade Cunningham","Alec Burks","James Wiseman"]}],"player":"James Wiseman","description":"MISS Wiseman Free Throw 1 of 2","shot_details":{"team":null,"points":null}},{"quarter":1,"time_remaining":"04:24","score":"ATL 26 - DET 16","players_on_court":[{"team":"ATL","players":["Saddiq Bey","AJ Griffin","Dejounte Murray","De'Andre Hunter","Onyeka Okongwu"]},{"team":"DET","players":["Kevin Knox II","Ausar Thompson","Cade Cunningham","Alec Burks","James Wiseman"]}],"player":null,"description":"PISTONS Rebound","shot_details":{"team":null,"points":null}},{"quarter":1,"time_remaining":"04:24","score":"ATL 26 - DET 16","players_on_court":[{"team":"ATL","players":["Saddiq Bey","AJ Griffin","Dejounte Murray","De'Andre Hunter","Onyeka Okongwu"]},{"team":"DET","players":["Kevin Knox II","Ausar Thompson","Marcus Sasser","Alec Burks","James Wiseman"]}],"player":null,"description":"SUBS: Sasser FOR Cunningham, Ivey FOR Thompson","shot_details":{"team":null,"points":null}},{"quarter":1,"time_remaining":"04:24","score":"ATL 26 - DET 17","players_on_court":[{"team":"ATL","players":["Saddiq Bey","AJ Griffin","Dejounte Murray","De'Andre Hunter","Onyeka Okongwu"]},{"team":"DET","players":["Kevin Knox II","Ausar Thompson","Cade Cunningham","Alec Burks","James Wiseman"]}],"player":"James Wiseman","description":"Wiseman Free Throw 2 of 2","shot_details":{"team":null,"points":null}},{"quarter":1,"time_remaining":"04:02","score":"ATL 29 - DET 17","players_on_court":[{"team":"ATL","players":["Saddiq Bey","AJ Griffin","Dejounte Murray","De'Andre Hunter","Onyeka Okongwu"]},{"team":"DET","players":["Kevin Knox II","Jaden Ivey","Marcus Sasser","Alec Burks","James Wiseman"]}],"player":"Dejounte Murray","description":"Murray 27' 3PT Pullup Jump Shot","shot_details":{"team":"ATL","points":3}},{"quarter":1,"time_remaining":"03:45","score":"ATL 29 - DET 19","players_on_court":[{"team":"ATL","players":["Saddiq Bey","AJ Griffin","Dejounte Murray","De'Andre Hunter","Onyeka Okongwu"]},{"team":"DET","players":["Kevin Knox II","Jaden Ivey","Marcus Sasser","Alec Burks","James Wiseman"]}],"player":"James Wiseman","description":"Wiseman 2' Layup","shot_details":{"team":"DET","points":2}},{"quarter":1,"time_remaining":"03:36","score":"ATL 29 - DET 19","players_on_court":[{"team":"ATL","players":["Saddiq Bey","AJ Griffin","Dejounte Murray","De'Andre Hunter","Onyeka Okongwu"]},{"team":"DET","players":["Kevin Knox II","Jaden Ivey","Marcus Sasser","Alec Burks","James Wiseman"]}],"player":"Alec Burks","description":"Burks P.FOUL on De'Andre Hunter","shot_details":{"team":null,"points":null}}]}"""
        game_context = json.loads(json_data)  # This properly converts null to None
        
        print("📋 Pre-built Context Summary:")
        print(f"   Away Team: {game_context['away_team']['name']} ({len(game_context['away_team']['players'])} players)")
        print(f"   Home Team: {game_context['home_team']['name']} ({len(game_context['home_team']['players'])} players)")
        print(f"   Recent Plays: {len(game_context['recent_plays'])} plays already loaded")
        print(f"   Current Score: {game_context['recent_plays'][-1]['score']}")
        print(f"   Last Play: {game_context['recent_plays'][-1]['description']}")
        print()
        
    else:
        print(f"❌ Invalid TEST_MODE: {TEST_MODE}")
        print("   Valid options: 'full_pipeline' or 'skip_stage1'")
        return
    
    try:
        # Configure rolling sequence parameters
        n_iterations = 750  # Change this to control how many rolling predictions
        
        print(f"\n🚀 Starting rolling prediction sequence (N={n_iterations})")
        
        # Run the rolling sequence based on the selected mode
        skip_stage1 = (TEST_MODE == "skip_stage1")
        results = predict_rolling_sequence(game_context, n_iterations=n_iterations, skip_stage1=skip_stage1)
        
        print("\n" + "=" * 80)
        print("🎯 ROLLING SEQUENCE RESULTS")
        print("=" * 80)
        
        # Show Stage 1 results
        if results.get("stage1_response"):
            print("\n📋 STAGE 1 (Initial next_plays):")
            try:
                stage1_json = json.loads(results["stage1_response"])
                print(f"Generated {len(stage1_json.get('next_plays', []))} initial plays")
                # Show first few plays
                for i, play in enumerate(stage1_json.get('next_plays', [])[:3]):
                    print(f"  {i+1}. {play.get('time_remaining', 'N/A')} - {play.get('description', 'No description')}")
                if len(stage1_json.get('next_plays', [])) > 3:
                    print(f"  ... and {len(stage1_json.get('next_plays', [])) - 3} more plays")
            except json.JSONDecodeError:
                print("❌ Could not parse Stage 1 response")
        
        # Show rolling iterations
        print(f"\n🔄 ROLLING ITERATIONS ({len(results.get('iterations', []))} completed):")
        for iteration_data in results.get("iterations", []):
            iteration_num = iteration_data.get("iteration", "?")
            print(f"\n--- Iteration {iteration_num} ---")
            
            if "error" in iteration_data:
                print(f"❌ Error: {iteration_data['error']}")
            elif "next_play" in iteration_data:
                next_play = iteration_data["next_play"]
                time_remaining = next_play.get("time_remaining", "N/A")
                description = next_play.get("description", "No description")
                score = next_play.get("score", "N/A")
                recent_plays_count = iteration_data.get("recent_plays_count", "?")
                
                print(f"⏰ Time: {time_remaining}")
                print(f"🏀 Play: {description}")
                print(f"📊 Score: {score}")
                print(f"📋 Recent plays window: {recent_plays_count} plays")
        
        print("\n" + "=" * 80)
        print("✅ ROLLING SEQUENCE COMPLETE!")
        print("=" * 80)
        
        # Summary
        total_predictions = len(results.get("iterations", []))
        successful_predictions = len([i for i in results.get("iterations", []) if "next_play" in i])
        print(f"📊 Summary: {successful_predictions}/{total_predictions} successful predictions")
        
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        platform = get_prediction_platform()
        print(f"\n💡 Troubleshooting tips for {platform.upper()}:")
        
        if platform == "openai":
            print("   • Make sure OPENAI_API_KEY environment variable is set")
            print("   • Verify the fine-tuned model ID is correct")
            print("   • Check your OpenAI account has access to the model")
        elif platform == "gemini":
            print("   • Ensure you've authenticated: gcloud auth login")
            print("   • Set correct project: gcloud config set project YOUR_PROJECT_ID")
            print("   • Set GEMINI_MODEL_1_ENDPOINT and GEMINI_MODEL_2_ENDPOINT environment variables")
            print("   • Verify your fine-tuned model endpoints are deployed and accessible")
            print("   • Check that the google-cloud-aiplatform library is installed")
        
        print("   • Check your internet connection")
        print("   • Verify the input data format is correct")

if __name__ == "__main__":
    main()
