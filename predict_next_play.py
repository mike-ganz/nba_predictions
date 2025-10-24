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
import re
import copy
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from game.prediction_client import PredictionClientFactory, BasePredictionClient

# Try to import ujson for faster JSON operations, fallback to standard json
try:
    import ujson
    json_dumps = ujson.dumps
    json_loads = ujson.loads
    print("Using ujson for faster JSON operations")
except ImportError:
    json_dumps = json.dumps
    json_loads = json.loads
    print("Using standard json library (consider installing ujson for better performance)")


_STATIC_STAGE1_CACHE: Dict[str, Any] = {}
_CLIENT_CACHE: Optional[Tuple[BasePredictionClient, Dict[str, str]]] = None
_STAGE1_CACHE: Dict[Tuple[str, str, bool], Dict[str, Any]] = {}

# Event code to human-readable mapping (static - zero overhead)
_EVENT_DESCRIPTIONS = {
    "made2": "made 2PT",
    "made3": "made 3PT", 
    "miss2": "missed 2PT",
    "miss3": "missed 3PT",
    "mft": "made FT",
    "xft": "missed FT",
    "d_reb": "def. rebound",
    "o_reb": "off. rebound",
    "tov": "turnover",
    "s_foul": "shooting foul",
    "p_foul": "personal foul",
    "o_foul": "offensive foul",
    "sub": "substitution",
    "timeout": "timeout",
    "period": "period end",
    "jumpball": "jump ball",
    "viol": "violation",
    "tech": "technical foul",
    "unknown": "unknown"
}


def _format_readable_play(play_tuple: list, away_players: list, home_players: list) -> str:
    """
    Format a compact play tuple into human-readable description.
    
    Args:
        play_tuple: 9-element play tuple [q, t, score, margin, actor, fouls, event, zone, lineup]
        away_players: Away team roster (ap) - list of [name, stats...]
        home_players: Home team roster (hp) - list of [name, stats...]
    
    Returns:
        Human-readable play description (e.g., "Turner missed 2PT")
    """
    if len(play_tuple) < 7:
        return str(play_tuple)  # Fallback for invalid format
    
    # Extract event code
    event_code = play_tuple[6]
    event_desc = _EVENT_DESCRIPTIONS.get(event_code, event_code)
    
    # Extract actor [team, player_index]
    actor = play_tuple[4] if len(play_tuple) > 4 else None
    
    # Get player name if available
    player_name = None
    if isinstance(actor, list) and len(actor) >= 2:
        team_code = actor[0]
        player_idx = actor[1]
        
        if isinstance(player_idx, int) and player_idx >= 0:
            try:
                if team_code == "A" and player_idx < len(away_players):
                    player_name = away_players[player_idx][0]  # First element is name
                elif team_code == "H" and player_idx < len(home_players):
                    player_name = home_players[player_idx][0]  # First element is name
            except (IndexError, TypeError):
                pass
    
    # Format description
    if player_name:
        # Get last name only for brevity (e.g., "De'Andre Hunter" -> "Hunter")
        last_name = player_name.split()[-1] if ' ' in player_name else player_name
        return f"{last_name} {event_desc}"
    else:
        # No player identified - show just event
        return event_desc


def _resolve_static_path(path_str: str) -> str:
    """Resolve Stage 1 static path relative to workspace when needed."""
    path_obj = Path(path_str)
    if path_obj.is_absolute():
        return str(path_obj)

    # Allow referencing files relative to the project root (this file's directory)
    base_dir = Path(__file__).resolve().parent
    candidate = base_dir / path_obj
    if candidate.exists():
        return str(candidate)
    return str(path_obj)


def _load_stage1_static_from_path(path: str) -> Optional[Any]:
    """Load and cache a static Stage 1 response from disk."""
    if not path:
        return None

    normalized_path = _resolve_static_path(path)

    if normalized_path in _STATIC_STAGE1_CACHE:
        return _STATIC_STAGE1_CACHE[normalized_path]

    if not os.path.exists(normalized_path):
        print(f"⚠️ Static Stage 1 response file not found: {normalized_path}")
        return None

    try:
        with open(normalized_path, "r", encoding="utf-8") as f:
            file_contents = f.read()
        try:
            data = json_loads(file_contents)
        except Exception:
            data = json.loads(file_contents)
        _STATIC_STAGE1_CACHE[normalized_path] = data
        print(f"🧪 Loaded static Stage 1 response from {normalized_path}")
        return data
    except Exception as exc:
        print(f"⚠️ Failed to load static Stage 1 response from {normalized_path}: {exc}")
        return None


def _normalize_stage1_payload(payload: Any) -> Any:
    """Normalize payload strings that contain embedded JSON."""
    if isinstance(payload, str):
        for loader in (json_loads, json.loads):
            try:
                payload = loader(payload)
                break
            except Exception:
                continue
    return payload


def _select_stage1_payload(static_data: Any, requested_key: Optional[str]) -> Optional[Any]:
    """Select appropriate Stage 1 payload from cached/static data."""
    payload = None

    if isinstance(static_data, dict):
        if requested_key and requested_key in static_data:
            payload = static_data[requested_key]
        elif any(k in static_data for k in ("y", "next_plays", "messages")):
            payload = static_data

    elif isinstance(static_data, list):
        if requested_key:
            for item in static_data:
                if isinstance(item, dict) and item.get("game_id") == requested_key:
                    payload = item.get("response") or item
                    break
        if payload is None and static_data and isinstance(static_data[0], list):
            payload = static_data

    elif isinstance(static_data, str):
        payload = static_data

    if isinstance(payload, dict) and "messages" in payload:
        # Extract assistant content from chat transcript style payloads
        messages = payload.get("messages", [])
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                payload = message.get("content")
                break

    if payload is not None:
        payload = _normalize_stage1_payload(payload)

    return payload


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
        
        print(f"OptimizedGameContext initialized - base context cached ({len(self._base_json_cached)} chars)")
    
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
        
        # For larger sequences, incorporate both head and tail plays to detect tail-only changes
        # Avoid creating intermediate tuples - use hash accumulation instead
        hash_acc = hash(play_count)
        # Head sample (first up to 5 plays)
        head_n = min(5, play_count)
        for i in range(head_n):
            play = plays[i]
            desc = play.get('description', '')
            time_r = play.get('time_remaining', '')
            hash_acc ^= hash(desc[:30])
            hash_acc ^= hash(time_r)
        # Tail sample (last up to 3 plays)
        tail_n = min(3, play_count)
        for i in range(play_count - tail_n, play_count):
            play = plays[i]
            desc = play.get('description', '')
            time_r = play.get('time_remaining', '')
            hash_acc ^= hash(desc[-30:])  # Different slice to reduce collision
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
        print(f"Restoring recent_plays: {len(self.current_recent_plays)} → {len(restored_plays)} plays")
        if restored_plays:
            last_play = restored_plays[-1]
            print(f"🕒 Rolling back to time: {last_play.get('time_remaining', 'N/A')}")
            print(f"📝 Last play: {last_play.get('description', 'N/A')[:60]}...")
        
        self.update_recent_plays(restored_plays)
        
        # Clear the plays cache since we've changed state
        self._plays_cache.clear()
        print(f"🧹 Cleared plays cache due to rollback")
        
    def inject_quarter_start(self, quarter: int) -> None:
        """
        Inject a clean quarter start context by removing all 00:00 plays 
        and adding a fresh quarter start play.
        
        Args:
            quarter: The quarter to start (2, 3, or 4)
        """
        print(f"Injecting Q{quarter} start context")
        
        # Filter out any plays at 00:00

# ===== Minimal platform helpers and rolling prediction (restored exports) =====

def get_prediction_platform() -> str:
    """Return prediction platform from env, defaulting to 'gemini'."""
    load_dotenv()
    platform = os.getenv("PREDICTION_PLATFORM", "gemini").lower()
    if platform not in ("openai", "gemini", "together"):
        platform = "gemini"
    return platform


def init_prediction_client() -> Tuple[BasePredictionClient, Dict[str, str]]:
    """Create or return a cached prediction client and its model config."""
    global _CLIENT_CACHE
    if _CLIENT_CACHE is not None:
        return _CLIENT_CACHE
    platform = get_prediction_platform()
    validation_mode = os.getenv("VALIDATION_MODE", "fast")
    client = PredictionClientFactory.create_client(platform, validation_mode)
    model_config = client.get_model_config()
    _CLIENT_CACHE = (client, model_config)
    return _CLIENT_CACHE


def predict_rolling_sequence(
    game_context: Dict[str, Any],
    n_iterations: int = 5,
    skip_stage1: bool = False,
    stage1_cache_key: Optional[Tuple[str, str, bool]] = None,
    pre_serialized_context: Optional[str] = None,
) -> Dict[str, Any]:
    """Rolling prediction pipeline with evolving context, matching prior behavior.
    - Stage 1: clean base context only (no plays) to get initial plays
    - Stage 2: rolling loop, updating context each iteration
    """
    client, model_config = init_prediction_client()
    
    # Detect compact vs verbose format by keys
    is_compact = ("A" in game_context and "H" in game_context) and ("away_team" not in game_context)
    
    # Build appropriate context manager
    if is_compact:
        # Minimal compact context manager to support rolling JSON
        class _Compact:
            def __init__(self, ctx: Dict[str, Any]):
                self.base = {k: v for k, v in ctx.items() if k not in ("p",)}
                self.p = list(ctx.get("p", []))
            def get_base_for_stage1(self) -> Dict[str, Any]:
                return dict(self.base)
            def update_plays(self, plays: list) -> None:
                self.p = list(plays)
            def add_play(self, play_tuple: list, max_len: int = 20) -> None:
                arr = self.p + [play_tuple]
                if len(arr) > max_len:
                    arr = arr[-max_len:]
                self.p = arr
            def get_json(self) -> str:
                obj = dict(self.base)
                obj["p"] = self.p
                return json_dumps(obj, separators=(",", ":"))
        ctx_mgr = _Compact(game_context)
    else:
        class _Verbose:
            def __init__(self, ctx: Dict[str, Any]):
                self.base = {k: v for k, v in ctx.items() if k != "recent_plays"}
                self.recent = list(ctx.get("recent_plays", []))
            def get_base_for_stage1(self) -> Dict[str, Any]:
                return dict(self.base)
            def update_recent(self, plays: list) -> None:
                self.recent = list(plays)
            def add_play(self, play_obj: Dict[str, Any], max_len: int = 20) -> None:
                arr = self.recent + [play_obj]
                if len(arr) > max_len:
                    arr = arr[-max_len:]
                self.recent = arr
            def get_json(self) -> str:
                obj = dict(self.base)
                if self.recent:
                    obj["recent_plays"] = self.recent
                return json_dumps(obj, separators=(",", ":"))
        ctx_mgr = _Verbose(game_context)

    results: Dict[str, Any] = {"stage1_response": None, "stage2_responses": [], "iterations": []}

    # Stage 1: clean base context (no plays)
    if not skip_stage1:
        try:
            if stage1_cache_key and stage1_cache_key in _STAGE1_CACHE:
                results["stage1_response"] = _STAGE1_CACHE[stage1_cache_key].get("raw_response")
                cached_plays = _STAGE1_CACHE[stage1_cache_key].get("plays") if is_compact else _STAGE1_CACHE[stage1_cache_key].get("recent_plays")
                if cached_plays is not None:
                    (ctx_mgr.update_plays if is_compact else ctx_mgr.update_recent)(cached_plays)
            else:
                stage1_ctx = ctx_mgr.get_base_for_stage1()
                stage1_json = json_dumps(stage1_ctx, separators=(",", ":"))
                content, _usage, game_ended, _rollback, _term = client.predict_with_validation(
                    context=stage1_json,
                    model_id=model_config["model_1_id"],
                    max_tokens=1500,
                    temperature=float(os.getenv("PREDICTION_TEMPERATURE", "1")),
                    max_retries=3,
                    stage1_mode=True,
                )
                results["stage1_response"] = content
                payload = json_loads(content)
                if is_compact:
                    plays = payload.get("y") if isinstance(payload, dict) else payload
                    if isinstance(plays, list):
                        ctx_mgr.update_plays(plays)
                else:
                    next_plays = payload.get("next_plays") if isinstance(payload, dict) else None
                    if isinstance(next_plays, list):
                        ctx_mgr.update_recent(next_plays)
                if stage1_cache_key:
                    cache = {"raw_response": content}
                    if is_compact:
                        cache["plays"] = ctx_mgr.p
                    else:
                        cache["recent_plays"] = ctx_mgr.recent
                    _STAGE1_CACHE[stage1_cache_key] = cache
                if game_ended:
                    results["termination_reason"] = "Game ended during Stage 1"
                    return results
        except Exception:
            results["stage1_response"] = "SKIPPED"

    # Stage 2: rolling
    # Cache temperature and verbosity once per call to avoid repeated env reads and reduce I/O
    _temp = float(os.getenv("PREDICTION_TEMPERATURE", "1"))
    _verbose_iter_logs = os.getenv("PREDICTION_VERBOSE_ITER_LOGS", "0") == "1"
    try:
        _iter_log_every = int(os.getenv("PREDICTION_ITER_LOG_EVERY", "10"))
    except Exception:
        _iter_log_every = 0

    for i in range(max(0, int(n_iterations))):
        iteration_json = ctx_mgr.get_json()
        try:
            resp_text, _usage, game_ended, needs_rollback, _term = client.predict_with_validation(
                context=iteration_json,
                model_id=model_config["model_2_id"],
                max_tokens=5000,
                temperature=_temp,
                max_retries=3,
                stage1_mode=False,
            )
            results["stage2_responses"].append(resp_text)
            parsed = None
            try:
                parsed = json_loads(resp_text)
            except Exception:
                pass
            # Defer context mutation until after handling special flags
            pending_play = None
            if is_compact and isinstance(parsed, dict) and "y" in parsed and isinstance(parsed["y"], list):
                play_tuple = parsed["y"][0] if parsed["y"] and isinstance(parsed["y"][0], list) else parsed["y"]
                if isinstance(play_tuple, list):
                    pending_play = ("compact", play_tuple)
                results["iterations"].append({"iteration": i + 1, "raw_response": resp_text})
            elif not is_compact and isinstance(parsed, dict) and "next_play" in parsed:
                pending_play = ("verbose", parsed["next_play"])
                results["iterations"].append({"iteration": i + 1, "next_play": parsed["next_play"]})
            else:
                results["iterations"].append({"iteration": i + 1, "raw_response": resp_text})

            # Human-readable iteration log: last 5 plays + game state (optional/periodic)
            _should_log_iter = _verbose_iter_logs or (_iter_log_every > 0 and ((i + 1) % _iter_log_every == 0))
            if _should_log_iter:
                try:
                    lines = []
                    if is_compact:
                        plays = ctx_mgr.p
                        if plays:
                            # Get rosters for player name lookup (minimal overhead - just pointers)
                            away_players = ctx_mgr.base.get("ap", [])
                            home_players = ctx_mgr.base.get("hp", [])
                            
                            latest = plays[-1]
                            q = latest[0] if len(latest) > 0 else "?"
                            t_sec = latest[1] if len(latest) > 1 else 0
                            m, s = (t_sec // 60, t_sec % 60)
                            score = latest[2] if len(latest) > 2 and isinstance(latest[2], list) and len(latest[2]) >= 2 else [0, 0]
                            away_abbr = ctx_mgr.base.get("A", "AWAY")
                            home_abbr = ctx_mgr.base.get("H", "HOME")
                            lines.append(f"Game State: Q{q} {m:02d}:{s:02d} | {away_abbr} {score[0]} - {home_abbr} {score[1]}")
                            recent = plays[-5:]
                            for idx, tup in enumerate(recent, 1):
                                tq = tup[0] if len(tup) > 0 else "?"
                                tt = tup[1] if len(tup) > 1 else 0
                                tm, ts = (tt // 60, tt % 60)
                                # Format readable description with player name
                                readable_desc = _format_readable_play(tup, away_players, home_players)
                                lines.append(f"  {idx}. Q{tq} {tm:02d}:{ts:02d} {readable_desc}")
                    else:
                        plays = ctx_mgr.recent
                        if plays:
                            latest = plays[-1]
                            q = latest.get("quarter", "?")
                            t = latest.get("time_remaining", "??:??")
                            score = latest.get("score", "N/A")
                            lines.append(f"Game State: Q{q} {t} | {score}")
                            recent = plays[-5:]
                            for idx, p in enumerate(recent, 1):
                                t2 = p.get("time_remaining", "??:??")
                                desc = p.get("description", "")
                                lines.append(f"  {idx}. [{t2}] {desc}")
                    if lines:
                        print("\n" + "\n".join(lines))
                except Exception:
                    pass

            if game_ended:
                results["termination_reason"] = f"Game ended at iteration {i + 1}"
                break
            if needs_rollback:
                # Handle quarter transition (do not apply pending_play)
                try:
                    # Read and clear target from validator if available
                    _val = getattr(client, "validator", None)
                    target_q = None
                    if _val is not None and hasattr(_val, "get_quarter_transition_target"):
                        target_q = _val.get_quarter_transition_target()
                    else:
                        target_q = getattr(_val, "quarter_transition_target", None)

                    # Capture last known score and margin BEFORE trimming
                    prev_score_compact = None
                    prev_margin_compact = 0
                    prev_score_verbose = None
                    if is_compact:
                        if getattr(ctx_mgr, "p", None):
                            for _pl in reversed(ctx_mgr.p):
                                if isinstance(_pl, list) and len(_pl) >= 3 and isinstance(_pl[2], list) and len(_pl[2]) >= 2:
                                    prev_score_compact = [_pl[2][0], _pl[2][1]]
                                    # If margin present (index 3), carry it forward
                                    if len(_pl) >= 4 and isinstance(_pl[3], int):
                                        prev_margin_compact = _pl[3]
                                    break
                    else:
                        if getattr(ctx_mgr, "recent", None):
                            for _pl in reversed(ctx_mgr.recent):
                                if isinstance(_pl, dict) and _pl.get("score"):
                                    prev_score_verbose = _pl.get("score")
                                    break

                    # Trim trailing 00:00 plays to move past end-of-period noise
                    if is_compact:
                        # Remove tail plays at t=0 or near-zero (<=2s) in same quarter that are administrative
                        while getattr(ctx_mgr, "p", None) and isinstance(ctx_mgr.p[-1], list):
                            last = ctx_mgr.p[-1]
                            t_ok = len(last) > 1 and isinstance(last[1], int) and last[1] <= 2
                            is_admin = False
                            if len(last) >= 7:
                                ev = str(last[6]).lower()
                                is_admin = ("period" in ev) or (ev in ("timeout", "jumpball", "unknown")) or ev.startswith("sub")
                            if t_ok and is_admin:
                                ctx_mgr.p = ctx_mgr.p[:-1]
                            else:
                                break
                        # Inject quarter header play if validator set target
                        if target_q is not None:
                            score_to_use = prev_score_compact if isinstance(prev_score_compact, list) else [0, 0]
                            # Compute margin from score if possible
                            try:
                                _m = prev_margin_compact
                                if isinstance(score_to_use, list) and len(score_to_use) >= 2:
                                    _m = int(score_to_use[0]) - int(score_to_use[1])
                            except Exception:
                                _m = prev_margin_compact
                            # Use administrative 'period' event at 12:00 for quarter start (no player attribution)
                            ctx_mgr.add_play([int(target_q), 720, score_to_use, _m, ["A", -1], 0, "period", None, 0])
                    else:
                        # Remove tail plays at 00:00 or near-zero (<=2s) in same quarter that are administrative
                        while getattr(ctx_mgr, "recent", None) and isinstance(ctx_mgr.recent[-1], dict):
                            last = ctx_mgr.recent[-1]
                            # parse time
                            tm = str(last.get("time_remaining", ""))
                            try:
                                parts = tm.split(":")
                                mm = int(parts[0]) if len(parts) > 0 else 0
                                ss = int(parts[1]) if len(parts) > 1 else 0
                                total = max(0, mm * 60 + ss)
                            except Exception:
                                total = 0
                            desc_l = str(last.get("description", "")).lower()
                            ev_l = str(last.get("event_code", "")).lower()
                            is_admin = ("period" in desc_l) or (ev_l == "period") or ("sub" in desc_l) or (ev_l.startswith("sub")) or ("timeout" in desc_l) or (ev_l == "timeout") or ("jump" in desc_l) or (ev_l.startswith("jump")) or ("unknown" in desc_l) or (ev_l == "unknown")
                            if total <= 2 and is_admin:
                                ctx_mgr.recent = ctx_mgr.recent[:-1]
                            else:
                                break
                        # Inject a clean quarter start marker
                        if target_q is not None:
                            score_to_use = prev_score_verbose if isinstance(prev_score_verbose, str) else (last.get("score") if 'last' in locals() else "")
                            ctx_mgr.add_play({
                                "quarter": int(target_q),
                                "time_remaining": "12:00",
                                "description": "quarter start",
                                "score": score_to_use
                            })
                except Exception:
                    pass
                # Retry next iteration with cleaned context
                continue
            # Apply pending play only if no special handling was required
            if pending_play is not None:
                kind, value = pending_play
                # If validator requested a time adjustment (fast_plus mode), apply it to this pending play
                try:
                    _val = getattr(client, "validator", None)
                    adj = getattr(_val, "pending_time_adjustment", None)
                    if isinstance(adj, dict) and isinstance(adj.get("subtract_seconds"), int):
                        sub_s = max(0, int(adj["subtract_seconds"]))
                        if kind == "compact" and isinstance(value, list) and len(value) >= 2:
                            # value = [q, t_sec, score, margin, actor, fouls, event, zone, lineup]
                            q = int(value[0]) if isinstance(value[0], int) else value[0]
                            t = int(value[1]) if isinstance(value[1], int) else 0
                            new_t = max(0, t - sub_s)
                            if t > 0 and new_t == 0:
                                # Crossing to 00:00: inject a period end marker instead
                                # Keep score and margin as-is to maintain continuity
                                score = value[2] if len(value) >= 3 else [0, 0]
                                margin = value[3] if len(value) >= 4 else 0
                                ctx_mgr.add_play([q, 0, score, margin, ["A", -1], 0, "period", None, 0])
                            else:
                                value[1] = new_t
                                ctx_mgr.add_play(value)
                        elif kind == "verbose" and isinstance(value, dict):
                            tm = str(value.get("time_remaining", ""))
                            # Parse MM:SS
                            try:
                                parts = tm.split(":")
                                mm = int(parts[0]) if len(parts) > 0 else 0
                                ss = int(parts[1]) if len(parts) > 1 else 0
                                total = max(0, mm * 60 + ss)
                            except Exception:
                                total = 0
                            new_total = max(0, total - sub_s)
                            if total > 0 and new_total == 0:
                                # Crossing to 00:00: inject "period end" play instead of applying the pending play
                                q = int(value.get("quarter", 1))
                                score = value.get("score", "")
                                ctx_mgr.add_play({
                                    "quarter": q,
                                    "time_remaining": "00:00",
                                    "description": "period end",
                                    "score": score
                                })
                            else:
                                mm2, ss2 = (new_total // 60, new_total % 60)
                                value["time_remaining"] = f"{mm2:02d}:{ss2:02d}"
                                ctx_mgr.add_play(value)
                        # Clear the adjustment after applying once
                        setattr(_val, "pending_time_adjustment", None)
                    else:
                        # No adjustment requested; apply normally
                        if kind == "compact":
                            ctx_mgr.add_play(value)
                        else:
                            ctx_mgr.add_play(value)
                except Exception:
                    # On any error, fall back to normal add
                    if kind == "compact":
                        ctx_mgr.add_play(value)
                    else:
                        ctx_mgr.add_play(value)
        except Exception as _e:
            results["iterations"].append({"iteration": i + 1, "error": str(_e)})
            continue
    
    return results