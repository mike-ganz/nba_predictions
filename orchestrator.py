#!/usr/bin/env python3
"""
NBA Prediction Orchestration System

Multi-game, multi-run simulation orchestrator for NBA play prediction models.
Provides systematic execution, results storage, and analysis capabilities.

Features:
- Run N simulations for specified games
- Store results in SQLite database
- Configure seasons, games, and simulation parameters
- Progress tracking and error recovery
- Results analysis and reporting

Usage:
    python orchestrator.py --games "22200001,22200002" --runs-per-game 10
    python orchestrator.py --config orchestrator_config.json
"""

import sqlite3
import json
import time
import threading
import logging
import argparse
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager

# Import existing prediction system
from predict_next_play import predict_rolling_sequence, init_prediction_client, get_prediction_platform
from config.settings import config, set_season_year
from game_context_builder import GameContextBuilder


@dataclass
class SimulationConfig:
    """Configuration for orchestration runs."""
    season_year: str
    games: List[str]  # Game IDs to simulate
    runs_per_game: int
    max_iterations_per_run: int = 2000
    skip_stage1: bool = False  # Run Stage 1 by default (first_n_plays mode)
    platform: str = "gemini"  # openai or gemini
    output_db: str = "simulation_results.db"
    log_level: str = "INFO"
    resume_on_error: bool = True
    timeout_minutes: int = 30  # Max time per simulation
    max_threads: int = 1  # Number of threads for parallel execution


@dataclass
class SimulationResult:
    """Results from a single simulation run."""
    run_id: str
    game_id: str
    season_year: str
    platform: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    status: str  # "completed", "timeout", "error", "game_ended"
    total_predictions: int
    successful_predictions: int
    final_score: Optional[str]
    final_quarter: Optional[int]
    final_time: Optional[str]
    termination_reason: Optional[str]
    scoring_plays: int
    non_scoring_plays: int
    scoring_rate: float
    error_message: Optional[str]
    iterations_data: str  # JSON string of iteration details
    
    # Validation termination information (added for detailed termination tracking)
    validation_termination_type: Optional[str] = None  # "game_ended", "rollback_time", etc.
    validation_termination_reason: Optional[str] = None  # Human-readable reason
    validation_trigger_condition: Optional[str] = None  # Specific trigger condition
    validation_termination_timestamp: Optional[str] = None  # When termination was decided
    validation_consecutive_count: Optional[int] = None  # Consecutive condition count
    validation_total_attempts: Optional[int] = None  # Total validation attempts
    validation_game_state_quarter: Optional[int] = None  # Game quarter at termination
    validation_game_state_time: Optional[str] = None  # Game time at termination
    validation_game_state_score: Optional[str] = None  # Game score at termination
    validation_context_json: Optional[str] = None  # JSON string of validation context
    
    # Validation failure information (for retries that exhausted attempts)
    validation_failure_timestamp: Optional[str] = None  # When validation started failing
    validation_total_failed_attempts: Optional[int] = None  # Total failed validation attempts
    validation_most_common_reason: Optional[str] = None  # Most frequent failure reason
    validation_most_common_reason_count: Optional[int] = None  # Count of most common reason
    validation_most_common_error_type: Optional[str] = None  # Most frequent error type
    validation_most_common_error_type_count: Optional[int] = None  # Count of most common error type
    validation_most_common_field: Optional[str] = None  # Most problematic field
    validation_most_common_field_count: Optional[int] = None  # Count of field errors
    validation_unique_reasons: Optional[int] = None  # Number of distinct failure reasons
    validation_unique_error_types: Optional[int] = None  # Number of distinct error types
    validation_unique_fields: Optional[int] = None  # Number of distinct problematic fields
    validation_failure_summary: Optional[str] = None  # JSON summary of all failure patterns
    validation_response_examples: Optional[str] = None  # JSON array of failed response examples


class GameContextLoader:
    """Loads game contexts for different games and scenarios."""
    
    def __init__(self):
        self.game_contexts = {}  # Cache loaded contexts
        
    def load_game_context(self, game_id: str, season_year: str) -> Dict[str, Any]:
        """
        Load game context for a specific game.
        
        For now, this uses the pre-built contexts from the original script.
        In the future, this could be expanded to load from historical data.
        """
        cache_key = f"{game_id}_{season_year}"
        
        if cache_key in self.game_contexts:
            return self.game_contexts[cache_key].copy()
        
        # Load game context - for now we'll use sample contexts
        # This is where you'd integrate with your historical data loading
        context = self._get_sample_context_for_game(game_id, season_year)
        self.game_contexts[cache_key] = context
        
        return context.copy()
    
    def _get_sample_context_for_game(self, game_id: str, season_year: str) -> Dict[str, Any]:
        """
        Get sample context for a game. 
        TODO: Replace with actual game data loading from your data files.
        """
        
        # Sample contexts - in practice you'd load these from your historical data
        sample_contexts = {
            "22200001": {
                "away_team": {"name": "ATL", "stats": {"OEFF": 115.6, "DEFF": 117.8, "PACE": 100.7, "REST_DAYS": 3}, "players": [{"name": "De'Andre Hunter", "profile": {"offense": 0.54, "defense": 1.77, "shot_selection": 0.21, "efficiency": -0.23, "MPG": 31, "usage": 18}}, {"name": "Jalen Johnson", "profile": {"offense": 1.15, "defense": 1.58, "shot_selection": 0.4, "efficiency": -1.47, "MPG": 30, "usage": 18}}]},
                "home_team": {"name": "DET", "stats": {"OEFF": 109.0, "DEFF": 115.9, "PACE": 97.7, "REST_DAYS": 2}, "players": [{"name": "Ausar Thompson", "profile": {"offense": 1.04, "defense": 3.99, "shot_selection": 0.84, "efficiency": 0.53, "MPG": 32, "usage": 19}}, {"name": "Isaiah Stewart", "profile": {"offense": 0.62, "defense": 0.79, "shot_selection": -0.12, "efficiency": -0.63, "MPG": 33, "usage": 16}}]},
                "recent_plays": [
                    {"quarter": 1, "time_remaining": "06:15", "score": "ATL 18 - DET 11", "players_on_court": [{"team": "ATL", "players": ["De'Andre Hunter", "Jalen Johnson"]}, {"team": "DET", "players": ["Ausar Thompson", "Isaiah Stewart"]}], "player": "De'Andre Hunter", "description": "Hunter 18' Jump Shot", "shot_details": {"team": "ATL", "points": 2}}
                ]
            },
            "22200002": {
                "away_team": {"name": "ORL", "stats": {"OEFF": 113.7, "DEFF": 112.7, "PACE": 96.5, "REST_DAYS": 2}, "players": [{"name": "Franz Wagner", "profile": {"offense": 2.34, "defense": 1.42, "shot_selection": 0.69, "efficiency": -0.67, "MPG": 33, "usage": 26}}]},
                "home_team": {"name": "CLE", "stats": {"OEFF": 114.9, "DEFF": 112.5, "PACE": 97.2, "REST_DAYS": 2}, "players": [{"name": "Donovan Mitchell", "profile": {"offense": 3.79, "defense": 2.55, "shot_selection": 0.2, "efficiency": -1.04, "MPG": 35, "usage": 31}}]},
                "recent_plays": [
                    {"quarter": 1, "time_remaining": "08:30", "score": "ORL 12 - CLE 15", "players_on_court": [{"team": "ORL", "players": ["Franz Wagner"]}, {"team": "CLE", "players": ["Donovan Mitchell"]}], "player": "Franz Wagner", "description": "Wagner 20' Pullup Jump Shot", "shot_details": {"team": "ORL", "points": 2}}
                ]
            }
        }
        
        if game_id in sample_contexts:
            return sample_contexts[game_id]
        else:
            # Default context - you'd want to load actual game data here
            logging.warning(f"No specific context for game {game_id}, using default")
            return sample_contexts["22200001"]  # Use first sample as default


class SimulationDatabase:
    """Manages SQLite database for simulation results."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._db_lock = threading.Lock()  # Thread-safe database access
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            # Enable WAL mode for better concurrent access
            conn.execute('PRAGMA journal_mode=WAL;')
            conn.execute('PRAGMA synchronous=NORMAL;')  # Balance safety and performance
            conn.execute('PRAGMA cache_size=10000;')    # Increase cache for better performance
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS simulation_runs (
                    run_id TEXT PRIMARY KEY,
                    game_id TEXT NOT NULL,
                    season_year TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    duration_seconds REAL,
                    status TEXT NOT NULL,
                    total_predictions INTEGER,
                    successful_predictions INTEGER,
                    final_score TEXT,
                    final_quarter INTEGER,
                    final_time TEXT,
                    termination_reason TEXT,
                    scoring_plays INTEGER,
                    non_scoring_plays INTEGER,
                    scoring_rate REAL,
                    error_message TEXT,
                    iterations_data TEXT,
                    validation_termination_type TEXT,
                    validation_termination_reason TEXT,
                    validation_trigger_condition TEXT,
                    validation_termination_timestamp TEXT,
                    validation_consecutive_count INTEGER,
                    validation_total_attempts INTEGER,
                    validation_game_state_quarter INTEGER,
                    validation_game_state_time TEXT,
                    validation_game_state_score TEXT,
                    validation_context_json TEXT,
                    validation_failure_timestamp TEXT,
                    validation_total_failed_attempts INTEGER,
                    validation_most_common_reason TEXT,
                    validation_most_common_reason_count INTEGER,
                    validation_most_common_error_type TEXT,
                    validation_most_common_error_type_count INTEGER,
                    validation_most_common_field TEXT,
                    validation_most_common_field_count INTEGER,
                    validation_unique_reasons INTEGER,
                    validation_unique_error_types INTEGER,
                    validation_unique_fields INTEGER,
                    validation_failure_summary TEXT,
                    validation_response_examples TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Index for efficient queries
            conn.execute('CREATE INDEX IF NOT EXISTS idx_game_season ON simulation_runs(game_id, season_year)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_status ON simulation_runs(status)')
            conn.commit()
    
    def save_result(self, result: SimulationResult):
        """Save a simulation result to database with thread safety."""
        with self._db_lock:  # Ensure thread-safe database access
            try:
                with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO simulation_runs (
                            run_id, game_id, season_year, platform, start_time, end_time,
                            duration_seconds, status, total_predictions, successful_predictions,
                            final_score, final_quarter, final_time, termination_reason,
                            scoring_plays, non_scoring_plays, scoring_rate, error_message, iterations_data,
                            validation_termination_type, validation_termination_reason, validation_trigger_condition,
                            validation_termination_timestamp, validation_consecutive_count, validation_total_attempts,
                            validation_game_state_quarter, validation_game_state_time, validation_game_state_score,
                            validation_context_json, validation_failure_timestamp, validation_total_failed_attempts,
                            validation_most_common_reason, validation_most_common_reason_count, validation_most_common_error_type,
                            validation_most_common_error_type_count, validation_most_common_field, validation_most_common_field_count,
                            validation_unique_reasons, validation_unique_error_types, validation_unique_fields,
                            validation_failure_summary, validation_response_examples
                        ) VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                    ''', (
                        result.run_id, result.game_id, result.season_year, result.platform,
                        result.start_time.isoformat(), result.end_time.isoformat() if result.end_time else None,
                        result.duration_seconds, result.status, result.total_predictions,
                        result.successful_predictions, result.final_score, result.final_quarter,
                        result.final_time, result.termination_reason, result.scoring_plays,
                        result.non_scoring_plays, result.scoring_rate, result.error_message,
                        result.iterations_data,
                        # Validation termination fields
                        result.validation_termination_type, result.validation_termination_reason, result.validation_trigger_condition,
                        result.validation_termination_timestamp, result.validation_consecutive_count, result.validation_total_attempts,
                        result.validation_game_state_quarter, result.validation_game_state_time, result.validation_game_state_score,
                        result.validation_context_json,
                        # Validation failure fields
                        result.validation_failure_timestamp, result.validation_total_failed_attempts, 
                        result.validation_most_common_reason, result.validation_most_common_reason_count,
                        result.validation_most_common_error_type, result.validation_most_common_error_type_count,
                        result.validation_most_common_field, result.validation_most_common_field_count,
                        result.validation_unique_reasons, result.validation_unique_error_types, result.validation_unique_fields,
                        result.validation_failure_summary, result.validation_response_examples
                    ))
                    conn.commit()
                    
                    # Log successful save with thread info
                    thread_name = threading.current_thread().name
                    logging.debug(f"Successfully saved result {result.run_id} to database (thread: {thread_name})")
                    
            except sqlite3.Error as e:
                logging.error(f"SQLite error saving result {result.run_id}: {e}")
                raise
            except Exception as e:
                logging.error(f"Unexpected error saving result {result.run_id}: {e}")
                raise
    
    def get_results_for_game(self, game_id: str, season_year: str) -> List[Dict]:
        """Get all results for a specific game."""
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT * FROM simulation_runs 
                WHERE game_id = ? AND season_year = ?
                ORDER BY start_time
            ''', (game_id, season_year))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all simulations."""
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            cursor = conn.execute('''
                SELECT 
                    COUNT(*) as total_simulations,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
                    COUNT(CASE WHEN status = 'error' THEN 1 END) as errors,
                    COUNT(CASE WHEN status = 'timeout' THEN 1 END) as timeouts,
                    AVG(duration_seconds) as avg_duration,
                    AVG(scoring_rate) as avg_scoring_rate,
                    COUNT(DISTINCT game_id) as unique_games
                FROM simulation_runs
            ''')
            return dict(cursor.fetchone())


class NBA_Orchestrator:
    """Main orchestration class for running multiple NBA simulations."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.db = SimulationDatabase(config.output_db)
        self.context_builder = GameContextBuilder(config.season_year)  # Use real data loader
        self.logger = self._setup_logging()
        
        # Set up environment for predictions
        set_season_year(config.season_year)
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for orchestration."""
        logger = logging.getLogger('NBA_Orchestrator')
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler('orchestrator.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def run_all_simulations(self) -> Dict[str, Any]:
        """Run all configured simulations."""
        total_runs = len(self.config.games) * self.config.runs_per_game
        completed_runs = 0
        failed_runs = 0
        
        self.logger.info(f"Starting orchestration: {total_runs} total simulations")
        self.logger.info(f"   Games: {self.config.games}")
        self.logger.info(f"   Runs per game: {self.config.runs_per_game}")
        self.logger.info(f"   Season: {self.config.season_year}")
        self.logger.info(f"   Platform: {self.config.platform}")
        
        start_time = time.time()
        
        for game_id in self.config.games:
            self.logger.info(f"\nStarting simulations for game {game_id}")
            
            for run_num in range(self.config.runs_per_game):
                run_id = f"{game_id}_{self.config.season_year}_{run_num + 1:04d}_{int(time.time())}"
                
                self.logger.info(f"   Run {run_num + 1}/{self.config.runs_per_game}: {run_id}")
                
                try:
                    result = self._run_single_simulation(run_id, game_id)
                    self.db.save_result(result)
                    
                    if result.status == "completed":
                        completed_runs += 1
                        self.logger.info(f"   Completed: {result.successful_predictions}/{result.total_predictions} predictions, {result.scoring_rate:.1f}% scoring")
                    else:
                        failed_runs += 1
                        self.logger.warning(f"   Status: {result.status} - {result.termination_reason}")
                
                except Exception as e:
                    failed_runs += 1
                    self.logger.error(f"   Failed: {str(e)}")
                    
                    # Still save the failed result
                    error_result = SimulationResult(
                        run_id=run_id, game_id=game_id, season_year=self.config.season_year,
                        platform=self.config.platform, start_time=datetime.now(),
                        end_time=datetime.now(), duration_seconds=0.0, status="error",
                        total_predictions=0, successful_predictions=0, final_score=None,
                        final_quarter=None, final_time=None, termination_reason=None,
                        scoring_plays=0, non_scoring_plays=0, scoring_rate=0.0,
                        error_message=str(e), iterations_data="[]"
                    )
                    self.db.save_result(error_result)
        
        total_time = time.time() - start_time
        
        summary = {
            "total_simulations": total_runs,
            "completed": completed_runs,
            "failed": failed_runs,
            "success_rate": (completed_runs / total_runs) * 100 if total_runs > 0 else 0,
            "total_duration_minutes": total_time / 60,
            "avg_time_per_simulation": total_time / total_runs if total_runs > 0 else 0
        }
        
        self.logger.info(f"\nOrchestration Complete!")
        self.logger.info(f"   Completed: {completed_runs}/{total_runs} ({summary['success_rate']:.1f}%)")
        self.logger.info(f"   Total time: {summary['total_duration_minutes']:.1f} minutes")
        self.logger.info(f"   Results saved to: {self.config.output_db}")
        
        return summary
    
    def _run_single_simulation(self, run_id: str, game_id: str) -> SimulationResult:
        """Run a single simulation and return results."""
        start_time = datetime.now()
        
        try:
            # Load game context using real data
            game_context = self.context_builder.build_game_context(
                game_id=game_id, 
                for_first_n_plays=(not self.config.skip_stage1)  # Clean context for Stage 1
            )
            
            # Run prediction
            results = predict_rolling_sequence(
                game_context=game_context,
                n_iterations=self.config.max_iterations_per_run,
                skip_stage1=self.config.skip_stage1
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Extract results
            iterations = results.get("iterations", [])
            total_predictions = len(iterations)
            successful_predictions = len([i for i in iterations if "next_play" in i])
            
            # Calculate scoring statistics
            scoring_plays = 0
            non_scoring_plays = 0
            
            for iteration_result in iterations:
                if "next_play" in iteration_result:
                    next_play = iteration_result["next_play"]
                    if "shot_details" in next_play and next_play["shot_details"]:
                        points = next_play["shot_details"].get("points")
                        if points is not None and points > 0:
                            scoring_plays += 1
                        else:
                            non_scoring_plays += 1
                    else:
                        non_scoring_plays += 1
            
            scoring_rate = (scoring_plays / (scoring_plays + non_scoring_plays)) * 100 if (scoring_plays + non_scoring_plays) > 0 else 0
            
            # Get final game state
            final_score = None
            final_quarter = None
            final_time = None
            
            if iterations and "next_play" in iterations[-1]:
                final_play = iterations[-1]["next_play"]
                final_score = final_play.get("score")
                final_quarter = final_play.get("quarter")
                final_time = final_play.get("time_remaining")
            
            # Determine status and termination reason
            status = "completed"
            termination_reason = results.get("termination_reason")
            
            if termination_reason:
                if "Game ended" in termination_reason:
                    status = "game_ended"
            
            # Extract validation termination information
            validation_termination = results.get("validation_termination", {})
            
            # Extract validation failure information (for retries that exhausted attempts)
            validation_failure = results.get("validation_failure", {})
            
            # Log validation termination if present
            if validation_termination:
                self.logger.info(f"🛑 Validation termination captured for {run_id}")
                self.logger.info(f"   Type: {validation_termination.get('validation_termination_type', 'N/A')}")
                self.logger.info(f"   Trigger: {validation_termination.get('validation_trigger_condition', 'N/A')}")
                self.logger.info(f"   Game State: Q{validation_termination.get('validation_game_state_quarter', '?')} {validation_termination.get('validation_game_state_time', 'N/A')}")
            
            # Log validation failure details if present
            if validation_failure:
                self.logger.info(f"🔍 Validation failure details captured for {run_id}")
                self.logger.info(f"   Most common reason: {validation_failure.get('validation_most_common_reason', 'N/A')}")
                self.logger.info(f"   Most common error type: {validation_failure.get('validation_most_common_error_type', 'N/A')}")
                self.logger.info(f"   Total failed attempts: {validation_failure.get('validation_total_failed_attempts', 0)}")
                self.logger.info(f"   Unique failure reasons: {validation_failure.get('validation_unique_reasons', 0)}")
            
            return SimulationResult(
                run_id=run_id,
                game_id=game_id,
                season_year=self.config.season_year,
                platform=self.config.platform,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                status=status,
                total_predictions=total_predictions,
                successful_predictions=successful_predictions,
                final_score=final_score,
                final_quarter=final_quarter,
                final_time=final_time,
                termination_reason=termination_reason,
                scoring_plays=scoring_plays,
                non_scoring_plays=non_scoring_plays,
                scoring_rate=scoring_rate,
                error_message=None,
                iterations_data=json.dumps(iterations[:10]),  # Store first 10 iterations
                # Validation termination information
                validation_termination_type=validation_termination.get('validation_termination_type'),
                validation_termination_reason=validation_termination.get('validation_termination_reason'),
                validation_trigger_condition=validation_termination.get('validation_trigger_condition'),
                validation_termination_timestamp=validation_termination.get('validation_termination_timestamp'),
                validation_consecutive_count=validation_termination.get('validation_consecutive_count'),
                validation_total_attempts=validation_termination.get('validation_total_attempts'),
                validation_game_state_quarter=validation_termination.get('validation_game_state_quarter'),
                validation_game_state_time=validation_termination.get('validation_game_state_time'),
                validation_game_state_score=validation_termination.get('validation_game_state_score'),
                validation_context_json=validation_termination.get('validation_context_json'),
                # Validation failure information
                validation_failure_timestamp=validation_failure.get('validation_failure_timestamp'),
                validation_total_failed_attempts=validation_failure.get('validation_total_failed_attempts'),
                validation_most_common_reason=validation_failure.get('validation_most_common_reason'),
                validation_most_common_reason_count=validation_failure.get('validation_most_common_reason_count'),
                validation_most_common_error_type=validation_failure.get('validation_most_common_error_type'),
                validation_most_common_error_type_count=validation_failure.get('validation_most_common_error_type_count'),
                validation_most_common_field=validation_failure.get('validation_most_common_field'),
                validation_most_common_field_count=validation_failure.get('validation_most_common_field_count'),
                validation_unique_reasons=validation_failure.get('validation_unique_reasons'),
                validation_unique_error_types=validation_failure.get('validation_unique_error_types'),
                validation_unique_fields=validation_failure.get('validation_unique_fields'),
                validation_failure_summary=validation_failure.get('validation_failure_summary'),
                validation_response_examples=validation_failure.get('validation_response_examples')
            )
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Check if validation failure info was captured
            validation_failure = {}
            try:
                import builtins
                if hasattr(builtins, '_last_validation_failure') and builtins._last_validation_failure:
                    validation_failure = builtins._last_validation_failure
                    self.logger.info(f"🔍 Using captured validation failure details for error case")
                    self.logger.info(f"   Most common reason: {validation_failure.get('validation_most_common_reason', 'N/A')}")
                    # Clear the global after use
                    builtins._last_validation_failure = None
            except:
                pass  # Ignore any issues accessing global validation failure info
            
            return SimulationResult(
                run_id=run_id,
                game_id=game_id,
                season_year=self.config.season_year,
                platform=self.config.platform,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                status="error",
                total_predictions=0,
                successful_predictions=0,
                final_score=None,
                final_quarter=None,
                final_time=None,
                termination_reason=None,
                scoring_plays=0,
                non_scoring_plays=0,
                scoring_rate=0.0,
                error_message=str(e),
                iterations_data="[]",
                # Validation termination information (None for error cases)
                validation_termination_type=None,
                validation_termination_reason=None,
                validation_trigger_condition=None,
                validation_termination_timestamp=None,
                validation_consecutive_count=None,
                validation_total_attempts=None,
                validation_game_state_quarter=None,
                validation_game_state_time=None,
                validation_game_state_score=None,
                validation_context_json=None,
                # Validation failure information (use captured data if available)
                validation_failure_timestamp=validation_failure.get('validation_failure_timestamp'),
                validation_total_failed_attempts=validation_failure.get('validation_total_failed_attempts'),
                validation_most_common_reason=validation_failure.get('validation_most_common_reason'),
                validation_most_common_reason_count=validation_failure.get('validation_most_common_reason_count'),
                validation_most_common_error_type=validation_failure.get('validation_most_common_error_type'),
                validation_most_common_error_type_count=validation_failure.get('validation_most_common_error_type_count'),
                validation_most_common_field=validation_failure.get('validation_most_common_field'),
                validation_most_common_field_count=validation_failure.get('validation_most_common_field_count'),
                validation_unique_reasons=validation_failure.get('validation_unique_reasons'),
                validation_unique_error_types=validation_failure.get('validation_unique_error_types'),
                validation_unique_fields=validation_failure.get('validation_unique_fields'),
                validation_failure_summary=validation_failure.get('validation_failure_summary'),
                validation_response_examples=validation_failure.get('validation_response_examples')
            )
    
    def analyze_results(self, game_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze results for specific game or all games."""
        if game_id:
            results = self.db.get_results_for_game(game_id, self.config.season_year)
            title = f"Game {game_id}"
        else:
            # Get all results
            with sqlite3.connect(self.db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('SELECT * FROM simulation_runs ORDER BY start_time')
                results = [dict(row) for row in cursor.fetchall()]
            title = "All Games"
        
        if not results:
            return {"error": "No results found"}
        
        # Calculate statistics
        completed = [r for r in results if r['status'] == 'completed']
        
        analysis = {
            "title": title,
            "total_simulations": len(results),
            "completed_simulations": len(completed),
            "success_rate": (len(completed) / len(results)) * 100 if results else 0,
            "avg_duration_minutes": sum(r['duration_seconds'] for r in results) / len(results) / 60 if results else 0,
        }
        
        if completed:
            analysis.update({
                "avg_predictions_per_run": sum(r['total_predictions'] for r in completed) / len(completed),
                "avg_success_rate": sum(r['successful_predictions'] / max(r['total_predictions'], 1) for r in completed) / len(completed) * 100,
                "avg_scoring_rate": sum(r['scoring_rate'] for r in completed) / len(completed),
                "termination_reasons": {}
            })
            
            # Count termination reasons
            for result in completed:
                reason = result.get('termination_reason') or 'normal_completion'
                analysis['termination_reasons'][reason] = analysis['termination_reasons'].get(reason, 0) + 1
        
        return analysis


def load_config_from_file(config_path: str) -> SimulationConfig:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        data = json.load(f)
    return SimulationConfig(**data)


def main():
    parser = argparse.ArgumentParser(description='NBA Prediction Orchestration System')
    
    # Configuration options
    parser.add_argument('--config', type=str, help='Path to JSON configuration file')
    parser.add_argument('--games', type=str, help='Comma-separated list of game IDs')
    parser.add_argument('--runs-per-game', type=int, default=5, help='Number of runs per game')
    parser.add_argument('--season', type=str, default='2023-2024', help='Season year (YYYY-YYYY)')
    parser.add_argument('--platform', type=str, default='gemini', choices=['openai', 'gemini'], help='AI platform')
    parser.add_argument('--max-iterations', type=int, default=2000, help='Max iterations per run')
    parser.add_argument('--output-db', type=str, default='simulation_results.db', help='Output database file')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    # Action options
    parser.add_argument('--analyze', action='store_true', help='Analyze existing results')
    parser.add_argument('--analyze-game', type=str, help='Analyze results for specific game ID')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config_from_file(args.config)
    else:
        if not args.games:
            print("Error: Either --config or --games must be specified")
            return
        
        games = [g.strip() for g in args.games.split(',')]
        config = SimulationConfig(
            season_year=args.season,
            games=games,
            runs_per_game=args.runs_per_game,
            max_iterations_per_run=args.max_iterations,
            platform=args.platform,
            output_db=args.output_db,
            log_level=args.log_level
        )
    
    # Create orchestrator
    orchestrator = NBA_Orchestrator(config)
    
    if args.analyze:
        # Analyze all results
        analysis = orchestrator.analyze_results()
        print(f"\n📊 Analysis Results - {analysis['title']}")
        print(f"Total simulations: {analysis['total_simulations']}")
        print(f"Completed: {analysis['completed_simulations']} ({analysis['success_rate']:.1f}%)")
        print(f"Average duration: {analysis['avg_duration_minutes']:.1f} minutes")
        if 'avg_predictions_per_run' in analysis:
            print(f"Average predictions per run: {analysis['avg_predictions_per_run']:.1f}")
            print(f"Average scoring rate: {analysis['avg_scoring_rate']:.1f}%")
    
    elif args.analyze_game:
        # Analyze specific game
        analysis = orchestrator.analyze_results(args.analyze_game)
        if 'error' not in analysis:
            print(f"\n📊 Analysis Results - {analysis['title']}")
            print(f"Total simulations: {analysis['total_simulations']}")
            print(f"Completed: {analysis['completed_simulations']} ({analysis['success_rate']:.1f}%)")
    
    else:
        # Run simulations
        summary = orchestrator.run_all_simulations()
        
        # Show final analysis
        analysis = orchestrator.analyze_results()
        print(f"\n📊 Final Analysis:")
        print(f"Success rate: {analysis['success_rate']:.1f}%")
        if 'avg_scoring_rate' in analysis:
            print(f"Average scoring rate: {analysis['avg_scoring_rate']:.1f}%")


if __name__ == "__main__":
    main()
