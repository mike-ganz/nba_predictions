#!/usr/bin/env python3
"""
Enhanced NBA Prediction Orchestration System

Improved version that properly handles season/game ID relationships
and provides better validation and auto-detection capabilities.
"""

import json
import argparse
import threading
import time
import signal
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

# Import the original orchestrator components
from orchestrator import (
    SimulationConfig, SimulationResult, SimulationDatabase, 
    NBA_Orchestrator as BaseOrchestrator, load_config_from_file
)

# Import the new validator
from season_game_validator import SeasonGameValidator


@dataclass
class ThreadInfo:
    """Information about a running thread."""
    thread_id: str
    game_id: str
    run_num: int
    start_time: datetime
    status: str  # "starting", "stage1", "rolling", "completed", "error"
    current_iteration: int = 0
    current_quarter: int = 1
    current_time: str = "12:00"
    current_score: str = ""
    future: Optional[Future] = None


class EnhancedSimulationConfig(SimulationConfig):
    """Enhanced configuration with season validation and threading support."""
    
    def __init__(self, **kwargs):
        # Auto-detect seasons if not specified
        if 'games' in kwargs and 'season_year' not in kwargs:
            validator = SeasonGameValidator()
            suggested_season = validator.suggest_season_for_games(kwargs['games'])
            if suggested_season:
                kwargs['season_year'] = suggested_season
                print(f"🔍 Auto-detected season: {suggested_season}")
        
        # Set Stage 1 enabled by default (matching first_n_plays training mode)
        if 'skip_stage1' not in kwargs:
            kwargs['skip_stage1'] = False
        
        # Set default threading
        if 'max_threads' not in kwargs:
            kwargs['max_threads'] = 1  # Default to single-threaded for safety
        
        super().__init__(**kwargs)
        
        # Validate configuration after initialization
        self._validate_config()
    
    def _validate_config(self):
        """Validate that games match the configured season."""
        validator = SeasonGameValidator()
        
        # Check each game
        invalid_games = []
        season_mismatches = []
        
        for game_id in self.games:
            is_valid, message = validator.validate_game_season_match(game_id, self.season_year)
            if not is_valid:
                game_info = validator.parse_game_id(game_id)
                if game_info.is_valid and game_info.detected_season != self.season_year:
                    season_mismatches.append((game_id, game_info.detected_season))
                else:
                    invalid_games.append((game_id, message))
        
        if invalid_games or season_mismatches:
            print(f"\n⚠️ Configuration Validation Issues:")
            print(f"   Configured Season: {self.season_year}")
            
            if invalid_games:
                print(f"   Invalid Game IDs:")
                for game_id, error in invalid_games:
                    print(f"     • {game_id}: {error}")
            
            if season_mismatches:
                print(f"   Season Mismatches:")
                for game_id, detected_season in season_mismatches:
                    print(f"     • {game_id} belongs to {detected_season}")
                
                # Group mismatches by season
                season_groups = {}
                for game_id, detected_season in season_mismatches:
                    if detected_season not in season_groups:
                        season_groups[detected_season] = []
                    season_groups[detected_season].append(game_id)
                
                print(f"\n💡 Suggestion: Consider running separate orchestrations:")
                print(f"   Current season ({self.season_year}): {[g for g in self.games if g not in [gid for gid, _ in season_mismatches]]}")
                for season, games in season_groups.items():
                    print(f"   {season}: {games}")
    
    def get_validation_report(self) -> str:
        """Get detailed validation report."""
        validator = SeasonGameValidator()
        return validator.format_validation_report(self.games, self.season_year)


class Enhanced_NBA_Orchestrator(BaseOrchestrator):
    """Enhanced orchestrator with better season handling and multithreading support."""
    
    def __init__(self, config: EnhancedSimulationConfig):
        # Validate and potentially fix configuration
        self.validator = SeasonGameValidator()
        self.config = config
        
        # Initialize threading components
        self._completed_runs = 0
        self._failed_runs = 0
        self._progress_lock = threading.Lock()
        self._thread_info_lock = threading.Lock()
        
        # Thread tracking and control
        self._active_threads: Dict[str, ThreadInfo] = {}
        self._shutdown_requested = threading.Event()
        self._cancelled_threads: Set[str] = set()
        
        # Monitoring thread
        self._monitor_thread = None
        self._monitor_active = False
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Initialize with validated configuration
        super().__init__(config)
        
        # Track database save failures for reporting
        self._save_failures = []
        self._save_failure_lock = threading.Lock()
        
        # Set up dedicated database error logging
        self._setup_database_error_logging()
        
        # Enhance database configuration for better multithreading
        self._enhance_database_config()
    
    def _setup_database_error_logging(self):
        """Set up dedicated logging for database errors."""
        # Create a separate logger for database errors
        self.db_error_logger = logging.getLogger('NBA_Database_Errors')
        self.db_error_logger.setLevel(logging.WARNING)
        
        # Create dedicated file handler for database errors
        db_error_file = f"database_errors_{int(time.time())}.log"
        db_error_handler = logging.FileHandler(db_error_file)
        db_error_formatter = logging.Formatter(
            '%(asctime)s - THREAD[%(thread)d] - %(levelname)s - %(message)s'
        )
        db_error_handler.setFormatter(db_error_formatter)
        self.db_error_logger.addHandler(db_error_handler)
        
        # Also create a console handler with distinctive formatting for critical errors
        console_error_handler = logging.StreamHandler()
        console_error_formatter = logging.Formatter(
            '\n🔥 DATABASE ERROR: %(message)s\n'
        )
        console_error_handler.setFormatter(console_error_formatter)
        console_error_handler.setLevel(logging.ERROR)
        self.db_error_logger.addHandler(console_error_handler)
        
        self.logger.info(f"Database error logging initialized: {db_error_file}")
    
    def _enhance_database_config(self):
        """Enhance database settings for better concurrent access."""
        try:
            import sqlite3
            with sqlite3.connect(self.config.output_db, check_same_thread=False) as conn:
                # Optimize for concurrent access
                conn.execute('PRAGMA journal_mode=WAL;')      # Write-Ahead Logging for better concurrency
                conn.execute('PRAGMA synchronous=NORMAL;')    # Balance safety and performance  
                conn.execute('PRAGMA cache_size=20000;')      # Increase cache size for better performance
                conn.execute('PRAGMA temp_store=MEMORY;')     # Store temp tables in memory
                conn.execute('PRAGMA mmap_size=268435456;')   # Enable memory-mapped I/O (256MB)
                conn.execute('PRAGMA wal_autocheckpoint=1000;') # Checkpoint WAL file every 1000 pages
                conn.execute('PRAGMA busy_timeout=30000;')    # 30 second timeout for busy database
                conn.commit()
                
                self.logger.debug("Enhanced database configuration for multithreading")
                
        except Exception as e:
            self.logger.warning(f"Failed to enhance database configuration: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals for graceful shutdown."""
        print(f"\n🛑 Received signal {signum}. Initiating graceful shutdown...")
        self._shutdown_requested.set()
        self._stop_monitoring()
        
        # Show active threads before shutdown
        self._show_thread_status()
        
        # Give threads more time to finish database operations
        print("⏳ Waiting for active threads to complete database writes...")
        
        # Wait longer and monitor for completion
        max_wait_time = 30  # 30 seconds should be enough for most simulations
        start_wait = time.time()
        
        while time.time() - start_wait < max_wait_time:
            with self._thread_info_lock:
                active_count = len([t for t in self._active_threads.values() 
                                  if t.future and not t.future.done()])
            
            if active_count == 0:
                print("✅ All threads completed. Database writes should be safe.")
                break
                
            print(f"   {active_count} threads still active... waiting")
            time.sleep(2)
        
        # Final database checkpoint to ensure WAL is committed
        try:
            print("💾 Performing final database checkpoint...")
            import sqlite3
            with sqlite3.connect(self.config.output_db, check_same_thread=False) as conn:
                conn.execute("PRAGMA wal_checkpoint(FULL);")
                conn.commit()
            print("✅ Database checkpoint completed")
        except Exception as e:
            print(f"⚠️ Database checkpoint failed: {e}")
        
        print("🏁 Graceful shutdown complete")
        sys.exit(0)
    
    def _register_thread(self, thread_id: str, game_id: str, run_num: int, future: Future) -> ThreadInfo:
        """Register a new thread for monitoring."""
        thread_info = ThreadInfo(
            thread_id=thread_id,
            game_id=game_id,
            run_num=run_num,
            start_time=datetime.now(),
            status="starting",
            future=future
        )
        
        with self._thread_info_lock:
            self._active_threads[thread_id] = thread_info
        
        return thread_info
    
    def _update_thread_status(self, thread_id: str, status: str, **kwargs):
        """Update thread status and additional info."""
        with self._thread_info_lock:
            if thread_id in self._active_threads:
                thread_info = self._active_threads[thread_id]
                thread_info.status = status
                
                # Update optional fields
                if 'iteration' in kwargs:
                    thread_info.current_iteration = kwargs['iteration']
                if 'quarter' in kwargs:
                    thread_info.current_quarter = kwargs['quarter']
                if 'time' in kwargs:
                    thread_info.current_time = kwargs['time']
                if 'score' in kwargs:
                    thread_info.current_score = kwargs['score']
    
    def _unregister_thread(self, thread_id: str):
        """Remove thread from active tracking."""
        with self._thread_info_lock:
            if thread_id in self._active_threads:
                del self._active_threads[thread_id]
    
    def _show_thread_status(self):
        """Display current status of all active threads."""
        with self._thread_info_lock:
            active_threads = dict(self._active_threads)
        
        if not active_threads:
            print("📭 No active threads")
            return
        
        print(f"\n🧵 Active Threads ({len(active_threads)}):")
        print("=" * 80)
        print(f"{'ID':<20} {'Game':<12} {'Run':<4} {'Status':<10} {'Iter':<6} {'Game State':<25} {'Duration'}")
        print("-" * 80)
        
        for thread_id, info in active_threads.items():
            duration = (datetime.now() - info.start_time).total_seconds()
            duration_str = f"{duration:.0f}s"
            
            game_state = ""
            if info.current_score:
                game_state = f"Q{info.current_quarter} {info.current_time} | {info.current_score}"
            elif info.status == "stage1":
                game_state = "Initial plays generation"
            
            print(f"{thread_id[:18]:<20} {info.game_id:<12} {info.run_num:<4} {info.status:<10} "
                  f"{info.current_iteration:<6} {game_state:<25} {duration_str}")
    
    def cancel_thread(self, thread_id: str) -> bool:
        """Cancel a specific thread by ID."""
        with self._thread_info_lock:
            if thread_id in self._active_threads:
                thread_info = self._active_threads[thread_id]
                if thread_info.future and not thread_info.future.done():
                    success = thread_info.future.cancel()
                    if success:
                        self._cancelled_threads.add(thread_id)
                        print(f"✅ Cancelled thread {thread_id}")
                        return True
                    else:
                        print(f"⚠️ Could not cancel thread {thread_id} (already running)")
                        return False
                else:
                    print(f"⚠️ Thread {thread_id} is not active or already completed")
                    return False
            else:
                print(f"❌ Thread {thread_id} not found")
                return False
    
    def cancel_all_threads(self):
        """Cancel all active threads."""
        with self._thread_info_lock:
            thread_ids = list(self._active_threads.keys())
        
        cancelled_count = 0
        for thread_id in thread_ids:
            if self.cancel_thread(thread_id):
                cancelled_count += 1
        
        print(f"🛑 Cancelled {cancelled_count}/{len(thread_ids)} threads")
        return cancelled_count
    
    def _start_monitoring(self):
        """Start the thread monitoring background process."""
        if self._monitor_active:
            return
        
        self._monitor_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        print("👁️ Thread monitoring started. Press 'Ctrl+C' twice to show status, 'Ctrl+C' thrice to exit")
    
    def _stop_monitoring(self):
        """Stop the thread monitoring process."""
        self._monitor_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1)
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        last_status_time = time.time()
        
        while self._monitor_active and not self._shutdown_requested.is_set():
            # Show status every 30 seconds
            if time.time() - last_status_time > 30:
                print("\n" + "="*60)
                self._show_thread_status()
                print("="*60)
                last_status_time = time.time()
            
            time.sleep(1)
    
    def _run_simulation_task(self, game_id: str, run_num: int, total_runs: int, 
                           thread_id: Optional[str] = None) -> SimulationResult:
        """Run a single simulation as a task that can be executed in parallel."""
        run_id = f"{game_id}_{self.config.season_year}_{run_num:04d}_{int(time.time())}"
        
        # Use provided thread_id or create one
        if not thread_id:
            thread_id = run_id
        
        # Check if thread was cancelled before starting
        if thread_id in self._cancelled_threads:
            self._unregister_thread(thread_id)
            cancelled_result = self._create_cancelled_result(run_id, game_id)
            # Result will be saved by the monitoring method, don't save twice
            return cancelled_result
        
        # Update thread status
        self._update_thread_status(thread_id, "starting")
        
        # Thread-safe progress tracking
        with self._progress_lock:
            current_run = self._completed_runs + self._failed_runs + 1
            self.logger.info(f"Starting run {current_run}/{total_runs}: {run_id}")
        
        try:
            # Update status to stage1
            self._update_thread_status(thread_id, "stage1")
            
            # Use the parent class's single simulation method (correct signature: run_id, game_id)
            result = self._run_single_simulation_monitored(run_id, game_id, thread_id)
            
            # Update progress
            with self._progress_lock:
                if result.status in ['completed', 'game_ended']:
                    self._completed_runs += 1
                    self._update_thread_status(thread_id, "completed")
                    self.logger.info(f"Completed {run_id} ({self._completed_runs}/{total_runs} successful)")
                else:
                    self._failed_runs += 1
                    self._update_thread_status(thread_id, "error")
                    self.logger.warning(f"Failed {run_id} ({self._failed_runs} failures)")
            
            # Unregister thread
            self._unregister_thread(thread_id)
            return result
            
        except Exception as e:
            with self._progress_lock:
                self._failed_runs += 1
                self.logger.error(f"Exception in {run_id}: {str(e)}")
            
            self._update_thread_status(thread_id, "error")
            self._unregister_thread(thread_id)
            
            # Create error result with all required fields
            start_time = datetime.now()
            result = SimulationResult(
                run_id=run_id,
                game_id=game_id,
                season_year=self.config.season_year,
                platform=self.config.platform,
                start_time=start_time,
                end_time=start_time,  # Same as start for error cases
                duration_seconds=0.0,
                status="error",
                total_predictions=0,
                successful_predictions=0,
                final_score=None,
                final_quarter=None,
                final_time=None,
                termination_reason="Exception during execution",
                scoring_plays=0,
                non_scoring_plays=0,
                scoring_rate=0.0,
                error_message=str(e),
                iterations_data="{}",
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
                # Validation failure information (None for error cases)
                validation_failure_timestamp=None,
                validation_total_failed_attempts=None,
                validation_most_common_reason=None,
                validation_most_common_reason_count=None,
                validation_most_common_error_type=None,
                validation_most_common_error_type_count=None,
                validation_most_common_field=None,
                validation_most_common_field_count=None,
                validation_unique_reasons=None,
                validation_unique_error_types=None,
                validation_unique_fields=None,
                validation_failure_summary=None,
                validation_response_examples=None
            )
            
            # Save error result - this is a top-level exception so save immediately
            self._safe_save_result(result, thread_id)
            return result
    
    def _run_single_simulation_monitored(self, run_id: str, game_id: str, thread_id: str) -> SimulationResult:
        """Run single simulation with thread monitoring."""
        # Check for cancellation frequently during execution
        if thread_id in self._cancelled_threads:
            cancelled_result = self._create_cancelled_result(run_id, game_id)
            # Save cancelled result and return
            self._safe_save_result(cancelled_result, thread_id)
            return cancelled_result
        
        # This would be where we'd integrate monitoring into the actual simulation
        # For now, we use the parent method but could enhance it with periodic status updates
        result = self._run_single_simulation(run_id, game_id)
        
        # 🔧 ENHANCED FIX: Save result to database with robust error handling
        # This is the single point where all successful results are saved
        self._safe_save_result(result, thread_id)
        
        return result
    
    def _safe_save_result(self, result: SimulationResult, thread_id: str) -> bool:
        """Safely save result to database with retry logic and proper error handling."""
        max_retries = 3
        retry_delay = 0.1  # Start with 100ms delay
        
        for attempt in range(max_retries):
            try:
                # Save to database
                self.db.save_result(result)
                
                # Log successful save with thread info
                thread_name = threading.current_thread().name
                self.logger.debug(f"Successfully saved result {result.run_id} to database "
                                f"(thread: {thread_name}, attempt: {attempt + 1})")
                return True
                
            except Exception as e:
                thread_name = threading.current_thread().name
                error_context = {
                    'run_id': result.run_id,
                    'game_id': result.game_id,
                    'thread_name': thread_name,
                    'thread_id': thread_id,
                    'attempt': attempt + 1,
                    'error': str(e),
                    'status': result.status,
                    'duration': result.duration_seconds
                }
                
                if attempt < max_retries - 1:
                    # Log retry attempt to dedicated database error log
                    self.db_error_logger.warning(
                        f"RETRY {attempt + 1}/{max_retries}: {result.run_id} "
                        f"({result.game_id}) - {str(e)} - Retrying in {retry_delay}s"
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Final attempt failed - log extensively and track failure
                    error_msg = (f"PERMANENT FAILURE: {result.run_id} ({result.game_id}) "
                               f"- Failed after {max_retries} attempts: {str(e)}")
                    
                    # Log to dedicated database error logger (both file and console with 🔥 format)
                    self.db_error_logger.error(error_msg)
                    
                    # Log additional details to database error file
                    self.db_error_logger.error(
                        f"Details: Status={result.status}, Duration={result.duration_seconds:.1f}s, "
                        f"Thread={thread_name}, ThreadID={thread_id}"
                    )
                    
                    # Track failure for summary reporting
                    with self._save_failure_lock:
                        self._save_failures.append(error_context)
                    
                    # Update thread status to show save error but continue processing
                    self._update_thread_status(thread_id, "save_error")
                    return False
        
        return False
    
    def _create_cancelled_result(self, run_id: str, game_id: str) -> SimulationResult:
        """Create a result for a cancelled simulation."""
        start_time = datetime.now()
        return SimulationResult(
            run_id=run_id,
            game_id=game_id,
            season_year=self.config.season_year,
            platform=self.config.platform,
            start_time=start_time,
            end_time=start_time,
            duration_seconds=0.0,
            status="cancelled",
            total_predictions=0,
            successful_predictions=0,
            final_score=None,
            final_quarter=None,
            final_time=None,
            termination_reason="Cancelled by user",
            scoring_plays=0,
            non_scoring_plays=0,
            scoring_rate=0.0,
            error_message="Thread was cancelled",
            iterations_data="{}",
            # Validation termination information (None for cancelled cases)
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
            # Validation failure information (None for cancelled cases)
            validation_failure_timestamp=None,
            validation_total_failed_attempts=None,
            validation_most_common_reason=None,
            validation_most_common_reason_count=None,
            validation_most_common_error_type=None,
            validation_most_common_error_type_count=None,
            validation_most_common_field=None,
            validation_most_common_field_count=None,
            validation_unique_reasons=None,
            validation_unique_error_types=None,
            validation_unique_fields=None,
            validation_failure_summary=None,
            validation_response_examples=None
        )
    
    def run_all_simulations(self) -> Dict[str, Any]:
        """Run simulations with enhanced season handling and multithreading support."""
        
        # Check if we have games from multiple seasons
        season_groups = self.validator.auto_detect_seasons(self.config.games)
        
        if len(season_groups) > 1:
            print(f"\n🔄 Multiple seasons detected: {list(season_groups.keys())}")
            print(f"   Current configuration targets: {self.config.season_year}")
            
            # Filter to only games that match the configured season
            valid_games = season_groups.get(self.config.season_year, [])
            
            if valid_games:
                print(f"   Running simulations for {len(valid_games)} games in {self.config.season_year}")
                self.config.games = valid_games
            else:
                print(f"   ❌ No games found for configured season {self.config.season_year}")
                print(f"   Available seasons: {list(season_groups.keys())}")
                
                # For automated runs, use the most common season
                most_common_season = max(season_groups.keys(), key=lambda k: len(season_groups[k]))
                print(f"\n⚡ Auto-selecting most common season: {most_common_season}")
                
                self.config.season_year = most_common_season
                self.config.games = season_groups[most_common_season]
                
                # Reinitialize with correct season
                from config.settings import set_season_year
                set_season_year(most_common_season)
        
        # Calculate total runs
        total_runs = len(self.config.games) * self.config.runs_per_game
        
        # Reset counters
        self._completed_runs = 0
        self._failed_runs = 0
        
        self.logger.info(f"Starting multithreaded orchestration:")
        self.logger.info(f"   Total simulations: {total_runs}")
        self.logger.info(f"   Games: {self.config.games}")
        self.logger.info(f"   Runs per game: {self.config.runs_per_game}")
        self.logger.info(f"   Season: {self.config.season_year}")
        self.logger.info(f"   Platform: {self.config.platform}")
        self.logger.info(f"   Max threads: {self.config.max_threads}")
        
        start_time = time.time()
        
        # Create list of all simulation tasks
        simulation_tasks = []
        for game_id in self.config.games:
            for run_num in range(self.config.runs_per_game):
                simulation_tasks.append((game_id, run_num + 1, total_runs))
        
        # Execute simulations using ThreadPoolExecutor
        if self.config.max_threads == 1:
            # Single-threaded execution for compatibility
            self.logger.info("Running in single-threaded mode")
            results = []
            for game_id, run_num, total in simulation_tasks:
                if self._shutdown_requested.is_set():
                    break
                result = self._run_simulation_task(game_id, run_num, total)
                results.append(result)
        
        else:
            # Multithreaded execution
            self.logger.info(f"Running with {self.config.max_threads} threads")
            self._start_monitoring()  # Start thread monitoring
            
            results = []
            
            with ThreadPoolExecutor(max_workers=self.config.max_threads) as executor:
                # Submit all tasks and register them for monitoring
                future_to_params = {}
                
                for game_id, run_num, total in simulation_tasks:
                    if self._shutdown_requested.is_set():
                        break
                    
                    # Create thread ID
                    thread_id = f"{game_id}_{run_num}_{int(time.time())}"
                    
                    # Submit task
                    future = executor.submit(self._run_simulation_task, game_id, run_num, total, thread_id)
                    future_to_params[future] = (game_id, run_num, thread_id)
                    
                    # Register thread for monitoring
                    self._register_thread(thread_id, game_id, run_num, future)
                
                print(f"🚀 Submitted {len(future_to_params)} tasks to thread pool")
                self._show_thread_status()
                
                # Collect results as they complete
                for future in as_completed(future_to_params):
                    if self._shutdown_requested.is_set():
                        break
                        
                    game_id, run_num, thread_id = future_to_params[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Progress update
                        completed_total = self._completed_runs + self._failed_runs
                        if completed_total % 5 == 0 or completed_total == total_runs:  # Update every 5 runs
                            with self._thread_info_lock:
                                save_errors = len([t for t in self._active_threads.values() 
                                                 if t.status == "save_error"])
                            
                            with self._save_failure_lock:
                                total_save_failures = len(self._save_failures)
                            
                            progress_msg = f"Progress: {completed_total}/{total_runs} simulations completed " \
                                         f"({completed_total/total_runs*100:.1f}%)"
                            
                            if total_save_failures > 0:
                                progress_msg += f" [🔥 {total_save_failures} DATABASE SAVE FAILURES]"
                            elif save_errors > 0:
                                progress_msg += f" [⚠️ {save_errors} threads with save errors]"
                            
                            self.logger.info(progress_msg)
                            
                            # Show prominent warning if we have database failures
                            if total_save_failures > 0 and completed_total % 10 == 0:
                                print(f"\n⚠️⚠️⚠️ DATABASE ALERT: {total_save_failures} results failed to save! ⚠️⚠️⚠️")
                                print("   Check database_errors_*.log for details\n")
                            
                    except Exception as e:
                        self.logger.error(f"Task failed for {game_id} run {run_num}: {str(e)}")
                        self._unregister_thread(thread_id)
            
            # Stop monitoring
            self._stop_monitoring()
        
        # Calculate summary statistics
        end_time = time.time()
        total_duration = end_time - start_time
        
        summary = {
            'total_simulations': total_runs,
            'completed_simulations': self._completed_runs,
            'failed_simulations': self._failed_runs,
            'success_rate': (self._completed_runs / total_runs * 100) if total_runs > 0 else 0,
            'total_duration_seconds': total_duration,
            'total_duration_minutes': total_duration / 60,
            'avg_duration_per_simulation': total_duration / total_runs if total_runs > 0 else 0,
            'threads_used': self.config.max_threads
        }
        
        # Log final summary with database save failure information
        self.logger.info(f"\nFinal Summary:")
        self.logger.info(f"   Completed: {self._completed_runs}/{total_runs} ({summary['success_rate']:.1f}%)")
        self.logger.info(f"   Failed: {self._failed_runs}")
        self.logger.info(f"   Total time: {summary['total_duration_minutes']:.1f} minutes")
        self.logger.info(f"   Avg per simulation: {summary['avg_duration_per_simulation']:.1f} seconds")
        if self.config.max_threads > 1:
            theoretical_single_thread_time = total_duration * self.config.max_threads
            speedup = theoretical_single_thread_time / total_duration if total_duration > 0 else 1
            self.logger.info(f"   Threading speedup: ~{speedup:.1f}x (with {self.config.max_threads} threads)")
        
        # Report database save failures prominently
        self._report_database_save_failures()
        
        return summary
    
    def _report_database_save_failures(self):
        """Generate prominent report of database save failures."""
        with self._save_failure_lock:
            failures = list(self._save_failures)  # Copy the list
        
        if not failures:
            self.logger.info("   Database saves: ✅ All successful")
            return
        
        # Print prominent console warning
        print("\n" + "="*80)
        print("🔥🔥🔥 DATABASE SAVE FAILURES DETECTED 🔥🔥🔥")
        print("="*80)
        print(f"   {len(failures)} simulation results FAILED to save to database!")
        print(f"   Check database_errors_*.log for detailed error information")
        
        # Group failures by error type
        error_types = {}
        games_affected = set()
        
        for failure in failures:
            error_key = failure['error'][:50] + "..." if len(failure['error']) > 50 else failure['error']
            if error_key not in error_types:
                error_types[error_key] = []
            error_types[error_key].append(failure)
            games_affected.add(failure['game_id'])
        
        print(f"\nFailed Results Summary:")
        print(f"   Games affected: {len(games_affected)} ({', '.join(sorted(games_affected))})")
        print(f"   Error types: {len(error_types)}")
        
        # Show most common errors
        for error_type, error_failures in sorted(error_types.items(), 
                                                key=lambda x: len(x[1]), reverse=True):
            print(f"     • {error_type}: {len(error_failures)} failures")
        
        print("\n💡 Recommendations:")
        print("   1. Check database_errors_*.log for full error details")
        print("   2. Verify database file permissions and disk space")
        print("   3. Consider reducing thread count to reduce database contention")
        print("   4. Run the affected games individually to retry failed saves")
        print("="*80 + "\n")
        
        # Also log to main logger for permanent record
        self.logger.error(f"DATABASE SAVE SUMMARY: {len(failures)} results failed to save")
        for error_type, error_failures in error_types.items():
            self.logger.error(f"   {error_type}: {len(error_failures)} occurrences")
    
    def _show_database_status(self):
        """Show current database save status for interactive monitoring."""
        with self._save_failure_lock:
            failures = list(self._save_failures)
        
        with self._thread_info_lock:
            save_error_threads = [t for t in self._active_threads.values() 
                                 if t.status == "save_error"]
        
        print(f"\n💾 Database Save Status:")
        print("=" * 50)
        
        if not failures and not save_error_threads:
            print("✅ All database saves successful so far")
        else:
            print(f"🔥 Failed saves: {len(failures)}")
            print(f"⚠️ Threads with save errors: {len(save_error_threads)}")
            
            if failures:
                # Group by game and error type
                games_affected = set(f['game_id'] for f in failures)
                error_types = set(f['error'][:30] + "..." if len(f['error']) > 30 
                                else f['error'] for f in failures)
                
                print(f"\nFailed Results Details:")
                print(f"   Games affected: {len(games_affected)}")
                print(f"   Games: {', '.join(sorted(games_affected))}")
                print(f"   Error types: {len(error_types)}")
                
                # Show recent failures
                recent_failures = failures[-5:] if len(failures) > 5 else failures
                print(f"\nMost recent failures:")
                for failure in recent_failures:
                    error_short = failure['error'][:40] + "..." if len(failure['error']) > 40 else failure['error']
                    print(f"     • {failure['run_id']} ({failure['game_id']}): {error_short}")
                
                if len(failures) > 5:
                    print(f"     ... and {len(failures) - 5} more (see database_errors_*.log)")
            
            if save_error_threads:
                print(f"\nThreads currently with save errors:")
                for thread in save_error_threads:
                    print(f"     • {thread.thread_id} - {thread.game_id}")
                    
        print("=" * 50)
    
    def interactive_thread_control(self):
        """Interactive thread monitoring and control interface."""
        print("\n🎮 Interactive Thread Control")
        print("Commands: 'status' | 'db-status' | 'cancel <thread_id>' | 'cancel-all' | 'quit'")
        print("=" * 60)
        
        while not self._shutdown_requested.is_set():
            try:
                command = input("\n> ").strip().lower()
                
                if command == 'quit' or command == 'q':
                    print("Exiting thread control...")
                    break
                    
                elif command == 'status' or command == 's':
                    self._show_thread_status()
                    
                elif command == 'db-status' or command == 'db':
                    self._show_database_status()
                    
                elif command == 'cancel-all' or command == 'ca':
                    self.cancel_all_threads()
                    
                elif command.startswith('cancel ') or command.startswith('c '):
                    parts = command.split(' ', 1)
                    if len(parts) > 1:
                        thread_id = parts[1].strip()
                        self.cancel_thread(thread_id)
                    else:
                        print("❌ Please specify a thread ID to cancel")
                        
                elif command == 'help' or command == 'h':
                    print("\n📋 Available commands:")
                    print("  status, s          - Show active threads")
                    print("  db-status, db      - Show database save status")
                    print("  cancel <id>, c <id> - Cancel specific thread")
                    print("  cancel-all, ca     - Cancel all threads")  
                    print("  help, h            - Show this help")
                    print("  quit, q            - Exit control interface")
                    
                elif command == '':
                    continue  # Empty command, just show prompt again
                    
                else:
                    print(f"❌ Unknown command: {command}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n🛑 Ctrl+C pressed. Use 'quit' to exit or 'cancel-all' to stop all threads.")
                continue
            except EOFError:
                print("\nExiting thread control...")
                break


def create_enhanced_config_from_args(args) -> EnhancedSimulationConfig:
    """Create enhanced configuration from command line arguments."""
    
    if args.config:
        # Load base config and enhance it
        base_data = {}
        with open(args.config, 'r') as f:
            base_data = json.load(f)
        return EnhancedSimulationConfig(**base_data)
    else:
        if not args.games:
            raise ValueError("Either --config or --games must be specified")
        
        games = [g.strip() for g in args.games.split(',')]
        
        config_data = {
            'games': games,
            'runs_per_game': args.runs_per_game,
            'max_iterations_per_run': args.max_iterations,
            'platform': args.platform,
            'output_db': args.output_db,
            'log_level': args.log_level,
            'max_threads': getattr(args, 'threads', 1)  # Default to 1 thread if not specified
        }
        
        # Only set season if explicitly provided, otherwise let auto-detection work
        if args.season:
            config_data['season_year'] = args.season
        
        return EnhancedSimulationConfig(**config_data)


def main():
    parser = argparse.ArgumentParser(description='Enhanced NBA Prediction Orchestration System')
    
    # Configuration options
    parser.add_argument('--config', type=str, help='Path to JSON configuration file')
    parser.add_argument('--games', type=str, help='Comma-separated list of game IDs')
    parser.add_argument('--runs-per-game', type=int, default=5, help='Number of runs per game')
    parser.add_argument('--season', type=str, help='Season year (YYYY-YYYY) - auto-detected if not provided')
    parser.add_argument('--platform', type=str, default='gemini', choices=['openai', 'gemini'], help='AI platform')
    parser.add_argument('--max-iterations', type=int, default=2000, help='Max iterations per run')
    parser.add_argument('--output-db', type=str, default='enhanced_simulation_results.db', help='Output database file')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--threads', type=int, default=1, help='Number of threads for parallel execution (default: 1)')
    parser.add_argument('--monitor-threads', action='store_true', help='Enable interactive thread monitoring and control')
    
    # Validation options
    parser.add_argument('--validate-only', action='store_true', help='Only validate game IDs, don\'t run simulations')
    parser.add_argument('--show-seasons', action='store_true', help='Show supported seasons')
    parser.add_argument('--auto-detect', type=str, help='Auto-detect seasons for comma-separated game IDs')
    
    # Action options  
    parser.add_argument('--analyze', action='store_true', help='Analyze existing results')
    parser.add_argument('--analyze-game', type=str, help='Analyze results for specific game ID')
    
    args = parser.parse_args()
    
    validator = SeasonGameValidator()
    
    if args.show_seasons:
        seasons = validator.get_supported_seasons()
        print("🏀 Supported NBA Seasons:")
        for season in seasons:
            print(f"  • {season}")
        return
    
    if args.auto_detect:
        games = [g.strip() for g in args.auto_detect.split(',')]
        print(f"🔍 Auto-detecting seasons for {len(games)} games...")
        
        season_groups = validator.auto_detect_seasons(games)
        
        print(f"\n📊 Results:")
        for season, game_list in season_groups.items():
            print(f"  {season}: {len(game_list)} games")
            for game_id in game_list[:5]:  # Show first 5
                print(f"    • {game_id}")
            if len(game_list) > 5:
                print(f"    ... and {len(game_list) - 5} more")
        
        suggested = validator.suggest_season_for_games(games)
        if suggested:
            print(f"\n💡 Suggested season for mixed batch: {suggested}")
        
        return
    
    # Create enhanced configuration
    try:
        config = create_enhanced_config_from_args(args)
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return
    
    if args.validate_only:
        print("🔍 Validation Report:")
        print(config.get_validation_report())
        return
    
    # Create enhanced orchestrator
    orchestrator = Enhanced_NBA_Orchestrator(config)
    
    if args.analyze:
        # Analyze all results
        analysis = orchestrator.analyze_results()
        print(f"\n📊 Analysis Results - {analysis.get('title', 'All Games')}")
        print(f"Total simulations: {analysis.get('total_simulations', 0)}")
        print(f"Completed: {analysis.get('completed_simulations', 0)} ({analysis.get('success_rate', 0):.1f}%)")
        if 'avg_scoring_rate' in analysis:
            print(f"Average scoring rate: {analysis['avg_scoring_rate']:.1f}%")
    
    elif args.analyze_game:
        # Analyze specific game
        analysis = orchestrator.analyze_results(args.analyze_game)
        if 'error' not in analysis:
            print(f"\n📊 Analysis Results - {analysis.get('title', args.analyze_game)}")
            print(f"Total simulations: {analysis.get('total_simulations', 0)}")
            print(f"Completed: {analysis.get('completed_simulations', 0)} ({analysis.get('success_rate', 0):.1f}%)")
    
    else:
        # Run simulations
        print("\n🏀 Enhanced NBA Prediction Orchestration")
        print("=" * 60)
        print(f"Configuration validation: {'✅ Passed' if len(config.games) > 0 else '❌ Failed'}")
        print(f"Season: {config.season_year}")
        print(f"Games: {len(config.games)} games")
        print(f"Runs per game: {config.runs_per_game}")
        print(f"Total simulations: {len(config.games) * config.runs_per_game}")
        print(f"Threading: {config.max_threads} thread{'s' if config.max_threads != 1 else ''}")
        
        # Start interactive monitoring if requested
        if args.monitor_threads and config.max_threads > 1:
            print("🎮 Interactive thread monitoring enabled")
            
            # Run simulations in a background thread
            import threading
            
            simulation_thread = threading.Thread(
                target=lambda: setattr(orchestrator, '_simulation_result', orchestrator.run_all_simulations()),
                daemon=False
            )
            simulation_thread.start()
            
            # Start interactive control
            try:
                orchestrator.interactive_thread_control()
            except KeyboardInterrupt:
                print("\n🛑 Received interrupt during interactive control")
                orchestrator._shutdown_requested.set()
            
            # Wait for simulations to complete
            simulation_thread.join()
            summary = getattr(orchestrator, '_simulation_result', {})
            
        else:
            # Standard execution
            summary = orchestrator.run_all_simulations()
        
        # Show final analysis
        print(f"\n🎯 Final Results:")
        print(f"  Success rate: {summary.get('success_rate', 0):.1f}%")
        print(f"  Total duration: {summary.get('total_duration_minutes', 0):.1f} minutes")
        print(f"  Database: {config.output_db}")
        
        if args.monitor_threads:
            print(f"  Threads used: {summary.get('threads_used', 1)}")


if __name__ == "__main__":
    main()
