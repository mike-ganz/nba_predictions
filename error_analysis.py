#!/usr/bin/env python3
"""
NBA Simulation Error Analysis Tool

Analyzes SQLite write errors and validation termination data from simulation runs.
Provides insights into database save failures and validation-triggered terminations.
"""

import sqlite3
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import argparse
import re


class SimulationErrorAnalyzer:
    """Analyzes simulation errors from database and log files."""
    
    def __init__(self, db_path: str = "enhanced_simulation_results.db"):
        """Initialize the analyzer with database path."""
        self.db_path = db_path
        self.db_exists = Path(db_path).exists()
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def get_database_error_logs(self, days_back: int = 7) -> List[Dict[str, Any]]:
        """Parse database error log files from recent days."""
        error_logs = []
        cutoff_time = datetime.now() - timedelta(days=days_back)
        
        # Find all database error log files
        log_files = glob.glob("database_errors_*.log")
        
        if not log_files:
            print(f"ℹ️ No database error log files found")
            return error_logs
        
        print(f"📁 Found {len(log_files)} database error log files")
        
        for log_file in sorted(log_files):
            try:
                # Extract timestamp from filename
                timestamp_match = re.search(r'database_errors_(\d+)\.log', log_file)
                if timestamp_match:
                    file_timestamp = datetime.fromtimestamp(int(timestamp_match.group(1)))
                    if file_timestamp < cutoff_time:
                        continue
                
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Parse log line
                        log_entry = self._parse_log_line(line, log_file, line_num)
                        if log_entry:
                            error_logs.append(log_entry)
                            
            except Exception as e:
                print(f"⚠️ Error reading {log_file}: {e}")
        
        return sorted(error_logs, key=lambda x: x.get('timestamp', ''), reverse=True)
    
    def _parse_log_line(self, line: str, log_file: str, line_num: int) -> Optional[Dict[str, Any]]:
        """Parse a single log line into structured data."""
        try:
            # Look for timestamp pattern
            timestamp_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})'
            timestamp_match = re.search(timestamp_pattern, line)
            
            # Look for thread ID pattern
            thread_pattern = r'THREAD\[(\d+)\]'
            thread_match = re.search(thread_pattern, line)
            
            # Look for level pattern
            level_pattern = r'- (WARNING|ERROR|CRITICAL) -'
            level_match = re.search(level_pattern, line)
            
            # Extract the message (everything after the level)
            message = line
            if level_match:
                message = line[level_match.end():].strip()
            
            # Determine error type
            error_type = "unknown"
            if "RETRY" in message:
                error_type = "retry_attempt"
            elif "PERMANENT FAILURE" in message:
                error_type = "permanent_failure"
            elif "database is locked" in message:
                error_type = "database_locked"
            elif "disk I/O error" in message:
                error_type = "disk_io_error"
            
            # Extract run_id and game_id if present
            run_id_match = re.search(r'([A-Z0-9_]+_\d{4}-\d{4}_\d{4}_\d+)', message)
            game_id_match = re.search(r'\((\d{8})\)', message)
            
            return {
                'timestamp': timestamp_match.group(1) if timestamp_match else None,
                'thread_id': thread_match.group(1) if thread_match else None,
                'level': level_match.group(1) if level_match else "INFO",
                'message': message,
                'error_type': error_type,
                'run_id': run_id_match.group(1) if run_id_match else None,
                'game_id': game_id_match.group(1) if game_id_match else None,
                'source_file': log_file,
                'line_number': line_num
            }
            
        except Exception as e:
            print(f"⚠️ Error parsing line {line_num} in {log_file}: {e}")
            return None
    
    def get_validation_terminations(self, days_back: int = 7) -> pd.DataFrame:
        """Get validation termination data from database."""
        if not self.db_exists:
            print(f"❌ Database not found: {self.db_path}")
            return pd.DataFrame()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get terminations from recent days
                cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
                
                query = """
                SELECT 
                    run_id, game_id, season_year, start_time, status, 
                    termination_reason, final_score,
                    validation_termination_type, validation_termination_reason,
                    validation_trigger_condition, validation_termination_timestamp,
                    validation_consecutive_count, validation_total_attempts,
                    validation_game_state_quarter, validation_game_state_time,
                    validation_game_state_score, validation_context_json,
                    validation_failure_timestamp, validation_total_failed_attempts,
                    validation_most_common_reason, validation_most_common_reason_count,
                    validation_most_common_error_type, validation_most_common_error_type_count,
                    validation_most_common_field, validation_most_common_field_count,
                    validation_unique_reasons, validation_unique_error_types, validation_unique_fields,
                    validation_failure_summary, validation_response_examples
                FROM simulation_runs 
                WHERE (validation_termination_type IS NOT NULL OR validation_failure_timestamp IS NOT NULL)
                  AND start_time >= ?
                ORDER BY start_time DESC
                """
                
                df = pd.read_sql_query(query, conn, params=[cutoff_date])
                
                if not df.empty:
                    # Parse timestamps
                    df['start_time'] = pd.to_datetime(df['start_time'])
                    df['validation_termination_timestamp'] = pd.to_datetime(df['validation_termination_timestamp'])
                
                return df
                
        except Exception as e:
            print(f"❌ Error querying database: {e}")
            return pd.DataFrame()
    
    def get_database_save_failures(self, days_back: int = 7) -> pd.DataFrame:
        """Get information about simulations that should have results but might have save failures."""
        if not self.db_exists:
            print(f"❌ Database not found: {self.db_path}")
            return pd.DataFrame()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
                
                # Get all recent runs with their save status
                query = """
                SELECT 
                    run_id, game_id, season_year, start_time, end_time, status,
                    duration_seconds, total_predictions, successful_predictions,
                    error_message, termination_reason
                FROM simulation_runs 
                WHERE start_time >= ?
                ORDER BY start_time DESC
                """
                
                df = pd.read_sql_query(query, conn, params=[cutoff_date])
                
                if not df.empty:
                    df['start_time'] = pd.to_datetime(df['start_time'])
                    df['end_time'] = pd.to_datetime(df['end_time'])
                
                return df
                
        except Exception as e:
            print(f"❌ Error querying database: {e}")
            return pd.DataFrame()
    
    def show_recent_database_errors(self, days_back: int = 7, max_errors: int = 50):
        """Display recent database write errors."""
        print(f"\n🔥 DATABASE WRITE ERRORS (Last {days_back} days)")
        print("=" * 80)
        
        error_logs = self.get_database_error_logs(days_back)
        
        if not error_logs:
            print("✅ No database errors found in the specified time period")
            return
        
        # Show summary
        error_types = {}
        for log in error_logs:
            error_type = log.get('error_type', 'unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        print(f"📊 Error Summary:")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"   {error_type}: {count} occurrences")
        
        # Show recent errors
        print(f"\n📋 Recent Errors (showing up to {max_errors}):")
        print("-" * 80)
        
        for i, error in enumerate(error_logs[:max_errors]):
            timestamp = error.get('timestamp', 'N/A')
            level = error.get('level', 'INFO')
            error_type = error.get('error_type', 'unknown')
            game_id = error.get('game_id', 'N/A')
            thread_id = error.get('thread_id', 'N/A')
            
            print(f"{i+1:2d}. [{timestamp}] {level} - Thread {thread_id}")
            print(f"    Type: {error_type} | Game: {game_id}")
            print(f"    Message: {error.get('message', '')[:100]}...")
            print()
    
    def show_validation_terminations(self, days_back: int = 7):
        """Display recent validation terminations."""
        print(f"\n🛑 VALIDATION TERMINATIONS (Last {days_back} days)")
        print("=" * 80)
        
        df = self.get_validation_terminations(days_back)
        
        if df.empty:
            print("✅ No validation terminations found in the specified time period")
            return
        
        # Summary by termination type
        termination_types = df['validation_termination_type'].value_counts()
        print(f"📊 Termination Summary ({len(df)} total):")
        for term_type, count in termination_types.items():
            print(f"   {term_type}: {count} occurrences")
        
        # Summary by trigger condition
        trigger_conditions = df['validation_trigger_condition'].value_counts()
        print(f"\n🎯 Trigger Conditions:")
        for trigger, count in trigger_conditions.items():
            print(f"   {trigger}: {count} occurrences")
        
        # Show recent terminations
        print(f"\n📋 Recent Terminations:")
        print("-" * 80)
        
        for i, (_, row) in enumerate(df.head(20).iterrows()):
            timestamp = row['validation_termination_timestamp']
            term_type = row['validation_termination_type']
            trigger = row['validation_trigger_condition']
            game_id = row['game_id']
            quarter = row['validation_game_state_quarter']
            time_remaining = row['validation_game_state_time']
            score = row['validation_game_state_score']
            
            print(f"{i+1:2d}. [{timestamp}] {term_type}")
            print(f"    Game: {game_id} | Trigger: {trigger}")
            print(f"    Game State: Q{quarter} {time_remaining} - {score}")
            print()
    
    def analyze_patterns(self, days_back: int = 7):
        """Analyze patterns in errors and terminations."""
        print(f"\n📈 ERROR PATTERN ANALYSIS (Last {days_back} days)")
        print("=" * 80)
        
        # Get data
        error_logs = self.get_database_error_logs(days_back)
        termination_df = self.get_validation_terminations(days_back)
        save_failures_df = self.get_database_save_failures(days_back)
        
        # Analyze database errors by game
        if error_logs:
            print("🎮 Database Errors by Game:")
            game_errors = {}
            for error in error_logs:
                game_id = error.get('game_id', 'unknown')
                game_errors[game_id] = game_errors.get(game_id, 0) + 1
            
            for game_id, count in sorted(game_errors.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"   {game_id}: {count} errors")
        
        # Analyze validation terminations by game
        if not termination_df.empty:
            print(f"\n🛑 Validation Terminations by Game:")
            game_terminations = termination_df['game_id'].value_counts()
            for game_id, count in game_terminations.head(10).items():
                print(f"   {game_id}: {count} terminations")
        
        # Analyze success rates
        if not save_failures_df.empty:
            print(f"\n📊 Simulation Status Distribution:")
            status_counts = save_failures_df['status'].value_counts()
            total = len(save_failures_df)
            for status, count in status_counts.items():
                percentage = (count / total) * 100
                print(f"   {status}: {count} ({percentage:.1f}%)")
    
    def analyze_validation_failures(self, days_back: int = 7) -> pd.DataFrame:
        """Analyze validation failures (retry exhaustion) from recent simulations."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get validation failures from recent days
                cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
                
                query = """
                SELECT 
                    run_id, game_id, season_year, start_time, status, 
                    termination_reason, error_message, duration_seconds,
                    validation_failure_timestamp, validation_total_failed_attempts,
                    validation_most_common_reason, validation_most_common_reason_count,
                    validation_most_common_error_type, validation_most_common_error_type_count,
                    validation_most_common_field, validation_most_common_field_count,
                    validation_unique_reasons, validation_unique_error_types, validation_unique_fields,
                    validation_failure_summary, validation_response_examples
                FROM simulation_runs 
                WHERE validation_failure_timestamp IS NOT NULL
                  AND start_time >= ?
                ORDER BY start_time DESC
                """
                
                df = pd.read_sql_query(query, conn, params=[cutoff_date])
                
                if not df.empty:
                    # Parse timestamps
                    df['start_time'] = pd.to_datetime(df['start_time'])
                    df['validation_failure_timestamp'] = pd.to_datetime(df['validation_failure_timestamp'])
                    
                    print(f"📊 VALIDATION FAILURE ANALYSIS (Last {days_back} days)")
                    print("=" * 70)
                    print(f"Total simulations with validation failures: {len(df)}")
                    print()
                    
                    # Summary statistics
                    print("🔍 FAILURE STATISTICS:")
                    print(f"   Average failed attempts per simulation: {df['validation_total_failed_attempts'].mean():.1f}")
                    print(f"   Max failed attempts: {df['validation_total_failed_attempts'].max()}")
                    print(f"   Average unique failure reasons: {df['validation_unique_reasons'].mean():.1f}")
                    print(f"   Average unique error types: {df['validation_unique_error_types'].mean():.1f}")
                    print(f"   Average unique problematic fields: {df['validation_unique_fields'].mean():.1f}")
                    print()
                    
                    # Most common failure reasons
                    print("🚫 MOST COMMON FAILURE REASONS:")
                    reason_counts = df['validation_most_common_reason'].value_counts()
                    for reason, count in reason_counts.head(10).items():
                        if reason and str(reason) != 'nan':
                            print(f"   {reason}: {count} simulations")
                    print()
                    
                    # Most common error types
                    print("⚠️ MOST COMMON ERROR TYPES:")
                    error_type_counts = df['validation_most_common_error_type'].value_counts()
                    for error_type, count in error_type_counts.head(10).items():
                        if error_type and str(error_type) != 'nan':
                            print(f"   {error_type}: {count} simulations")
                    print()
                    
                    # Most problematic fields
                    print("🎯 MOST PROBLEMATIC FIELDS:")
                    field_counts = df['validation_most_common_field'].value_counts()
                    for field, count in field_counts.head(10).items():
                        if field and str(field) != 'nan':
                            print(f"   {field}: {count} simulations")
                    print()
                    
                    # Game-specific analysis
                    print("🏀 FAILURES BY GAME:")
                    game_failures = df.groupby('game_id').agg({
                        'run_id': 'count',
                        'validation_total_failed_attempts': 'mean',
                        'validation_most_common_reason': lambda x: x.mode().iloc[0] if not x.empty else 'N/A'
                    }).rename(columns={'run_id': 'failure_count', 'validation_total_failed_attempts': 'avg_failed_attempts', 'validation_most_common_reason': 'common_reason'})
                    
                    for game_id, row in game_failures.head(10).iterrows():
                        print(f"   Game {game_id}: {row['failure_count']} failures, avg {row['avg_failed_attempts']:.1f} attempts, common: {row['common_reason']}")
                    print()
                    
                    # Show detailed examples
                    print("💡 EXAMPLE VALIDATION FAILURES:")
                    for idx, row in df.head(3).iterrows():
                        print(f"   {row['start_time']:%Y-%m-%d %H:%M} - Game {row['game_id']}")
                        print(f"      Failed attempts: {row['validation_total_failed_attempts']}")
                        print(f"      Most common reason: {row['validation_most_common_reason']}")
                        print(f"      Most common error: {row['validation_most_common_error_type']}")
                        print(f"      Duration: {row['duration_seconds']:.1f}s")
                        # Show example response if available
                        if row['validation_response_examples']:
                            try:
                                examples = json.loads(row['validation_response_examples'])
                                if examples and len(examples) > 0:
                                    print(f"      Example response: {examples[0][:100]}...")
                            except:
                                pass
                        print()
                
                else:
                    print(f"📊 No validation failures found in the last {days_back} days")
                    print("✅ This suggests the validation logic is working correctly!")
                
                return df
                
        except Exception as e:
            print(f"Error analyzing validation failures: {e}")
            return None
    
    def export_error_report(self, days_back: int = 7, output_file: str = "error_report.xlsx"):
        """Export comprehensive error report to Excel."""
        print(f"\n📄 EXPORTING ERROR REPORT")
        print("=" * 80)
        
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Database errors
                error_logs = self.get_database_error_logs(days_back)
                if error_logs:
                    error_df = pd.DataFrame(error_logs)
                    error_df.to_excel(writer, sheet_name='Database Errors', index=False)
                    print(f"✅ Database errors exported: {len(error_logs)} records")
                
                # Validation terminations
                termination_df = self.get_validation_terminations(days_back)
                if not termination_df.empty:
                    termination_df.to_excel(writer, sheet_name='Validation Terminations', index=False)
                    print(f"✅ Validation terminations exported: {len(termination_df)} records")
                
                # Validation failures
                failure_df = self.analyze_validation_failures(days_back)
                if failure_df is not None and not failure_df.empty:
                    failure_df.to_excel(writer, sheet_name='Validation Failures', index=False)
                    print(f"✅ Validation failures exported: {len(failure_df)} records")
                
                # All simulation runs
                runs_df = self.get_database_save_failures(days_back)
                if not runs_df.empty:
                    runs_df.to_excel(writer, sheet_name='All Simulation Runs', index=False)
                    print(f"✅ All simulation runs exported: {len(runs_df)} records")
            
            print(f"📁 Report saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Error exporting report: {e}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='NBA Simulation Error Analysis Tool')
    parser.add_argument('--db', type=str, default='enhanced_simulation_results.db',
                       help='Path to simulation results database')
    parser.add_argument('--days', type=int, default=7,
                       help='Number of days back to analyze (default: 7)')
    parser.add_argument('--max-errors', type=int, default=50,
                       help='Maximum number of errors to display (default: 50)')
    parser.add_argument('--export', type=str,
                       help='Export comprehensive report to Excel file')
    parser.add_argument('--database-errors-only', action='store_true',
                       help='Show only database write errors')
    parser.add_argument('--validation-only', action='store_true',
                      help='Show only validation terminations')
    parser.add_argument('--validation-failures', action='store_true',
                      help='Show only validation failure analysis')
    parser.add_argument('--patterns-only', action='store_true',
                       help='Show only pattern analysis')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = SimulationErrorAnalyzer(args.db)
    
    print("🔍 NBA SIMULATION ERROR ANALYZER")
    print("=" * 80)
    print(f"Database: {args.db}")
    print(f"Analysis period: Last {args.days} days")
    print(f"Time range: {datetime.now() - timedelta(days=args.days)} to {datetime.now()}")
    
    # Show requested analysis
    if args.export:
        analyzer.export_error_report(args.days, args.export)
    
    elif args.database_errors_only:
        analyzer.show_recent_database_errors(args.days, args.max_errors)
    
    elif args.validation_only:
        analyzer.show_validation_terminations(args.days)
    
    elif args.validation_failures:
        analyzer.analyze_validation_failures(args.days)
    
    elif args.patterns_only:
        analyzer.analyze_patterns(args.days)
    
    else:
        # Show everything
        analyzer.show_recent_database_errors(args.days, args.max_errors)
        analyzer.show_validation_terminations(args.days)
        analyzer.analyze_validation_failures(args.days)
        analyzer.analyze_patterns(args.days)


if __name__ == "__main__":
    main()
