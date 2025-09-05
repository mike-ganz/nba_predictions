#!/usr/bin/env python3
"""
Simple script to view NBA simulation results from SQLite database.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from tabulate import tabulate
from collections import defaultdict

def view_game_summary(db_path: str = "enhanced_simulation_results.db"):
    """View a pretty table summary of results grouped by game ID."""
    
    if not Path(db_path).exists():
        print(f"❌ Database file not found: {db_path}")
        return
    
    print(f"🏀 NBA Simulation Results Summary")
    print("=" * 100)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get detailed game summary with all metrics
        query = """
        SELECT 
            game_id,
            season_year,
            COUNT(*) as total_sims,
            SUM(CASE WHEN status = 'game_ended' THEN 1 ELSE 0 END) as completed_sims,
            SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_sims,
            ROUND(AVG(successful_predictions), 1) as avg_predictions,
            ROUND(MIN(successful_predictions), 0) as min_predictions,
            ROUND(MAX(successful_predictions), 0) as max_predictions,
            ROUND(AVG(duration_seconds), 1) as avg_duration_sec,
            GROUP_CONCAT(DISTINCT final_score) as final_scores
        FROM simulation_runs 
        GROUP BY game_id, season_year
        ORDER BY game_id, season_year
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        if not results:
            print("⚠️ No simulation results found!")
            conn.close()
            return
        
        # Prepare table data
        headers = [
            "Game ID", 
            "Season", 
            "Total\nSims", 
            "Completed", 
            "Errors", 
            "Success\nRate %",
            "Avg\nPredictions", 
            "Prediction\nRange", 
            "Avg Duration\n(min)",
            "Final Scores"
        ]
        
        table_data = []
        total_sims = 0
        total_completed = 0
        total_errors = 0
        
        for row in results:
            game_id, season, sims, completed, errors, avg_pred, min_pred, max_pred, avg_dur, scores = row
            
            # Calculate success rate
            success_rate = round((completed / sims * 100), 1) if sims > 0 else 0
            
            # Format prediction range
            pred_range = f"{int(min_pred)}-{int(max_pred)}" if min_pred != max_pred else str(int(min_pred))
            
            # Format duration in minutes
            avg_dur_min = round(avg_dur / 60, 1) if avg_dur else 0
            
            # Format final scores (remove None values and clean up)
            if scores:
                score_list = [s for s in scores.split(',') if s and s != 'None']
                formatted_scores = '\n'.join(score_list) if score_list else "N/A"
            else:
                formatted_scores = "N/A"
            
            # Status indicators
            completed_str = f"✅ {completed}" if completed > 0 else f"❌ {completed}"
            error_str = f"⚠️ {errors}" if errors > 0 else str(errors)
            
            table_data.append([
                game_id,
                season,
                sims,
                completed_str,
                error_str,
                f"{success_rate}%",
                avg_pred or "N/A",
                pred_range if min_pred else "N/A",
                f"{avg_dur_min}m",
                formatted_scores
            ])
            
            total_sims += sims
            total_completed += completed
            total_errors += errors
        
        # Print the main table
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid", stralign="center"))
        
        # Print overall summary
        overall_success_rate = round((total_completed / total_sims * 100), 1) if total_sims > 0 else 0
        
        print(f"\n📊 Overall Summary:")
        print(f"   🎯 Total Simulations: {total_sims}")
        print(f"   ✅ Completed: {total_completed} ({overall_success_rate}%)")
        print(f"   ⚠️  Errors: {total_errors} ({round(total_errors/total_sims*100, 1)}%)")
        
        # Get most recent successful simulation
        cursor.execute("""
            SELECT game_id, final_score, successful_predictions, duration_seconds, created_at
            FROM simulation_runs 
            WHERE status = 'game_ended'
            ORDER BY created_at DESC 
            LIMIT 1
        """)
        
        recent = cursor.fetchone()
        if recent:
            game, score, preds, dur, created = recent
            print(f"   🏆 Latest Success: Game {game} - {score} ({preds} plays, {dur/60:.1f}min)")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error reading database: {e}")

def view_simulation_results(db_path: str = "enhanced_simulation_results.db"):
    """View simulation results from SQLite database."""
    
    if not Path(db_path).exists():
        print(f"❌ Database file not found: {db_path}")
        return
    
    print(f"📊 NBA Simulation Results from {db_path}")
    print("=" * 60)
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Check tables
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"📋 Tables: {[t[0] for t in tables]}")
        
        # Count total runs
        cursor.execute("SELECT COUNT(*) FROM simulation_runs")
        total_runs = cursor.fetchone()[0]
        print(f"🎯 Total simulation runs: {total_runs}")
        
        if total_runs == 0:
            print("\n⚠️ No simulation results found. Run some simulations first!")
            conn.close()
            return
        
        # Get summary statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as total_runs,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors,
                AVG(duration_seconds) as avg_duration,
                AVG(successful_predictions) as avg_predictions
            FROM simulation_runs
        """)
        
        summary = cursor.fetchone()
        if summary:
            print(f"\n📈 Summary Statistics:")
            print(f"   Total runs: {summary[0]}")
            print(f"   Completed: {summary[1]} ({summary[1]/summary[0]*100:.1f}%)")
            print(f"   Errors: {summary[2]} ({summary[2]/summary[0]*100:.1f}%)")
            print(f"   Avg duration: {summary[3]:.1f}s")
            print(f"   Avg predictions per run: {summary[4]:.1f}")
        
        # Get recent runs
        print(f"\n🏀 Recent Simulation Runs:")
        cursor.execute("""
            SELECT 
                run_id, game_id, season_year, status, 
                total_predictions, successful_predictions, 
                final_score, duration_seconds, 
                created_at
            FROM simulation_runs 
            ORDER BY created_at DESC 
            LIMIT 10
        """)
        
        runs = cursor.fetchall()
        if runs:
            for i, run in enumerate(runs, 1):
                run_id, game_id, season, status, total_pred, success_pred, final_score, duration, created = run
                success_rate = (success_pred/total_pred*100) if total_pred > 0 else 0
                print(f"   {i}. Game {game_id} ({season}) - {status}")
                print(f"      Run ID: {run_id}")
                print(f"      Predictions: {success_pred}/{total_pred} ({success_rate:.1f}%)")
                print(f"      Final Score: {final_score or 'N/A'}")
                print(f"      Duration: {duration:.1f}s | Created: {created}")
                print()
        
        # Game-wise summary
        print(f"📊 Results by Game:")
        cursor.execute("""
            SELECT 
                game_id, 
                COUNT(*) as runs,
                AVG(successful_predictions) as avg_predictions,
                AVG(duration_seconds) as avg_duration
            FROM simulation_runs 
            GROUP BY game_id 
            ORDER BY game_id
        """)
        
        game_summary = cursor.fetchall()
        if game_summary:
            for game_id, runs, avg_pred, avg_dur in game_summary:
                print(f"   Game {game_id}: {runs} runs, {avg_pred:.1f} avg predictions, {avg_dur:.1f}s avg duration")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error reading database: {e}")

def view_detailed_run(run_id: str, db_path: str = "enhanced_simulation_results.db"):
    """View detailed information for a specific run."""
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM simulation_runs WHERE run_id = ?
        """, (run_id,))
        
        result = cursor.fetchone()
        if not result:
            print(f"❌ Run {run_id} not found")
            conn.close()
            return
        
        # Get column names
        cursor.execute("PRAGMA table_info(simulation_runs)")
        columns = [col[1] for col in cursor.fetchall()]
        
        print(f"🔍 Detailed Results for Run: {run_id}")
        print("=" * 60)
        
        for i, col in enumerate(columns):
            value = result[i]
            if col == 'iterations_data' and value:
                print(f"{col}: [JSON data - {len(value)} characters]")
            else:
                print(f"{col}: {value}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error reading run details: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg == "--detailed" or arg == "-d":
            # View detailed results
            view_simulation_results()
        elif arg == "--help" or arg == "-h":
            print("🏀 NBA Simulation Results Viewer")
            print("Usage:")
            print("  python view_results.py           # Game summary table (default)")
            print("  python view_results.py -d       # Detailed view with all runs")
            print("  python view_results.py [RUN_ID] # View specific run details")
            print("  python view_results.py -h       # Show this help")
        elif len(arg) > 10:  # Looks like a run ID
            # View specific run
            view_detailed_run(arg)
        else:
            print(f"❌ Unknown option: {arg}")
            print("Use --help or -h for usage information")
    else:
        # Default: Show game summary table
        view_game_summary()
