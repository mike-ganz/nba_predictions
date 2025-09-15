#!/usr/bin/env python3
"""
Check the new multithreaded simulation database
"""
import sqlite3

def main():
    conn = sqlite3.connect('enhanced_simulation_results_multithreaded.db')
    cursor = conn.cursor()
    
    # Get all simulation results
    cursor.execute('''
        SELECT run_id, game_id, final_score, status, 
               successful_predictions, total_predictions
        FROM simulation_runs 
        ORDER BY start_time
    ''')
    
    results = cursor.fetchall()
    
    print("="*80)
    print("MULTITHREADED SIMULATION RESULTS")
    print("="*80)
    print(f"Total simulations: {len(results)}")
    print()
    
    if results:
        print(f"{'#':<3} {'Game ID':<12} {'Final Score':<25} {'Status':<12} {'Predictions'}")
        print("-" * 80)
        
        for i, (run_id, game_id, final_score, status, success_pred, total_pred) in enumerate(results):
            score_display = final_score if final_score else "In Progress"
            pred_display = f"{success_pred or 0}/{total_pred or 0}"
            print(f"{i+1:<3} {game_id:<12} {score_display:<25} {status:<12} {pred_display}")
    
    # Get unique games
    cursor.execute('SELECT DISTINCT game_id FROM simulation_runs')
    unique_games = [row[0] for row in cursor.fetchall()]
    
    print(f"\nUnique games: {len(unique_games)}")
    if unique_games:
        print(f"Games: {', '.join(unique_games)}")
    
    conn.close()

if __name__ == "__main__":
    main()
