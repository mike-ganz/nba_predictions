#!/usr/bin/env python3
"""
Check error details from the multithreaded database
"""
import sqlite3

def main():
    conn = sqlite3.connect('enhanced_simulation_results_multithreaded.db')
    cursor = conn.cursor()
    
    # Get error details
    cursor.execute('''
        SELECT run_id, error_message, start_time, end_time, duration_seconds
        FROM simulation_runs 
        WHERE status = "error"
        ORDER BY start_time
    ''')
    
    errors = cursor.fetchall()
    
    print("="*80)
    print("ERROR ANALYSIS")
    print("="*80)
    print(f"Found {len(errors)} error runs")
    print()
    
    for i, (run_id, error_msg, start_time, end_time, duration) in enumerate(errors):
        print(f"ERROR {i+1}:")
        print(f"  Run ID: {run_id}")
        print(f"  Duration: {duration} seconds")
        print(f"  Time: {start_time} to {end_time}")
        print(f"  Error: {error_msg}")
        print("-" * 60)
    
    # Also check if there are any other status types
    cursor.execute('SELECT status, COUNT(*) FROM simulation_runs GROUP BY status')
    status_counts = cursor.fetchall()
    
    print("\nSTATUS SUMMARY:")
    for status, count in status_counts:
        print(f"  {status}: {count}")
    
    conn.close()

if __name__ == "__main__":
    main()
