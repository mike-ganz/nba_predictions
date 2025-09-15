#!/usr/bin/env python3
"""
Check team names in various database files to identify the issue
"""
import sqlite3
import os

def check_database(db_path):
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return
    
    print(f"\n🔍 Checking: {db_path}")
    print("-" * 50)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get sample final scores
        cursor.execute('''
            SELECT game_id, final_score, status 
            FROM simulation_runs 
            WHERE final_score IS NOT NULL 
            LIMIT 10
        ''')
        
        results = cursor.fetchall()
        
        if results:
            for game_id, final_score, status in results:
                print(f"  Game {game_id}: {final_score} ({status})")
        else:
            print("  No final scores found")
            
        conn.close()
        
    except Exception as e:
        print(f"  Error reading database: {e}")

def main():
    print("="*60)
    print("TEAM NAME INVESTIGATION - CHECKING ALL DATABASES")
    print("="*60)
    
    # List of database files to check
    databases = [
        "team_name_debug_test.db",
        "enhanced_simulation_results.db",
        "simulation_results.db",
        "minimal_test.db"
    ]
    
    for db in databases:
        check_database(db)
    
    print(f"\n🎯 SUMMARY:")
    print("- Our test (game 22400542) showed correct team names: NOP vs BOS")
    print("- Check the samples above for any inconsistencies")
    print("- Look for patterns like 'ATL vs DET' when it should be something else")

if __name__ == "__main__":
    main()
