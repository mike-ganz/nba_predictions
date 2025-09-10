import sqlite3

conn = sqlite3.connect('minimal_test.db')
cursor = conn.cursor()

# Check simulation results
cursor.execute('SELECT * FROM simulation_runs')
row = cursor.fetchone()

if row:
    print("🎯 MINIMAL TEST RESULTS:")
    print(f"   Game ID: {row[1]}")
    print(f"   Status: {row[7]}")  
    print(f"   Final Score: {row[10]}")
    print(f"   Planned Iterations: 20")
    print(f"   Actual Predictions: {row[9] if row[9] else 'N/A'}")
    print(f"   Success: {'YES' if row[7] == 'game_ended' else 'NO'}")
    print(f"   Duration: {row[6] if row[6] else 'N/A'} seconds")
    
    if row[10]:  # final_score
        print(f"\n🏀 GAME RESULT:")
        print(f"   Final Score: {row[10]}")
        print(f"   Quarter: {row[11] if len(row) > 11 else 'N/A'}")
        print(f"   Time: {row[12] if len(row) > 12 else 'N/A'}")
else:
    print("❌ No simulation results found")

conn.close()
