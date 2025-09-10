import sqlite3

conn = sqlite3.connect('minimal_test_v2.db')
cursor = conn.cursor()

cursor.execute('SELECT * FROM simulation_runs')
row = cursor.fetchone()

print("🎯 TEST V2 RESULTS:")
print(f"   Game ID: {row[1]}")
print(f"   Status: {row[7]}")  
print(f"   Final Score: {row[10]}")
print(f"   Success: {'YES' if row[7] == 'game_ended' else 'NO'}")
print(f"   Duration: {row[6] if row[6] else 'N/A'} seconds")

conn.close()
