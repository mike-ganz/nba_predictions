import sqlite3

# Connect to the downloaded database
conn = sqlite3.connect('enhanced_simulation_results_current.db')
cursor = conn.cursor()

# Check simulation progress
cursor.execute('SELECT COUNT(*) FROM simulation_runs WHERE status = "game_ended"')
completed = cursor.fetchone()[0]

cursor.execute('SELECT COUNT(DISTINCT game_id) FROM simulation_runs WHERE status = "game_ended"')
games = cursor.fetchone()[0]

cursor.execute('SELECT COUNT(*) FROM simulation_runs')
total_runs = cursor.fetchone()[0]

print(f"📊 CURRENT SIMULATION PROGRESS:")
print(f"   Completed simulations: {completed}")
print(f"   Games processed: {games}")
print(f"   Total simulation records: {total_runs}")
print(f"   Success rate: {(completed/total_runs*100):.1f}%" if total_runs > 0 else "   Success rate: N/A")

conn.close()
