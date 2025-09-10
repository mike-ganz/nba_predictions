import sqlite3

conn = sqlite3.connect('enhanced_simulation_results_latest.db')
cursor = conn.cursor()

# Check completed simulations
cursor.execute('SELECT COUNT(*) FROM simulation_runs WHERE status = "game_ended"')
completed = cursor.fetchone()[0]

# Check total games processed
cursor.execute('SELECT COUNT(DISTINCT game_id) FROM simulation_runs WHERE status = "game_ended"')
games = cursor.fetchone()[0]

# Check all simulation records
cursor.execute('SELECT COUNT(*) FROM simulation_runs')
total = cursor.fetchone()[0]

# Check recent games being processed
cursor.execute('SELECT DISTINCT game_id FROM simulation_runs ORDER BY created_at DESC LIMIT 10')
recent_games = [row[0] for row in cursor.fetchall()]

print(f"🎯 CURRENT PROGRESS:")
print(f"   Completed simulations: {completed}")
print(f"   Games processed: {games}")
print(f"   Total records: {total}")
print(f"   Success rate: {(completed/total*100):.1f}%" if total > 0 else "   Success rate: N/A")
print(f"\n📋 Recent games being processed:")
for game in recent_games[:5]:
    print(f"   - {game}")

conn.close()
