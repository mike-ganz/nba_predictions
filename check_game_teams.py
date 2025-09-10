import sqlite3

conn = sqlite3.connect('enhanced_simulation_results_latest.db')
cursor = conn.cursor()

# Get all unique game IDs and their contexts
print("🏀 GAME ID TO TEAM MAPPINGS:")
print("=" * 60)

# First, let's see what data is available in simulation_runs
cursor.execute("SELECT DISTINCT game_id FROM simulation_runs ORDER BY game_id LIMIT 10")
game_ids = [row[0] for row in cursor.fetchall()]

print(f"Found {len(game_ids)} unique game IDs in database")
print("\nFirst 10 game IDs:")
for game_id in game_ids:
    print(f"  - {game_id}")

# Check if we have any context or team information
cursor.execute("PRAGMA table_info(simulation_runs)")
columns = cursor.fetchall()
print(f"\n📊 Available columns in simulation_runs:")
for col in columns:
    print(f"  - {col[1]} ({col[2]})")

# Check if actual_game_results table exists and has team data
try:
    cursor.execute("SELECT game_id, away_team, home_team FROM actual_game_results WHERE game_id IN (22400530, 22400531, 22400532, 22400533, 22400534) ORDER BY game_id")
    actual_games = cursor.fetchall()
    
    if actual_games:
        print(f"\n🎯 TEAM MAPPINGS FROM ACTUAL GAME RESULTS:")
        print("-" * 50)
        for game_id, away_team, home_team in actual_games:
            print(f"  {game_id}: {away_team} @ {home_team}")
    else:
        print(f"\n❌ No team data found in actual_game_results table")
        
except sqlite3.OperationalError:
    print(f"\n❌ actual_game_results table not found in this database")

conn.close()
