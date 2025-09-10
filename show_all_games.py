import sqlite3

conn = sqlite3.connect('enhanced_simulation_results_latest.db')
cursor = conn.cursor()

# Get all games from actual_game_results
cursor.execute("SELECT game_id, away_team, home_team FROM actual_game_results ORDER BY CAST(game_id AS INTEGER)")
games = cursor.fetchall()

print("🏀 COMPLETE GAME MAPPINGS:")
print("=" * 70)

for game_id, away_team, home_team in games:
    # Check if this game is in our config
    is_target = game_id in ['22400530', '22400531', '22400532', '22400533', '22400534', 
                           '22400535', '22400536', '22400537', '22400538', '22400539', 
                           '22400540', '22400541', '22400542', '22400543', '22400544',
                           '22400545', '22400546', '22400547', '22400548', '22400549',
                           '22400550', '22400553', '22400554', '22400555']
    
    marker = "🎯 TARGET" if is_target else "     "
    print(f"{marker} {game_id}: {away_team} @ {home_team}")

print(f"\n📊 Total games in database: {len(games)}")

# Check which target games we have team data for
target_games = ['22400530', '22400531', '22400532', '22400533', '22400534', 
                '22400535', '22400536', '22400537', '22400538', '22400539', 
                '22400540', '22400541', '22400542', '22400543', '22400544',
                '22400545', '22400546', '22400547', '22400548', '22400549',
                '22400550', '22400553', '22400554', '22400555']

found_targets = [game_id for game_id, _, _ in games if game_id in target_games]
print(f"📋 Target games with team data: {len(found_targets)}/{len(target_games)}")

conn.close()
