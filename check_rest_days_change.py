"""Check if rest days actually changed in the data."""

import json

# Load old data
old_games = []
with open('backups/pre_rest_days_fix_20251108_174325/games_2025_2026_current_norm.jsonl', 'r') as f:
    for line in f:
        game = json.loads(line)
        if game['date'] == '2025-11-07':
            old_games.append(game)

# Load new data
new_games = []
with open('data/games_2025_2026_current_norm.jsonl', 'r') as f:
    for line in f:
        game = json.loads(line)
        if game['date'] == '2025-11-07':
            new_games.append(game)

print("="*70)
print("DID REST DAYS ACTUALLY CHANGE IN THE DATA?")
print("="*70)
print()

changed = False
for old_game in old_games:
    game_id = old_game['game_id']
    new_game = [g for g in new_games if g['game_id'] == game_id][0]
    
    old_away_rest = old_game['teams']['A']['rest_days']
    new_away_rest = new_game['teams']['A']['rest_days']
    old_home_rest = old_game['teams']['H']['rest_days']
    new_home_rest = new_game['teams']['H']['rest_days']
    
    if old_away_rest != new_away_rest or old_home_rest != new_home_rest:
        print(f"{game_id}: CHANGED")
        print(f"  Away: {old_away_rest} -> {new_away_rest}")
        print(f"  Home: {old_home_rest} -> {new_home_rest}")
        changed = True

if not changed:
    print("NO CHANGES DETECTED!")
    print()
    print("The rest days in the 'new' data are IDENTICAL to the 'old' data.")
    print()
    print("Sample (showing they're the same):")
    for i in range(3):
        old_game = old_games[i]
        new_game = [g for g in new_games if g['game_id'] == old_game['game_id']][0]
        print(f"\n{old_game['game_id']}:")
        print(f"  OLD: Away={old_game['teams']['A']['rest_days']}, Home={old_game['teams']['H']['rest_days']}")
        print(f"  NEW: Away={new_game['teams']['A']['rest_days']}, Home={new_game['teams']['H']['rest_days']}")
    
    print()
    print("POSSIBLE EXPLANATION:")
    print("  The current season data may have ALREADY been regenerated with")
    print("  the fix during an earlier run of the script!")

print()

