import json
import pandas as pd

print("="*80)
print("25-26 SEASON DATA DIAGNOSIS")
print("="*80)
print()

# Load games
with open('data/games_2025_2026_current_norm.jsonl') as f:
    games = [json.loads(line) for line in f]

print(f"Total games: {len(games)}")
print()

# Check for player data
has_players = sum(1 for g in games if 'players' in g and g['players'])
print(f"Games with player data: {has_players}")
print(f"Games WITHOUT player data: {len(games) - has_players}")
print()

if has_players == 0:
    print("⚠️  NO PLAYER DATA FOUND IN 25-26 SEASON!")
    print("This would cause major feature issues:")
    print("  - star_out defaults to 0")
    print("  - minutes_missing_top2 defaults to 0")
    print("  - usage_share_top2 defaults to baseline")
    print("  - team_weighted_ts defaults to baseline")
    print()

# Check team stats
print("Checking team statistics...")
game1 = games[0]
home_team = game1['teams']['H']
away_team = game1['teams']['A']

print(f"\nSample game: {game1.get('date')}")
print(f"Home team stats keys: {list(home_team.keys())[:10]}")
print()

# Check normalized features
norm_features = [k for k in home_team.keys() if '_norm' in k]
print(f"Normalized features present: {len(norm_features)}")
if norm_features:
    print(f"Examples: {norm_features[:5]}")
    print()
    print("Sample values:")
    for feat in norm_features[:5]:
        print(f"  {feat}: Home={home_team.get(feat)}, Away={away_team.get(feat)}")
else:
    print("⚠️  NO NORMALIZED FEATURES!")

print()
print("="*80)

