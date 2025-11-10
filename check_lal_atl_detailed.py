"""Check detailed differences for LAL @ ATL between day-of and backlook."""
import json

# Load day-of data
print("Loading day-of data...")
day_of_games = []
with open('data/games_todays.jsonl', 'r') as f:
    for line in f:
        day_of_games.append(json.loads(line))

# Load backlook data
print("Loading backlook data...")
backlook_games = []
with open('data/games_2025_2026_current_new_norm.jsonl', 'r') as f:
    for line in f:
        backlook_games.append(json.loads(line))

# Find LAL @ ATL
day_lal_atl = next((g for g in day_of_games if g['game_id'] == '2025-11-08-LAL-ATL'), None)
back_lal_atl = next((g for g in backlook_games if g['game_id'] == '2025-11-08-LAL-ATL'), None)

if not day_lal_atl:
    print("ERROR: LAL @ ATL not found in day-of data")
    exit(1)

if not back_lal_atl:
    print("ERROR: LAL @ ATL not found in backlook data")
    exit(1)

print("\n" + "="*80)
print("LAL @ ATL DETAILED COMPARISON")
print("="*80)

# Market data
print("\nMarket Data:")
print(f"  Day-of:")
print(f"    Spread: {day_lal_atl['market']['spread_home']}")
print(f"    Total: {day_lal_atl['market'].get('total', 'N/A')}")
print(f"    ML Home: {day_lal_atl['market'].get('moneyline_home', 'N/A')}")
print(f"    ML Away: {day_lal_atl['market'].get('moneyline_away', 'N/A')}")

print(f"\n  Backlook:")
print(f"    Spread: {back_lal_atl['market']['spread_home']}")
print(f"    Total: {back_lal_atl['market'].get('total', 'N/A')}")
print(f"    ML Home: {back_lal_atl['market'].get('moneyline_home', 'N/A')}")
print(f"    ML Away: {back_lal_atl['market'].get('moneyline_away', 'N/A')}")

# Player data
print("\nPlayer Data:")

day_away = day_lal_atl.get('away_players', [])
day_home = day_lal_atl.get('home_players', [])
back_away = back_lal_atl.get('players', {}).get('A', [])
back_home = back_lal_atl.get('players', {}).get('H', [])

print(f"\n  LAL (Away):")
print(f"    Day-of: {len(day_away)} players")
day_away_inj = [p for p in day_away if p.get('projected_minutes') == 0.0]
print(f"    Day-of injured: {len(day_away_inj)}")
if day_away_inj:
    print(f"      Names: {[p['player_name'] for p in day_away_inj]}")
    print(f"      Baseline minutes: {[p['baseline_minutes'] for p in day_away_inj]}")

print(f"    Backlook: {len(back_away)} players")
back_away_inj = [p for p in back_away if p.get('projected_minutes') == 0.0]
print(f"    Backlook injured: {len(back_away_inj)}")
if back_away_inj:
    print(f"      Names: {[p['player_name'] for p in back_away_inj]}")
    print(f"      Baseline minutes: {[p['baseline_minutes'] for p in back_away_inj]}")

print(f"\n  ATL (Home):")
print(f"    Day-of: {len(day_home)} players")
day_home_inj = [p for p in day_home if p.get('projected_minutes') == 0.0]
print(f"    Day-of injured: {len(day_home_inj)}")
if day_home_inj:
    print(f"      Names: {[p['player_name'] for p in day_home_inj]}")
    print(f"      Baseline minutes: {[p['baseline_minutes'] for p in day_home_inj]}")

print(f"    Backlook: {len(back_home)} players")
back_home_inj = [p for p in back_home if p.get('projected_minutes') == 0.0]
print(f"    Backlook injured: {len(back_home_inj)}")
if back_home_inj:
    print(f"      Names: {[p['player_name'] for p in back_home_inj]}")
    print(f"      Baseline minutes: {[p['baseline_minutes'] for p in back_home_inj]}")

# Compute injury features
def compute_injury_features(players):
    sorted_players = sorted(players, key=lambda p: p.get('baseline_minutes', 0), reverse=True)
    
    minutes_missing_top2 = 0.0
    if len(sorted_players) >= 2:
        for p in sorted_players[:2]:
            if p.get('projected_minutes') == 0.0:
                minutes_missing_top2 += p.get('baseline_minutes', 0)
    
    star_out = 0
    for p in sorted_players:
        if p.get('projected_minutes') == 0.0 and p.get('baseline_usage_rate', 0) >= 0.27:
            star_out = 1
            break
    
    return minutes_missing_top2, star_out

day_away_mins, day_away_star = compute_injury_features(day_away)
day_home_mins, day_home_star = compute_injury_features(day_home)
back_away_mins, back_away_star = compute_injury_features(back_away)
back_home_mins, back_home_star = compute_injury_features(back_home)

print("\n" + "="*80)
print("INJURY FEATURES COMPARISON")
print("="*80)

print(f"\nLAL (Away):")
print(f"  Day-of:   minutes_missing_top2={day_away_mins:.1f}, star_out={day_away_star}")
print(f"  Backlook: minutes_missing_top2={back_away_mins:.1f}, star_out={back_away_star}")
print(f"  Match: {'YES' if day_away_mins == back_away_mins and day_away_star == back_away_star else 'NO'}")

print(f"\nATL (Home):")
print(f"  Day-of:   minutes_missing_top2={day_home_mins:.1f}, star_out={day_home_star}")
print(f"  Backlook: minutes_missing_top2={back_home_mins:.1f}, star_out={back_home_star}")
print(f"  Match: {'YES' if day_home_mins == back_home_mins and day_home_star == back_home_star else 'NO'}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if (day_away_mins == back_away_mins and day_away_star == back_away_star and
    day_home_mins == back_home_mins and day_home_star == back_home_star):
    print("\nInjury features MATCH between day-of and backlook!")
    print("Remaining difference (0.74 pts) is likely due to:")
    print("  - Moneyline differences affecting implied win probabilities")
    print("  - Total differences")
    print("  - Minor numerical precision differences")
else:
    print("\nInjury features DO NOT MATCH")
    print("This explains the prediction difference")

