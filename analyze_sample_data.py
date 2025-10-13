"""
Comprehensive analysis of sample training data
"""

import json
from collections import Counter, defaultdict

print("COMPREHENSIVE ANALYSIS OF SAMPLE TRAINING DATA")
print("=" * 70)

# Load data
examples = []
with open("sample_training_data.jsonl", 'r') as f:
    for line in f:
        examples.append(json.loads(line))

print(f"\nTotal examples: {len(examples)}")

# Analysis metrics
play_lengths = []
shot_zones = []
assist_tracking = {'total_made_baskets': 0, 'assisted': 0, 'unassisted': 0}
margin_stats = []
foul_counts = []
event_types = Counter()
games_analyzed = set()

# Validation errors
errors = []

for idx, example in enumerate(examples):
    games_analyzed.add(example.get('A', 'UNK') + ' @ ' + example.get('H', 'UNK'))
    
    # Analyze plays
    if 'p' in example:
        for play_idx, play in enumerate(example['p']):
            # Check length
            play_lengths.append(len(play))
            
            if len(play) != 10:
                errors.append(f"Example {idx}, Play {play_idx}: Wrong length ({len(play)} vs 10)")
                continue
            
            # Extract fields
            quarter = play[0]
            time = play[1]
            score = play[2]
            margin = play[3]
            actor = play[4]
            actor_fouls = play[5]
            event = play[6]
            shot_zone = play[7]
            assist_by = play[8]
            lineup_id = play[9]
            
            # Validate margin calculation
            if isinstance(score, list) and len(score) == 2:
                expected_margin = score[0] - score[1]
                if margin != expected_margin:
                    errors.append(f"Ex {idx}, Play {play_idx}: Margin mismatch ({margin} vs {expected_margin})")
            
            # Track event types
            event_types[event] += 1
            
            # Track shot zones
            if event in ['made2', 'miss2', 'made3', 'miss3']:
                if shot_zone:
                    shot_zones.append(shot_zone)
                else:
                    errors.append(f"Ex {idx}, Play {play_idx}: Missing shot zone for {event}")
            
            # Track assists
            if event in ['made2', 'made3']:
                assist_tracking['total_made_baskets'] += 1
                if assist_by:
                    assist_tracking['assisted'] += 1
                else:
                    assist_tracking['unassisted'] += 1
            
            # Track margins
            margin_stats.append(margin)
            
            # Track fouls
            foul_counts.append(actor_fouls)
            
            # Validate actor fouls match player array
            if isinstance(actor, list) and len(actor) == 2 and actor[1] >= 0:
                team = actor[0]
                idx_player = actor[1]
                
                # Check if fouls match
                player_array_key = 'ap' if team == 'A' else 'hp'
                if player_array_key in example and idx_player < len(example[player_array_key]):
                    player_array = example[player_array_key][idx_player]
                    if len(player_array) >= 14:
                        expected_fouls = player_array[13]
                        if actor_fouls != expected_fouls:
                            errors.append(f"Ex {idx}, Play {play_idx}: Foul mismatch ({actor_fouls} vs {expected_fouls})")

# Print analysis
print(f"\nGames analyzed: {len(games_analyzed)}")
for game in sorted(games_analyzed):
    print(f"  - {game}")

print(f"\n" + "=" * 70)
print("PLAY FORMAT VALIDATION")
print("=" * 70)

# Play length distribution
length_counter = Counter(play_lengths)
print(f"\nPlay array lengths:")
for length, count in sorted(length_counter.items()):
    pct = count / len(play_lengths) * 100
    status = "✅" if length == 10 else "❌"
    print(f"  {status} {length} values: {count:,} plays ({pct:.1f}%)")

print(f"\n" + "=" * 70)
print("SHOT ZONE ANALYSIS")
print("=" * 70)

zone_counter = Counter(shot_zones)
print(f"\nShot zones classified ({len(shot_zones)} shooting events):")
for zone, count in zone_counter.most_common():
    pct = count / len(shot_zones) * 100
    print(f"  '{zone}': {count:,} ({pct:.1f}%)")

print(f"\n" + "=" * 70)
print("ASSIST TRACKING ANALYSIS")
print("=" * 70)

print(f"\nMade baskets: {assist_tracking['total_made_baskets']}")
if assist_tracking['total_made_baskets'] > 0:
    assisted_pct = assist_tracking['assisted'] / assist_tracking['total_made_baskets'] * 100
    print(f"  Assisted: {assist_tracking['assisted']} ({assisted_pct:.1f}%)")
    print(f"  Unassisted: {assist_tracking['unassisted']} ({100-assisted_pct:.1f}%)")

print(f"\n" + "=" * 70)
print("MARGIN ANALYSIS")
print("=" * 70)

print(f"\nMargin statistics ({len(margin_stats)} plays):")
print(f"  Min: {min(margin_stats)}")
print(f"  Max: {max(margin_stats)}")
print(f"  Close games (|margin| <= 5): {sum(1 for m in margin_stats if abs(m) <= 5)} ({sum(1 for m in margin_stats if abs(m) <= 5)/len(margin_stats)*100:.1f}%)")
print(f"  Tied (margin = 0): {sum(1 for m in margin_stats if m == 0)} ({sum(1 for m in margin_stats if m == 0)/len(margin_stats)*100:.1f}%)")

print(f"\n" + "=" * 70)
print("FOUL COUNT ANALYSIS")
print("=" * 70)

foul_counter = Counter(foul_counts)
print(f"\nFoul distribution ({len(foul_counts)} plays):")
for fouls, count in sorted(foul_counter.items()):
    pct = count / len(foul_counts) * 100
    status = "⚠️" if fouls >= 5 else ""
    print(f"  {fouls} fouls: {count:,} ({pct:.1f}%) {status}")

print(f"\n" + "=" * 70)
print("EVENT TYPE DISTRIBUTION")
print("=" * 70)

print(f"\nTop 15 event types:")
for event, count in event_types.most_common(15):
    pct = count / sum(event_types.values()) * 100
    print(f"  {event}: {count:,} ({pct:.1f}%)")

print(f"\n" + "=" * 70)
print("ERROR SUMMARY")
print("=" * 70)

if errors:
    print(f"\n❌ Found {len(errors)} errors:")
    for error in errors[:10]:  # Show first 10
        print(f"  - {error}")
    if len(errors) > 10:
        print(f"  ... and {len(errors) - 10} more")
else:
    print("\n✅ NO ERRORS FOUND!")

print(f"\n" + "=" * 70)
print("SAMPLE EXAMPLES")
print("=" * 70)

# Show 2 complete examples
for i in [0, len(examples)//2]:
    print(f"\nExample {i+1}:")
    ex = examples[i]
    print(f"  Game: {ex.get('A', '?')} @ {ex.get('H', '?')}")
    print(f"  Team stats: {len(ex.get('as', []))} values (away), {len(ex.get('hs', []))} values (home)")
    print(f"  Players: {len(ex.get('ap', []))} away, {len(ex.get('hp', []))} home")
    
    if 'p' in ex and len(ex['p']) > 0:
        print(f"  Plays: {len(ex['p'])} total")
        print(f"\n  Last 3 plays:")
        for j, play in enumerate(ex['p'][-3:]):
            print(f"    {j+1}. {play}")
            if len(play) == 10:
                print(f"       Q{play[0]}, {play[1]}s, score {play[2]}, margin {play[3]}")
                print(f"       Actor: {play[4]}, fouls: {play[5]}, event: {play[6]}")
                print(f"       Shot zone: {play[7]}, Assist: {play[8]}, Lineup: {play[9]}")

print(f"\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

summary_status = []
summary_status.append(("Play format", "10 values", all(l == 10 for l in play_lengths)))
summary_status.append(("Shot zones", f"{len(shot_zones)} classified", len(shot_zones) > 0))
summary_status.append(("Assist tracking", f"{assist_tracking['assisted']} assisted", True))
summary_status.append(("Margin calculation", "Validated", len([e for e in errors if 'Margin mismatch' in e]) == 0))
summary_status.append(("Actor fouls", "Validated", len([e for e in errors if 'Foul mismatch' in e]) == 0))

print("\n")
for item, value, status in summary_status:
    icon = "✅" if status else "❌"
    print(f"  {icon} {item}: {value}")

if len(errors) == 0:
    print(f"\n🎉 ALL VALIDATION CHECKS PASSED!")
    print(f"   10-value format is working correctly across {len(examples)} examples")
else:
    print(f"\n⚠️  Found {len(errors)} validation errors - needs investigation")

print(f"\n" + "=" * 70)

