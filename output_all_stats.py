"""
Output ALL calculated stats for each player for vetting
"""

from player_stats_from_pbp import calculate_player_pbp_stats
import json

# Test players
test_players = [
    "Karl-Anthony Towns",
    "Mitchell Robinson",
    "Trae Young",
    "Stephen Curry",
    "Shai Gilgeous-Alexander",
    "Luka Doncic",
    "Kristaps Porzingis"
]

season = "2023-2024"
max_date = "2024-01-01"

print("=" * 100)
print("COMPLETE STATS BREAKDOWN - ALL CALCULATED VALUES")
print("=" * 100)
print(f"Season: {season} | Date Range: Start through {max_date}")
print("=" * 100)

all_results = {}

for player in test_players:
    print(f"\n{'=' * 100}")
    print(f"PLAYER: {player}")
    print('=' * 100)
    
    stats = calculate_player_pbp_stats(player, max_date, season)
    
    if stats:
        all_results[player] = stats
        
        # Context first
        print(f"\n📊 CONTEXT METRICS")
        print(f"{'games_played':<30} {stats['games_played']:>10,}")
        print(f"{'total_possessions':<30} {stats['total_possessions']:>10,}")
        print(f"{'total_fga':<30} {stats['total_fga']:>10,}")
        print(f"{'season':<30} {stats['season']:>10}")
        print(f"{'max_date':<30} {stats['max_date']:>10}")
        
        # Shot Location Rates
        print(f"\n🎯 SHOT LOCATION RATES (as % of total FGA)")
        print(f"{'rim_attempt_rate':<30} {stats['rim_attempt_rate']:>10.6f}  ({stats['rim_attempt_rate']*100:>6.2f}%)")
        print(f"{'corner_3_rate':<30} {stats['corner_3_rate']:>10.6f}  ({stats['corner_3_rate']*100:>6.2f}%)")
        print(f"{'non_corner_3_rate':<30} {stats['non_corner_3_rate']:>10.6f}  ({stats['non_corner_3_rate']*100:>6.2f}%)")
        print(f"{'short_mid_rate':<30} {stats['short_mid_rate']:>10.6f}  ({stats['short_mid_rate']*100:>6.2f}%)")
        print(f"{'long_mid_rate':<30} {stats['long_mid_rate']:>10.6f}  ({stats['long_mid_rate']*100:>6.2f}%)")
        print(f"{'mid_range_rate':<30} {stats['mid_range_rate']:>10.6f}  ({stats['mid_range_rate']*100:>6.2f}%)")
        
        # Calculate total for validation
        total_shot_dist = (stats['rim_attempt_rate'] + stats['corner_3_rate'] + 
                          stats['non_corner_3_rate'] + stats['mid_range_rate'])
        print(f"{'[VALIDATION] Total':<30} {total_shot_dist:>10.6f}  ({total_shot_dist*100:>6.2f}%)")
        
        # Shot Type Breakdown
        print(f"\n🏀 SHOT TYPE BREAKDOWN (3-pointers)")
        print(f"{'pullup_3_rate':<30} {stats['pullup_3_rate']:>10.6f}  ({stats['pullup_3_rate']*100:>6.2f}%)")
        print(f"{'catch_shoot_3_rate':<30} {stats['catch_shoot_3_rate']:>10.6f}  ({stats['catch_shoot_3_rate']*100:>6.2f}%)")
        total_3pt = stats['corner_3_rate'] + stats['non_corner_3_rate']
        print(f"{'[REFERENCE] Total 3PT Rate':<30} {total_3pt:>10.6f}  ({total_3pt*100:>6.2f}%)")
        
        # Shot Quality
        print(f"\n⭐ SHOT QUALITY METRICS")
        print(f"{'ftr (FTA/FGA)':<30} {stats['ftr']:>10.6f}")
        print(f"{'shooting_fouls_per_100':<30} {stats['shooting_fouls_per_100']:>10.3f}")
        print(f"{'and1_rate (per 100 FGA)':<30} {stats['and1_rate']:>10.6f}  ({stats['and1_rate']:>6.2f}%)")
        print(f"{'assisted_2pt_rate':<30} {stats['assisted_2pt_rate']:>10.6f}  ({stats['assisted_2pt_rate']*100:>6.2f}%)")
        print(f"{'assisted_3pt_rate':<30} {stats['assisted_3pt_rate']:>10.6f}  ({stats['assisted_3pt_rate']*100:>6.2f}%)")
        
        # Creation
        print(f"\n🎨 ON-BALL CREATION & DECISION MAKING")
        print(f"{'assists_per_100':<30} {stats['assists_per_100']:>10.3f}")
        print(f"{'turnovers_per_100':<30} {stats['turnovers_per_100']:>10.3f}")
        print(f"{'ast_to_ratio':<30} {stats['ast_to_ratio']:>10.3f}")
        
        # Defense
        print(f"\n🛡️  DEFENSIVE IMPACT")
        print(f"{'steals_per_100':<30} {stats['steals_per_100']:>10.3f}")
        print(f"{'blocks_per_100':<30} {stats['blocks_per_100']:>10.3f}")
        print(f"{'def_reb_share':<30} {stats['def_reb_share']:>10.6f}  ({stats['def_reb_share']*100:>6.2f}%)")
        print(f"{'shooting_fouls_per_100':<30} {stats['shooting_fouls_per_100']:>10.3f}")
        print(f"{'total_fouls_per_100':<30} {stats['total_fouls_per_100']:>10.3f}")
        
        # Derived calculations for validation
        print(f"\n🔢 DERIVED CALCULATIONS (for validation)")
        unassisted_2pt = 1 - stats['assisted_2pt_rate']
        unassisted_3pt = 1 - stats['assisted_3pt_rate']
        print(f"{'unassisted_2pt_rate':<30} {unassisted_2pt:>10.6f}  ({unassisted_2pt*100:>6.2f}%)")
        print(f"{'unassisted_3pt_rate':<30} {unassisted_3pt:>10.6f}  ({unassisted_3pt*100:>6.2f}%)")
        
        # Raw counts (calculated from rates)
        print(f"\n📈 RAW COUNTS (derived from rates)")
        rim_attempts = int(stats['rim_attempt_rate'] * stats['total_fga'])
        corner_3_attempts = int(stats['corner_3_rate'] * stats['total_fga'])
        non_corner_3_attempts = int(stats['non_corner_3_rate'] * stats['total_fga'])
        mid_attempts = int(stats['mid_range_rate'] * stats['total_fga'])
        pullup_3_attempts = int(stats['pullup_3_rate'] * stats['total_fga'])
        catch_shoot_3_attempts = int(stats['catch_shoot_3_rate'] * stats['total_fga'])
        
        print(f"{'rim_attempts (est.)':<30} {rim_attempts:>10,}")
        print(f"{'corner_3_attempts (est.)':<30} {corner_3_attempts:>10,}")
        print(f"{'non_corner_3_attempts (est.)':<30} {non_corner_3_attempts:>10,}")
        print(f"{'mid_range_attempts (est.)':<30} {mid_attempts:>10,}")
        print(f"{'pullup_3_attempts (est.)':<30} {pullup_3_attempts:>10,}")
        print(f"{'catch_shoot_3_attempts (est.)':<30} {catch_shoot_3_attempts:>10,}")
        
        assists_total = int(stats['assists_per_100'] * stats['total_possessions'] / 100)
        turnovers_total = int(stats['turnovers_per_100'] * stats['total_possessions'] / 100)
        steals_total = int(stats['steals_per_100'] * stats['total_possessions'] / 100)
        blocks_total = int(stats['blocks_per_100'] * stats['total_possessions'] / 100)
        
        print(f"{'assists (est.)':<30} {assists_total:>10,}")
        print(f"{'turnovers (est.)':<30} {turnovers_total:>10,}")
        print(f"{'steals (est.)':<30} {steals_total:>10,}")
        print(f"{'blocks (est.)':<30} {blocks_total:>10,}")
        
    else:
        print(f"❌ No data found")

# Export to JSON for further analysis
print(f"\n\n{'=' * 100}")
print("EXPORTING TO JSON")
print('=' * 100)

with open('all_player_stats_export.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("✅ Exported all stats to: all_player_stats_export.json")

# Summary table with all stats
print(f"\n\n{'=' * 100}")
print("SUMMARY TABLE - ALL STATS")
print('=' * 100)

if all_results:
    # Header
    metrics = [
        'games_played', 'total_possessions', 'total_fga',
        'rim_attempt_rate', 'corner_3_rate', 'non_corner_3_rate',
        'pullup_3_rate', 'catch_shoot_3_rate', 'mid_range_rate',
        'ftr', 'shooting_fouls_per_100', 'and1_rate',
        'assisted_2pt_rate', 'assisted_3pt_rate',
        'assists_per_100', 'turnovers_per_100', 'ast_to_ratio',
        'steals_per_100', 'blocks_per_100', 'def_reb_share',
        'total_fouls_per_100'
    ]
    
    # Create CSV-style output
    print("\nPlayer," + ",".join(metrics))
    
    for player, stats in all_results.items():
        row = [player]
        for metric in metrics:
            value = stats.get(metric, 0)
            if isinstance(value, float):
                row.append(f"{value:.6f}")
            else:
                row.append(str(value))
        print(",".join(row))

print(f"\n{'=' * 100}")
print("COMPLETE - ALL STATS OUTPUT")
print('=' * 100)

