"""
Test PBP stats module with multiple players representing different playstyles
"""

from player_stats_from_pbp import calculate_player_pbp_stats, get_player_pbp_stats_summary

# Test players with different profiles
test_players = [
    "Karl-Anthony Towns",
    "Mitchell Robinson",
    "Trae Young",
    "Stephen Curry",
    "Shai Gilgeous-Alexander",
    "Luka Doncic",
    "Steven Adams",
    "Kristaps Porzingis"
]

season = "2023-2024"
max_date = "2024-01-01"  # First ~2 months of season

print("=" * 80)
print("NBA PLAY-BY-PLAY STATS - COMPREHENSIVE PLAYER TESTING")
print("=" * 80)
print(f"Season: {season}")
print(f"Date Range: Start through {max_date}")
print("=" * 80)

results = {}

for player in test_players:
    print(f"\n{'=' * 80}")
    print(f"TESTING: {player}")
    print('=' * 80)
    
    try:
        stats = calculate_player_pbp_stats(player, max_date, season)
        
        if stats:
            results[player] = stats
            print(get_player_pbp_stats_summary(stats))
            
            # Additional detailed breakdown
            print("\n📋 DETAILED BREAKDOWN:")
            print("-" * 80)
            
            # Shot distribution
            total_3pt = stats['corner_3_rate'] + stats['non_corner_3_rate']
            print(f"\n🎯 Shot Distribution:")
            print(f"  Rim:           {stats['rim_attempt_rate']:6.1%} of FGA")
            print(f"  Short Mid:     {stats['short_mid_rate']:6.1%} of FGA")
            print(f"  Long Mid:      {stats['long_mid_rate']:6.1%} of FGA")
            print(f"  Corner 3:      {stats['corner_3_rate']:6.1%} of FGA")
            print(f"  Non-Corner 3:  {stats['non_corner_3_rate']:6.1%} of FGA")
            print(f"  TOTAL 3PT:     {total_3pt:6.1%} of FGA")
            
            # Self-creation indicators
            print(f"\n🎨 Self-Creation Indicators:")
            print(f"  Pull-up 3s:           {stats['pullup_3_rate']:6.1%} of FGA")
            print(f"  Catch-Shoot 3s:       {stats['catch_shoot_3_rate']:6.1%} of FGA")
            print(f"  Assisted 2PT Rate:    {stats['assisted_2pt_rate']:6.1%}")
            print(f"  Assisted 3PT Rate:    {stats['assisted_3pt_rate']:6.1%}")
            print(f"  Unassisted 2PT Rate:  {1 - stats['assisted_2pt_rate']:6.1%}")
            print(f"  Unassisted 3PT Rate:  {1 - stats['assisted_3pt_rate']:6.1%}")
            
            # Efficiency & Getting to line
            print(f"\n⚡ Efficiency & Fouls:")
            print(f"  Free Throw Rate:      {stats['ftr']:6.3f}")
            print(f"  Shooting Fouls/100:   {stats['shooting_fouls_per_100']:6.1f}")
            print(f"  And-1 Rate:           {stats['and1_rate']:6.2f}%")
            
            # Playmaking
            print(f"\n🏀 Playmaking:")
            print(f"  Assists/100:          {stats['assists_per_100']:6.1f}")
            print(f"  Turnovers/100:        {stats['turnovers_per_100']:6.1f}")
            print(f"  Ast/TO Ratio:         {stats['ast_to_ratio']:6.2f}")
            
            # Defense
            print(f"\n🛡️  Defensive Impact:")
            print(f"  Steals/100:           {stats['steals_per_100']:6.1f}")
            print(f"  Blocks/100:           {stats['blocks_per_100']:6.1f}")
            print(f"  Def Reb Share:        {stats['def_reb_share']:6.1%}")
            print(f"  Shooting Fouls/100:   {stats['shooting_fouls_per_100']:6.1f}")
            print(f"  Total Fouls/100:      {stats['total_fouls_per_100']:6.1f}")
            
            # Context
            print(f"\n📈 Context:")
            print(f"  Total Possessions:    {stats['total_possessions']:6,}")
            print(f"  Total FGA:            {stats['total_fga']:6,}")
            print(f"  Games Played:         {stats['games_played']:6}")
            
        else:
            print(f"❌ No stats found for {player}")
            print("   Possible reasons:")
            print("   - Player name spelling doesn't match dataset")
            print("   - Player didn't play before the max_date")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

# Summary comparison table
print("\n\n" + "=" * 80)
print("SUMMARY COMPARISON TABLE")
print("=" * 80)

if results:
    print(f"\n{'Player':<25} {'Rim%':>6} {'3PT%':>6} {'Mid%':>6} {'AST/100':>8} {'BLK/100':>8}")
    print("-" * 80)
    
    for player, stats in results.items():
        rim = stats['rim_attempt_rate']
        three = stats['corner_3_rate'] + stats['non_corner_3_rate']
        mid = stats['mid_range_rate']
        ast = stats['assists_per_100']
        blk = stats['blocks_per_100']
        
        print(f"{player:<25} {rim:>5.1%} {three:>5.1%} {mid:>5.1%} {ast:>8.1f} {blk:>8.1f}")

print("\n" + "=" * 80)
print("TESTING COMPLETE")
print("=" * 80)

