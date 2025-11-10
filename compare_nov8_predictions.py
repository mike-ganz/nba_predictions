"""Compare Nov 8 predictions between day-of and new backlook."""
import pandas as pd

# Load predictions
day_of = pd.read_csv('predictions/predictions_champion_2025-11-08.csv')
backlook_new = pd.read_csv('predictions/current_season_new_predictions.csv')

# Filter to Nov 8
nov8_games = day_of[day_of['date'] == '2025-11-08']
backlook_nov8 = backlook_new[backlook_new['date'] == '2025-11-08']

print("="*80)
print("COMPARISON: Day-of vs New Backlook Predictions for Nov 8, 2025")
print("="*80)

# Merge on game_id
merged = nov8_games.merge(
    backlook_nov8,
    on='game_id',
    suffixes=('_day', '_back')
)

print(f"\nFound {len(merged)} matching games\n")

for _, row in merged.iterrows():
    game_id = row['game_id']
    away = row['away_team_day']
    home = row['home_team_day']
    
    spread_day = row['market_spread_home_day']
    spread_back = row['market_spread_home_back']
    
    pred_day = row['pred_margin_mu_day']
    pred_back = row['pred_margin_mu_back']
    
    diff = abs(pred_day - pred_back)
    
    match_str = "MATCH" if diff < 0.5 else "DIFFER" if diff < 2.0 else "LARGE DIFF"
    
    print(f"{away} @ {home}")
    print(f"  Game ID: {game_id}")
    print(f"  Spread: day={spread_day:.1f}, back={spread_back:.1f}, diff={abs(spread_day-spread_back):.1f}")
    print(f"  Prediction: day={pred_day:.3f}, back={pred_back:.3f}, diff={diff:.3f} [{match_str}]")
    print()

# Focus on LAL @ ATL
print("="*80)
print("LAL @ ATL DETAILED COMPARISON")
print("="*80)

lal_atl = merged[merged['game_id'] == '2025-11-08-LAL-ATL']
if len(lal_atl) > 0:
    row = lal_atl.iloc[0]
    print(f"\nGame: {row['away_team_day']} @ {row['home_team_day']}")
    print(f"  Spread (day-of): {row['market_spread_home_day']:.1f}")
    print(f"  Spread (backlook): {row['market_spread_home_back']:.1f}")
    print(f"  Prediction (day-of): {row['pred_margin_mu_day']:.6f}")
    print(f"  Prediction (backlook): {row['pred_margin_mu_back']:.6f}")
    print(f"  Difference: {abs(row['pred_margin_mu_day'] - row['pred_margin_mu_back']):.6f}")
    
    if abs(row['pred_margin_mu_day'] - row['pred_margin_mu_back']) < 0.5:
        print("\n  RESULT: PREDICTIONS MATCH!")
    else:
        print("\n  RESULT: Predictions still differ")
        if abs(row['market_spread_home_day'] - row['market_spread_home_back']) > 0.1:
            print("  Note: Spreads are different, which explains some variance")
else:
    print("\nLAL @ ATL game not found in data")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

exact_matches = len(merged[abs(merged['pred_margin_mu_day'] - merged['pred_margin_mu_back']) < 0.5])
close_matches = len(merged[abs(merged['pred_margin_mu_day'] - merged['pred_margin_mu_back']) < 1.0])

print(f"\nExact matches (<0.5 diff): {exact_matches}/{len(merged)}")
print(f"Close matches (<1.0 diff): {close_matches}/{len(merged)}")

# Check if spread differences explain prediction differences
spread_diff_games = merged[abs(merged['market_spread_home_day'] - merged['market_spread_home_back']) > 0.1]
print(f"\nGames with different spreads: {len(spread_diff_games)}")

