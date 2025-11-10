"""Compare ATS picks between old and new predictions for Nov 8 games."""
import pandas as pd

# Load predictions
old_preds = pd.read_csv('predictions/predictions_champion_2025-11-08.csv')
new_preds = pd.read_csv('predictions/current_season_new_predictions.csv')

# Filter to Nov 8
old_nov8 = old_preds[old_preds['date'] == '2025-11-08'].copy()
new_nov8 = new_preds[new_preds['date'] == '2025-11-08'].copy()

# Merge on game_id
merged = old_nov8.merge(
    new_nov8,
    on='game_id',
    suffixes=('_old', '_new')
)

# Sort by game_id for consistent ordering
merged = merged.sort_values('game_id')

print("="*90)
print("ATS COMPARISON: Old Model vs New Model (November 8, 2025)")
print("="*90)
print()

ats_changed = 0
total_games = len(merged)

for idx, row in merged.iterrows():
    game_id = row['game_id']
    away = row['away_team_old']
    home = row['home_team_old']
    
    spread_old = row['market_spread_home_old']
    spread_new = row['market_spread_home_new']
    
    pred_old = row['pred_margin_mu_old']
    pred_new = row['pred_margin_mu_new']
    
    # Determine ATS picks
    # If predicted margin > spread, we pick the home team to cover
    # If predicted margin < spread, we pick the away team to cover
    
    # Old model ATS
    old_edge = pred_old - spread_old
    if old_edge > 0:
        old_pick = f"{home} (cover)"
        old_pick_short = "HOME"
    else:
        old_pick = f"{away} (cover)"
        old_pick_short = "AWAY"
    
    # New model ATS
    new_edge = pred_new - spread_new
    if new_edge > 0:
        new_pick = f"{home} (cover)"
        new_pick_short = "HOME"
    else:
        new_pick = f"{away} (cover)"
        new_pick_short = "AWAY"
    
    # Check if ATS pick changed
    pick_changed = old_pick_short != new_pick_short
    if pick_changed:
        ats_changed += 1
    
    # Print game details
    print(f"{away} @ {home}")
    print(f"  Spread: Old={spread_old:+.1f}, New={spread_new:+.1f}" + 
          (f" (CHANGED)" if abs(spread_old - spread_new) > 0.01 else ""))
    print(f"  Old Model: Pred={pred_old:+.2f}, Edge={old_edge:+.2f} -> Pick: {old_pick_short}")
    print(f"  New Model: Pred={pred_new:+.2f}, Edge={new_edge:+.2f} -> Pick: {new_pick_short}")
    
    if pick_changed:
        print(f"  ** ATS PICK CHANGED: {old_pick_short} -> {new_pick_short} **")
    else:
        print(f"  ATS Pick: UNCHANGED ({old_pick_short})")
    
    print()

print("="*90)
print("SUMMARY")
print("="*90)
print(f"\nTotal Games: {total_games}")
print(f"ATS Picks Changed: {ats_changed} ({ats_changed/total_games*100:.1f}%)")
print(f"ATS Picks Unchanged: {total_games - ats_changed} ({(total_games - ats_changed)/total_games*100:.1f}%)")

# Show games where spread changed
spread_changed = merged[abs(merged['market_spread_home_old'] - merged['market_spread_home_new']) > 0.01]
print(f"\nGames with Spread Changes: {len(spread_changed)}")
if len(spread_changed) > 0:
    print("(Spread changes often explain ATS pick changes)")

