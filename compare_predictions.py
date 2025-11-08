"""Compare predictions to understand what the fix achieved."""

import pandas as pd

# Load all three prediction sets
daily = pd.read_csv('predictions/predictions_champion_2025-11-07.csv')
old_eval = pd.read_csv('backups/pre_rest_days_fix_20251108_174325/current_season_champion_2025_2026_predictions.csv')
new_eval = pd.read_csv('predictions/current_season_champion_corrected_predictions.csv')

print("="*70)
print("WHAT THE FIX ACTUALLY CHANGED")
print("="*70)
print()

# Focus on November 7 games
nov7_daily = daily.sort_values('game_id')
nov7_old = old_eval[old_eval['date'] == '2025-11-07'].sort_values('game_id')
nov7_new = new_eval[new_eval['date'] == '2025-11-07'].sort_values('game_id')

print("THREE DIFFERENT PREDICTION SOURCES:")
print("-"*70)
print()
print("1. DAILY PREDICTIONS (Nov 7 before games)")
print("   - Used: OLD model (trained on BUGGY rest days +1)")
print("   - Input: CORRECT rest days (from NBA schedule)")
print()
print("2. EVALUATION OLD (post-game, before fix)")
print("   - Used: OLD model (trained on BUGGY rest days +1)")
print("   - Input: BUGGY rest days (from boxscores, +1 error)")
print()
print("3. EVALUATION NEW (post-game, after fix)")
print("   - Used: NEW model (retrained on CORRECT rest days)")
print("   - Input: CORRECT rest days (from boxscores, now fixed)")
print()
print()

print("EXAMPLE GAMES:")
print("-"*70)

for game_id in ['2025-11-07-BOS-ORL', '2025-11-07-CHI-MIL', '2025-11-07-CHA-MIA']:
    daily_pred = nov7_daily[nov7_daily['game_id'] == game_id]['pred_margin_mu'].values[0]
    old_pred = nov7_old[nov7_old['game_id'] == game_id]['pred_margin_mu'].values[0]
    new_pred = nov7_new[nov7_new['game_id'] == game_id]['pred_margin_mu'].values[0]
    
    print(f"\n{game_id}:")
    print(f"  Daily (old model + correct rest):  {daily_pred:7.3f}")
    print(f"  Eval OLD (old model + buggy rest): {old_pred:7.3f}  [Diff: {old_pred - daily_pred:+.3f}]")
    print(f"  Eval NEW (new model + correct rest): {new_pred:7.3f}  [Diff: {new_pred - daily_pred:+.3f}]")

print()
print()
print("="*70)
print("WHAT GOT FIXED:")
print("="*70)
print()
print("BEFORE FIX:")
print("  Training: Model trained on data with rest_days = WRONG (+1)")
print("  Daily:    Model predicts with rest_days = CORRECT (from schedule)")
print("  Eval:     Model predicts with rest_days = WRONG (+1, from boxscores)")
print("  RESULT:   Daily and Eval predictions DIFFERED")
print()
print("AFTER FIX:")
print("  Training: Model trained on data with rest_days = CORRECT")
print("  Daily:    Model predicts with rest_days = CORRECT (from schedule)")
print("  Eval:     Model predicts with rest_days = CORRECT (from boxscores)")
print("  RESULT:   Everything uses CORRECT data, model properly trained")
print()
print("IMPORTANT NOTE:")
print("  - The daily predictions still differ from new eval predictions")
print("  - This is because the MODEL ITSELF changed (retrained)")
print("  - The old model was trained on incorrect data")
print("  - The new model is trained on correct data")
print("  - Both now use correct inputs, but different models = different outputs")
print()

