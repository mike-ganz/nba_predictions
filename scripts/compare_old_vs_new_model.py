"""
Compare old model (21-24) vs new model (21-25) on 2025-26 season.
"""
import pandas as pd
import numpy as np

print("="*80)
print("MODEL COMPARISON: OLD (21-24) vs NEW (21-25)")
print("2025-26 Season Performance")
print("="*80)
print()

# Load predictions from both models
old_pred = pd.read_csv('predictions/current_season_2025_2026_predictions.csv')
new_pred = pd.read_csv('predictions/current_season_2025_2026_predictions_NEW_MODEL.csv')

print(f"Total games: {len(old_pred)}")
print()

# Calculate ATS correctness for both
def calculate_ats_metrics(df, model_name):
    df = df.copy()
    df['actual_home_covers'] = (df['actual_margin'] > -df['market_spread_home']).astype(int)
    df['predicted_home_covers'] = (df['pred_margin_mu'] > -df['market_spread_home']).astype(int)
    df['ats_correct'] = (df['actual_home_covers'] == df['predicted_home_covers']).astype(int)
    
    # Overall
    ats_acc = df['ats_correct'].mean() * 100
    correct = df['ats_correct'].sum()
    
    # ROI
    wins = correct
    losses = len(df) - correct
    profit = wins * 0.909 - losses * 1.1
    total_wagered = len(df) * 1.1
    roi = (profit / total_wagered * 100) if total_wagered > 0 else 0.0
    
    # MAE
    mae = (df['pred_margin_mu'] - df['actual_margin']).abs().mean()
    
    # By favorite type
    df['home_favored'] = df['market_spread_home'] < 0
    df['away_favored'] = df['market_spread_home'] > 0
    
    home_fav = df[df['home_favored']]
    away_fav = df[df['away_favored']]
    
    home_fav_acc = home_fav['ats_correct'].mean() * 100 if len(home_fav) > 0 else 0
    away_fav_acc = away_fav['ats_correct'].mean() * 100 if len(away_fav) > 0 else 0
    
    return {
        'model': model_name,
        'n_games': len(df),
        'ats_acc': ats_acc,
        'correct': correct,
        'roi': roi,
        'mae': mae,
        'home_fav_games': len(home_fav),
        'home_fav_acc': home_fav_acc,
        'away_fav_games': len(away_fav),
        'away_fav_acc': away_fav_acc,
    }

old_metrics = calculate_ats_metrics(old_pred, "Old Model (21-24)")
new_metrics = calculate_ats_metrics(new_pred, "New Model (21-25)")

print("="*80)
print("OVERALL PERFORMANCE")
print("="*80)
print()
print(f"{'Metric':<25} {'Old Model (21-24)':<20} {'New Model (21-25)':<20} {'Difference':<15}")
print("-"*80)
print(f"{'Games':<25} {old_metrics['n_games']:<20} {new_metrics['n_games']:<20} {new_metrics['n_games'] - old_metrics['n_games']:<15}")
print(f"{'ATS Accuracy':<25} {old_metrics['ats_acc']:.2f}%{'':<15} {new_metrics['ats_acc']:.2f}%{'':<15} {new_metrics['ats_acc'] - old_metrics['ats_acc']:+.2f}%")
print(f"{'Correct Predictions':<25} {old_metrics['correct']:<20} {new_metrics['correct']:<20} {new_metrics['correct'] - old_metrics['correct']:+}")
print(f"{'ROI':<25} {old_metrics['roi']:+.2f}%{'':<15} {new_metrics['roi']:+.2f}%{'':<15} {new_metrics['roi'] - old_metrics['roi']:+.2f}%")
print(f"{'MAE (points)':<25} {old_metrics['mae']:.2f}{'':<17} {new_metrics['mae']:.2f}{'':<17} {new_metrics['mae'] - old_metrics['mae']:+.2f}")
print()

print("="*80)
print("BY FAVORITE TYPE")
print("="*80)
print()
print("HOME FAVORED:")
print(f"  Old Model: {old_metrics['home_fav_games']} games, {old_metrics['home_fav_acc']:.2f}% ATS")
print(f"  New Model: {new_metrics['home_fav_games']} games, {new_metrics['home_fav_acc']:.2f}% ATS")
print(f"  Difference: {new_metrics['home_fav_acc'] - old_metrics['home_fav_acc']:+.2f}%")
print()
print("AWAY FAVORED:")
print(f"  Old Model: {old_metrics['away_fav_games']} games, {old_metrics['away_fav_acc']:.2f}% ATS")
print(f"  New Model: {new_metrics['away_fav_games']} games, {new_metrics['away_fav_acc']:.2f}% ATS")
print(f"  Difference: {new_metrics['away_fav_acc'] - old_metrics['away_fav_acc']:+.2f}%")
print()

# Determine winner
print("="*80)
print("VERDICT")
print("="*80)
print()

if new_metrics['ats_acc'] > old_metrics['ats_acc']:
    print(f"✅ NEW MODEL WINS by {new_metrics['ats_acc'] - old_metrics['ats_acc']:.2f} percentage points")
    print(f"   ATS: {new_metrics['ats_acc']:.2f}% vs {old_metrics['ats_acc']:.2f}%")
    print(f"   ROI: {new_metrics['roi']:+.2f}% vs {old_metrics['roi']:+.2f}%")
elif old_metrics['ats_acc'] > new_metrics['ats_acc']:
    print(f"✅ OLD MODEL WINS by {old_metrics['ats_acc'] - new_metrics['ats_acc']:.2f} percentage points")
    print(f"   ATS: {old_metrics['ats_acc']:.2f}% vs {new_metrics['ats_acc']:.2f}%")
    print(f"   ROI: {old_metrics['roi']:+.2f}% vs {new_metrics['roi']:+.2f}%")
else:
    print("🤝 TIE - Both models perform equally")

print()

# Save detailed comparison
comparison_df = pd.DataFrame([old_metrics, new_metrics])
comparison_df.to_csv('predictions/model_comparison_21_24_vs_21_25.csv', index=False)
print("✅ Detailed comparison saved to: predictions/model_comparison_21_24_vs_21_25.csv")
print()
print("="*80)

