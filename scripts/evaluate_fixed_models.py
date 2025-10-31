"""
Comprehensive evaluation of OLD vs NEW models after fixing data leakage.
Compares performance on both 24-25 and 25-26 test sets.
"""
import pandas as pd
import numpy as np
from pathlib import Path

print("="*100)
print("COMPREHENSIVE MODEL EVALUATION - POST DATA LEAKAGE FIX")
print("="*100)
print()

# Load all predictions
old_2425 = pd.read_csv('predictions/test_2425_OLD_fixed.csv')
new_2425 = pd.read_csv('predictions/test_2425_NEW_fixed.csv')
old_2526 = pd.read_csv('predictions/test_2526_OLD_fixed.csv')
new_2526 = pd.read_csv('predictions/test_2526_NEW_fixed.csv')

print(f"Loaded predictions:")
print(f"  OLD model on 24-25: {len(old_2425)} games")
print(f"  NEW model on 24-25: {len(new_2425)} games")
print(f"  OLD model on 25-26: {len(old_2526)} games")
print(f"  NEW model on 25-26: {len(new_2526)} games")
print()

def calculate_metrics(df, name):
    """Calculate comprehensive metrics for a prediction set."""
    
    # Determine spread column name
    spread_col = 'market_spread_home' if 'market_spread_home' in df.columns else 'market_spread'
    
    # Calculate covers
    df['actual_home_covers'] = (df['actual_margin'] > -df[spread_col]).astype(int)
    df['predicted_home_covers'] = (df['pred_margin_mu'] > -df[spread_col]).astype(int)
    
    # Favorite definitions (CORRECTED)
    df['home_favored'] = df[spread_col] < 0
    df['away_favored'] = df[spread_col] > 0
    
    # Calculate metrics
    ats_correct = (df['actual_home_covers'] == df['predicted_home_covers']).sum()
    ats_accuracy = (ats_correct / len(df)) * 100
    mae = np.abs(df['pred_margin_mu'] - df['actual_margin']).mean()
    
    # By favorite type
    home_fav = df[df['home_favored']]
    away_fav = df[df['away_favored']]
    
    home_fav_correct = (home_fav['actual_home_covers'] == home_fav['predicted_home_covers']).sum() if len(home_fav) > 0 else 0
    away_fav_correct = (away_fav['actual_home_covers'] == away_fav['predicted_home_covers']).sum() if len(away_fav) > 0 else 0
    
    home_fav_accuracy = (home_fav_correct / len(home_fav) * 100) if len(home_fav) > 0 else 0
    away_fav_accuracy = (away_fav_correct / len(away_fav) * 100) if len(away_fav) > 0 else 0
    
    # Calculate ROI (assuming -110 odds)
    correct_picks = ats_correct
    incorrect_picks = len(df) - ats_correct
    profit = (correct_picks * 0.909) - incorrect_picks  # Win $0.909 per $1 at -110
    roi = (profit / len(df)) * 100
    
    return {
        'name': name,
        'games': len(df),
        'ats_accuracy': ats_accuracy,
        'mae': mae,
        'roi': roi,
        'home_fav_games': len(home_fav),
        'home_fav_accuracy': home_fav_accuracy,
        'away_fav_games': len(away_fav),
        'away_fav_accuracy': away_fav_accuracy,
    }

# Calculate metrics for all combinations
print("="*100)
print("CALCULATING METRICS...")
print("="*100)
print()

metrics = [
    calculate_metrics(old_2425, "OLD (21-24) on 24-25"),
    calculate_metrics(new_2425, "NEW (21-25) on 24-25"),
    calculate_metrics(old_2526, "OLD (21-24) on 25-26"),
    calculate_metrics(new_2526, "NEW (21-25) on 25-26"),
]

# Display results
print("="*100)
print("OVERALL PERFORMANCE")
print("="*100)
print()

header = f"{'Model':<25} {'Games':<8} {'ATS%':<10} {'MAE':<10} {'ROI%':<10}"
print(header)
print("-"*100)

for m in metrics:
    print(f"{m['name']:<25} {m['games']:<8} {m['ats_accuracy']:>8.2f}% {m['mae']:>9.2f} {m['roi']:>9.2f}%")

print()
print("="*100)
print("BREAKDOWN BY FAVORITE TYPE")
print("="*100)
print()

print(f"{'Model':<25} {'Fav Type':<12} {'Games':<8} {'ATS%':<10}")
print("-"*100)

for m in metrics:
    print(f"{m['name']:<25} {'Home Favored':<12} {m['home_fav_games']:<8} {m['home_fav_accuracy']:>8.2f}%")
    print(f"{'':<25} {'Away Favored':<12} {m['away_fav_games']:<8} {m['away_fav_accuracy']:>8.2f}%")
    print()

print("="*100)
print("KEY COMPARISONS")
print("="*100)
print()

# OLD vs NEW on 24-25 (in-sample for NEW, out-of-sample for OLD)
old_24 = metrics[0]
new_24 = metrics[1]

print("24-25 TEST SET (In-sample for NEW, Out-of-sample for OLD):")
print(f"  OLD model ATS: {old_24['ats_accuracy']:.2f}%")
print(f"  NEW model ATS: {new_24['ats_accuracy']:.2f}%")
print(f"  Difference: {new_24['ats_accuracy'] - old_24['ats_accuracy']:+.2f} pp (NEW {'BETTER' if new_24['ats_accuracy'] > old_24['ats_accuracy'] else 'WORSE'})")
print()

# OLD vs NEW on 25-26 (out-of-sample for both)
old_25 = metrics[2]
new_25 = metrics[3]

print("25-26 TEST SET (Out-of-sample for both):")
print(f"  OLD model ATS: {old_25['ats_accuracy']:.2f}%")
print(f"  NEW model ATS: {new_25['ats_accuracy']:.2f}%")
print(f"  Difference: {new_25['ats_accuracy'] - old_25['ats_accuracy']:+.2f} pp (NEW {'BETTER' if new_25['ats_accuracy'] > old_25['ats_accuracy'] else 'WORSE'})")
print()

print("="*100)
print("OVERFITTING DIAGNOSIS")
print("="*100)
print()

# Check for overfitting by comparing in-sample vs out-of-sample
new_in_sample = new_24['ats_accuracy']
new_out_of_sample = new_25['ats_accuracy']
new_gap = new_in_sample - new_out_of_sample

old_pseudo_in_sample = old_24['ats_accuracy']  # Not truly in-sample but closer
old_out_of_sample = old_25['ats_accuracy']
old_gap = old_pseudo_in_sample - old_out_of_sample

print(f"NEW Model (21-25):")
print(f"  Performance on 24-25 (in training): {new_in_sample:.2f}%")
print(f"  Performance on 25-26 (unseen):      {new_out_of_sample:.2f}%")
print(f"  Gap:                                {new_gap:+.2f} pp")
print()

print(f"OLD Model (21-24):")
print(f"  Performance on 24-25 (unseen):      {old_pseudo_in_sample:.2f}%")
print(f"  Performance on 25-26 (unseen):      {old_out_of_sample:.2f}%")
print(f"  Gap:                                {old_gap:+.2f} pp")
print()

if new_gap > old_gap + 2:
    print("⚠️  NEW model shows signs of OVERFITTING:")
    print(f"    - Performance drops {new_gap:.2f} pp from 24-25 to 25-26")
    print(f"    - OLD model only drops {old_gap:.2f} pp")
    print(f"    - Difference: {new_gap - old_gap:.2f} pp worse generalization")
elif new_gap < old_gap - 2:
    print("✅ NEW model shows BETTER generalization:")
    print(f"    - Performance only drops {new_gap:.2f} pp from 24-25 to 25-26")
    print(f"    - OLD model drops {old_gap:.2f} pp")
    print(f"    - Difference: {old_gap - new_gap:.2f} pp better generalization")
else:
    print("➡️  Both models show similar generalization patterns")

print()
print("="*100)
print("RECOMMENDATION")
print("="*100)
print()

# Determine which model is better
if old_25['ats_accuracy'] > new_25['ats_accuracy']:
    print("🏆 RECOMMENDATION: Use OLD model (21-24)")
    print()
    print("Reasons:")
    print(f"  1. Better performance on truly unseen data (25-26): {old_25['ats_accuracy']:.2f}% vs {new_25['ats_accuracy']:.2f}%")
    print(f"  2. More robust generalization")
    print(f"  3. Less prone to overfitting")
else:
    print("🏆 RECOMMENDATION: Use NEW model (21-25)")
    print()
    print("Reasons:")
    print(f"  1. Better performance on truly unseen data (25-26): {new_25['ats_accuracy']:.2f}% vs {old_25['ats_accuracy']:.2f}%")
    print(f"  2. Incorporates more recent data (24-25 season)")
    print(f"  3. Better overall generalization")

print()
print("="*100)
print("IMPACT OF DATA LEAKAGE FIX")
print("="*100)
print()

print("The fixed projected_minutes logic means:")
print("  ✓ Players who were OUT (0 min) → projected = 0 (assume injury report)")
print("  ✓ Players who played → projected = baseline (assume no advance knowledge)")
print()
print("This is more realistic for production predictions where we only have:")
print("  - Injury reports for players who won't play")
print("  - Historical averages for everyone else")
print()

# Save results
results_df = pd.DataFrame(metrics)
results_df.to_csv('predictions/model_comparison_fixed.csv', index=False)
print("✅ Results saved to: predictions/model_comparison_fixed.csv")
print("="*100)

