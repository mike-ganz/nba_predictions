"""
Test overfitting hypothesis: NEW model should perform better on 24-25 (its training data)
but worse on 25-26 (truly unseen).
"""
import pandas as pd
import numpy as np

print("="*80)
print("OVERFITTING HYPOTHESIS TEST")
print("="*80)
print()

print("Hypothesis: If NEW model overfit to 24-25, then:")
print("  1. NEW should OUTPERFORM OLD on 24-25 (in-sample for NEW)")
print("  2. OLD should OUTPERFORM NEW on 25-26 (out-of-sample for both)")
print()

def calculate_ats(df, name):
    """Calculate ATS metrics"""
    df['actual_home_covers'] = (df['actual_margin'] > -df['market_spread_home']).astype(int)
    df['predicted_home_covers'] = (df['pred_margin_mu'] > -df['market_spread_home']).astype(int)
    df['correct'] = (df['actual_home_covers'] == df['predicted_home_covers']).astype(int)
    
    ats_accuracy = df['correct'].mean() * 100
    mae = np.abs(df['pred_margin_mu'] - df['actual_margin']).mean()
    
    # ROI
    correct = df['correct'].sum()
    incorrect = len(df) - correct
    profit = (correct * 0.909) - incorrect
    roi = (profit / len(df)) * 100
    
    return {
        'name': name,
        'games': len(df),
        'ats': ats_accuracy,
        'mae': mae,
        'roi': roi
    }

# Load 24-25 predictions
old_2425 = pd.read_csv('predictions/test_2425_OLD_fixed.csv')
new_2425 = pd.read_csv('predictions/test_2425_NEW_fixed.csv')

# Load 25-26 predictions
old_2526 = pd.read_csv('predictions/test_2526_OLD_fixed_CORRECTED.csv')
new_2526 = pd.read_csv('predictions/test_2526_NEW_fixed_CORRECTED.csv')

# Calculate metrics
old_2425_metrics = calculate_ats(old_2425, "OLD on 24-25")
new_2425_metrics = calculate_ats(new_2425, "NEW on 24-25")
old_2526_metrics = calculate_ats(old_2526, "OLD on 25-26")
new_2526_metrics = calculate_ats(new_2526, "NEW on 25-26")

print("="*80)
print("24-25 SEASON RESULTS (In-sample for NEW, Out-of-sample for OLD)")
print("="*80)
print()

print(f"{'Model':<20} {'Games':<8} {'ATS %':<10} {'MAE':<10} {'ROI %':<10}")
print("-"*80)
print(f"{old_2425_metrics['name']:<20} {old_2425_metrics['games']:<8} {old_2425_metrics['ats']:>8.2f}% {old_2425_metrics['mae']:>9.2f} {old_2425_metrics['roi']:>9.2f}%")
print(f"{new_2425_metrics['name']:<20} {new_2425_metrics['games']:<8} {new_2425_metrics['ats']:>8.2f}% {new_2425_metrics['mae']:>9.2f} {new_2425_metrics['roi']:>9.2f}%")
print()

diff_2425 = new_2425_metrics['ats'] - old_2425_metrics['ats']
if diff_2425 > 0:
    print(f"✓ NEW model performs BETTER on 24-25 by {diff_2425:+.2f} pp")
    print(f"  This supports overfitting (NEW trained on this data)")
else:
    print(f"✗ NEW model performs WORSE on 24-25 by {diff_2425:+.2f} pp")
    print(f"  This does NOT support overfitting hypothesis!")

print()
print("="*80)
print("25-26 SEASON RESULTS (Out-of-sample for both)")
print("="*80)
print()

print(f"{'Model':<20} {'Games':<8} {'ATS %':<10} {'MAE':<10} {'ROI %':<10}")
print("-"*80)
print(f"{old_2526_metrics['name']:<20} {old_2526_metrics['games']:<8} {old_2526_metrics['ats']:>8.2f}% {old_2526_metrics['mae']:>9.2f} {old_2526_metrics['roi']:>9.2f}%")
print(f"{new_2526_metrics['name']:<20} {new_2526_metrics['games']:<8} {new_2526_metrics['ats']:>8.2f}% {new_2526_metrics['mae']:>9.2f} {new_2526_metrics['roi']:>9.2f}%")
print()

diff_2526 = old_2526_metrics['ats'] - new_2526_metrics['ats']
if diff_2526 > 0:
    print(f"✓ OLD model performs BETTER on 25-26 by {diff_2526:+.2f} pp")
    print(f"  This supports overfitting (NEW overfit to 24-25, fails on 25-26)")
else:
    print(f"✗ OLD model performs WORSE on 25-26 by {diff_2526:+.2f} pp")
    print(f"  This does NOT support overfitting hypothesis!")

print()
print("="*80)
print("HYPOTHESIS TEST RESULTS")
print("="*80)
print()

# Check both conditions
condition1 = diff_2425 > 0  # NEW better on 24-25
condition2 = diff_2526 > 0  # OLD better on 25-26

if condition1 and condition2:
    print("✅ HYPOTHESIS CONFIRMED!")
    print()
    print("Evidence of overfitting:")
    print(f"  • NEW model better on 24-25 (its training data): {diff_2425:+.2f} pp")
    print(f"  • OLD model better on 25-26 (unseen data): {diff_2526:+.2f} pp")
    print()
    print("The NEW model learned patterns specific to 24-25 that don't generalize.")
    print("This is classic overfitting behavior.")
elif condition1 and not condition2:
    print("🤔 MIXED RESULTS")
    print()
    print("NEW is better on 24-25 (expected for overfitting)")
    print("But NEW is also better on 25-26 (unexpected!)")
    print()
    print("This suggests NEW model is actually better, not overfit.")
elif not condition1 and condition2:
    print("🤔 MIXED RESULTS")
    print()
    print("NEW is worse on 24-25 (unexpected)")
    print("And OLD is better on 25-26 (expected)")
    print()
    print("This suggests different issue than overfitting.")
else:
    print("❌ HYPOTHESIS REJECTED")
    print()
    print("NEW model performs worse on both 24-25 AND 25-26.")
    print("This is not classic overfitting behavior.")

print()
print("="*80)
print("PERFORMANCE SUMMARY")
print("="*80)
print()

print(f"OLD Model: {old_2425_metrics['ats']:.2f}% (24-25) → {old_2526_metrics['ats']:.2f}% (25-26)")
print(f"NEW Model: {new_2425_metrics['ats']:.2f}% (24-25) → {new_2526_metrics['ats']:.2f}% (25-26)")
print()

old_gap = old_2425_metrics['ats'] - old_2526_metrics['ats']
new_gap = new_2425_metrics['ats'] - new_2526_metrics['ats']

print(f"OLD Model gap (24-25 to 25-26): {old_gap:+.2f} pp")
print(f"NEW Model gap (24-25 to 25-26): {new_gap:+.2f} pp")
print()

if abs(new_gap) > abs(old_gap) + 5:
    print("⚠️  NEW model shows MUCH LARGER performance swing between seasons")
    print("   This indicates lower stability/generalization")
elif abs(new_gap) < abs(old_gap) - 5:
    print("✓ NEW model shows MORE STABLE performance across seasons")
    print("   This indicates better generalization")
else:
    print("≈ Both models show similar stability across seasons")

print()
print("="*80)

