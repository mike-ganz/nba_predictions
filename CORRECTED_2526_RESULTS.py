"""
Final corrected results for 25-26 season after fixing normalization
"""
import pandas as pd
import numpy as np

print("="*80)
print("CORRECTED 25-26 SEASON RESULTS")
print("="*80)
print()

# Load corrected predictions
old_df = pd.read_csv('predictions/test_2526_OLD_fixed_CORRECTED.csv')
new_df = pd.read_csv('predictions/test_2526_NEW_fixed_CORRECTED.csv')

def calculate_ats(df, model_name):
    df['actual_home_covers'] = (df['actual_margin'] > -df['market_spread_home']).astype(int)
    df['predicted_home_covers'] = (df['pred_margin_mu'] > -df['market_spread_home']).astype(int)
    df['correct'] = (df['actual_home_covers'] == df['predicted_home_covers']).astype(int)
    
    ats_accuracy = df['correct'].mean() * 100
    mae = np.abs(df['pred_margin_mu'] - df['actual_margin']).mean()
    
    # ROI calculation
    correct = df['correct'].sum()
    incorrect = len(df) - correct
    profit = (correct * 0.909) - incorrect
    roi = (profit / len(df)) * 100
    
    print(f"{model_name}:")
    print(f"  Games: {len(df)}")
    print(f"  ATS Accuracy: {ats_accuracy:.2f}%")
    print(f"  MAE: {mae:.2f} points")
    print(f"  ROI: {roi:+.2f}%")
    print(f"  Predicted home covers: {df['predicted_home_covers'].sum()} ({df['predicted_home_covers'].mean()*100:.1f}%)")
    print()
    
    return ats_accuracy, roi

old_ats, old_roi = calculate_ats(old_df, "OLD Model (21-24)")
new_ats, new_roi = calculate_ats(new_df, "NEW Model (21-25)")

print("="*80)
print("COMPARISON")
print("="*80)
print()

print(f"ATS Accuracy: OLD={old_ats:.2f}% vs NEW={new_ats:.2f}% (diff: {new_ats-old_ats:+.2f} pp)")
print(f"ROI: OLD={old_roi:+.2f}% vs NEW={new_roi:+.2f}% (diff: {new_roi-old_roi:+.2f} pp)")
print()

if abs(old_ats - new_ats) < 3:
    print("✅ Both models perform similarly (within 3 pp)")
else:
    winner = "NEW" if new_ats > old_ats else "OLD"
    print(f"📊 {winner} model performs better by {abs(new_ats - old_ats):.2f} pp")

print()
print("="*80)
print("IMPORTANT NOTE")
print("="*80)
print()
print("⚠️  Sample size: Only 72 games!")
print("    95% CI: ±11 percentage points")
print("    Need 500+ games for stable estimates")
print()
print("The previous 56.94% ATS was INVALID (100% home bias bug)")
print(f"Corrected ATS: {old_ats:.2f}% (more realistic but still small sample)")
print()

