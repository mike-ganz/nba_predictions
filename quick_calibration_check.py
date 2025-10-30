"""Quick check: Does high confidence = lower accuracy? (Inverted calibration test)"""
import pandas as pd
import numpy as np

print("\n" + "="*70)
print("QUICK CALIBRATION CHECK")
print("="*70 + "\n")

# Load 2024-2025 predictions
df = pd.read_csv('reports/2425_with_players/per_game_predictions.csv')

print(f"Total games: {len(df)}\n")

# Calculate ATS correctness
df['home_margin'] = df['actual_home'] - df['actual_away']
df['home_covers'] = df['home_margin'] + df['market_spread_home'] > 0
df['away_covers'] = df['home_margin'] + df['market_spread_home'] < 0

# Model picks the side with higher cover probability
df['model_pick_home'] = df['cover_prob_home'] > df['cover_prob_away']
df['ats_correct'] = np.where(
    df['model_pick_home'],
    df['home_covers'],
    df['away_covers']
)

# Calculate confidence (max of home/away cover probability)
df['max_cover_prob'] = df[['cover_prob_home', 'cover_prob_away']].max(axis=1)

# Create confidence buckets
df['confidence_bucket'] = pd.cut(df['max_cover_prob'],
                                  bins=[0.5, 0.55, 0.6, 0.65, 0.7, 1.0],
                                  labels=['50-55%', '55-60%', '60-65%', '65-70%', '70%+'])

# Calculate accuracy by confidence level
print("ACCURACY BY CONFIDENCE LEVEL:")
print("-" * 70)
print(f"{'Confidence':<15} {'Games':<10} {'ATS Accuracy':<15} {'Expected':<15}")
print("-" * 70)

results = []
for bucket in ['50-55%', '55-60%', '60-65%', '65-70%', '70%+']:
    subset = df[df['confidence_bucket'] == bucket]
    if len(subset) > 0:
        actual_acc = subset['ats_correct'].mean()
        expected_acc = subset['max_cover_prob'].mean()
        games = len(subset)
        
        results.append({
            'bucket': bucket,
            'games': games,
            'actual': actual_acc,
            'expected': expected_acc
        })
        
        print(f"{bucket:<15} {games:<10} {actual_acc:<14.1%} {expected_acc:<14.1%}")

print("\n" + "="*70)
print("HYPOTHESIS TEST: Is Calibration Inverted?")
print("="*70 + "\n")

# Compare highest confidence vs lowest confidence
if len(results) >= 2:
    lowest_conf = results[0]
    highest_conf = results[-1]
    
    print(f"Lowest confidence ({lowest_conf['bucket']}):")
    print(f"  Accuracy: {lowest_conf['actual']:.1%}")
    
    print(f"\nHighest confidence ({highest_conf['bucket']}):")
    print(f"  Accuracy: {highest_conf['actual']:.1%}")
    
    diff = highest_conf['actual'] - lowest_conf['actual']
    print(f"\nDifference: {diff:+.1%}")
    
    if diff < -0.02:  # More than 2% worse
        print("\n" + "="*70)
        print("RESULT: CALIBRATION IS INVERTED")
        print("="*70)
        print("\nHigh confidence picks perform WORSE than low confidence picks.")
        print("This is a severe calibration failure.")
        print("\nHypothesis CONFIRMED: Calibration is the problem.")
        print("\nRecommended action: Simplify calibration or train baseline model.")
        inverted = True
    elif diff < 0:
        print("\n" + "="*70)
        print("RESULT: CALIBRATION IS FLAT/SLIGHTLY INVERTED")
        print("="*70)
        print("\nHigh confidence picks don't perform better.")
        print("Calibration is not working as intended.")
        print("\nHypothesis PARTIALLY CONFIRMED.")
        inverted = True
    else:
        print("\n" + "="*70)
        print("RESULT: CALIBRATION IS NOT INVERTED")
        print("="*70)
        print("\nHigh confidence picks DO perform better (as expected).")
        print("Calibration is working correctly.")
        print("\nHypothesis REJECTED: Calibration is not the primary issue.")
        print("Need to investigate other factors (player features, etc.)")
        inverted = False

# Correlation check
corr = df['max_cover_prob'].corr(df['ats_correct'].astype(float))
print(f"\nCorrelation (confidence vs correctness): {corr:+.3f}")

if corr < 0:
    print("  -> NEGATIVE correlation: Higher confidence = Lower accuracy")
elif corr < 0.1:
    print("  -> WEAK correlation: Confidence doesn't predict accuracy")
else:
    print("  -> POSITIVE correlation: Confidence predicts accuracy (good)")

# Additional analysis: Compare to validation set
print("\n" + "="*70)
print("COMPARISON TO VALIDATION SET")
print("="*70 + "\n")

val_df = pd.read_csv('reports/val_with_players/per_game_predictions.csv')

# Calculate ATS correctness for validation
val_df['home_margin'] = val_df['actual_home'] - val_df['actual_away']
val_df['home_covers'] = val_df['home_margin'] + val_df['market_spread_home'] > 0
val_df['away_covers'] = val_df['home_margin'] + val_df['market_spread_home'] < 0
val_df['model_pick_home'] = val_df['cover_prob_home'] > val_df['cover_prob_away']
val_df['ats_correct'] = np.where(
    val_df['model_pick_home'],
    val_df['home_covers'],
    val_df['away_covers']
)

val_df['max_cover_prob'] = val_df[['cover_prob_home', 'cover_prob_away']].max(axis=1)
val_df['confidence_bucket'] = pd.cut(val_df['max_cover_prob'],
                                      bins=[0.5, 0.55, 0.6, 0.65, 0.7, 1.0],
                                      labels=['50-55%', '55-60%', '60-65%', '65-70%', '70%+'])

print("Validation Set Calibration:")
for bucket in ['50-55%', '55-60%', '60-65%', '65-70%', '70%+']:
    subset = val_df[val_df['confidence_bucket'] == bucket]
    if len(subset) > 0:
        actual_acc = subset['ats_correct'].mean()
        print(f"  {bucket}: {actual_acc:.1%} ({len(subset)} games)")

val_corr = val_df['max_cover_prob'].corr(val_df['ats_correct'].astype(float))
print(f"\nValidation correlation: {val_corr:+.3f}")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70 + "\n")

if inverted or corr < 0.1:
    print("RECOMMENDATION: Proceed to Option A (Baseline Model)")
    print("\nTrain a model without player features to test if they're the issue:")
    print("  python train.py --data data/games_train.jsonl --output artifacts/run_baseline")
    print("  python evaluate.py --data data/games_predict_2024_2025.jsonl \\")
    print("    --model artifacts/run_baseline --reports-dir reports/baseline_2425")
else:
    print("Calibration appears to work correctly.")
    print("Need to investigate other factors:")
    print("  - Player features quality")
    print("  - Distribution shift")
    print("  - Model complexity")
    print("\nRun full Phase 1-2 diagnostics for deeper analysis.")

print("\n" + "="*70)

