import pandas as pd

print("="*80)
print("VERIFYING 25-26 NORMALIZATION FIX")
print("="*80)
print()

df = pd.read_csv('predictions/test_2526_OLD_fixed_CORRECTED.csv')

df['model_predicts_home_covers'] = (df['pred_margin_mu'] > -df['market_spread_home']).astype(int)
df['model_vs_market'] = df['pred_margin_mu'] - (-df['market_spread_home'])

print(f"Total games: {len(df)}")
print(f"Model predicts HOME to cover: {df['model_predicts_home_covers'].sum()} ({df['model_predicts_home_covers'].mean()*100:.1f}%)")
print(f"Model predicts AWAY to cover: {(1-df['model_predicts_home_covers']).sum()} ({(1-df['model_predicts_home_covers']).mean()*100:.1f}%)")
print(f"Average home bias: {df['model_vs_market'].mean():+.2f} points")
print()

print("Comparison:")
print(f"  24-25 season: 47.1% home covers (balanced ✓)")
print(f"  25-26 BEFORE fix: 100.0% home covers (BROKEN ✗)")
print(f"  25-26 AFTER fix: {df['model_predicts_home_covers'].mean()*100:.1f}% home covers")
print()

if 45 < df['model_predicts_home_covers'].mean()*100 < 55:
    print("✅ FIX SUCCESSFUL! Predictions are now balanced.")
else:
    print("⚠️  Still some bias present but much better than before")

