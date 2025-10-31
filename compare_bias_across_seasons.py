import pandas as pd

print("="*80)
print("HOME BIAS COMPARISON ACROSS SEASONS")
print("="*80)
print()

for season, file in [("24-25", "test_2425_OLD_fixed.csv"), ("25-26", "test_2526_OLD_fixed.csv")]:
    df = pd.read_csv(f'predictions/{file}')
    
    df['model_predicts_home_covers'] = (df['pred_margin_mu'] > -df['market_spread_home']).astype(int)
    df['model_vs_market'] = df['pred_margin_mu'] - (-df['market_spread_home'])
    
    print(f"{season} SEASON:")
    print(f"  Total games: {len(df)}")
    print(f"  Model predicts HOME to cover: {df['model_predicts_home_covers'].sum()} ({df['model_predicts_home_covers'].mean()*100:.1f}%)")
    print(f"  Model predicts AWAY to cover: {(1-df['model_predicts_home_covers']).sum()} ({(1-df['model_predicts_home_covers']).mean()*100:.1f}%)")
    print(f"  Average home bias: {df['model_vs_market'].mean():+.2f} points")
    print(f"  Predicted margin mean: {df['pred_margin_mu'].mean():.2f}")
    print(f"  Market spread mean: {df['market_spread_home'].mean():.2f}")
    print()

print("="*80)
print("DIAGNOSIS")
print("="*80)
print()

print("The 25-26 season shows EXTREME home bias compared to 24-25.")
print("This suggests a problem with the 25-26 data or processing.")
print()
print("Possible causes:")
print("  1. Missing/corrupted player data for 25-26")
print("  2. League normalization issues for 25-26")
print("  3. Incorrect feature extraction for current season")
print("  4. Small sample creating extreme baseline")
print()

