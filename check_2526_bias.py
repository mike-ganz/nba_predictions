import pandas as pd

df = pd.read_csv('predictions/test_2526_OLD_fixed.csv')

print("="*80)
print("25-26 PREDICTION ANALYSIS")
print("="*80)
print()

# Check predictions
df['model_predicts_home_covers'] = (df['pred_margin_mu'] > -df['market_spread_home']).astype(int)

print(f"Total games: {len(df)}")
print(f"Model predicts HOME to cover: {df['model_predicts_home_covers'].sum()} ({df['model_predicts_home_covers'].mean()*100:.1f}%)")
print(f"Model predicts AWAY to cover: {(1-df['model_predicts_home_covers']).sum()} ({(1-df['model_predicts_home_covers']).mean()*100:.1f}%)")
print()

print("="*80)
print("PREDICTION DISTRIBUTION ANALYSIS")
print("="*80)
print()

print("Predicted margin (mu) distribution:")
print(df['pred_margin_mu'].describe())
print()

print("Market spread distribution:")
print(df['market_spread_home'].describe())
print()

print("Difference (pred_margin - (-spread)) distribution:")
df['pred_minus_threshold'] = df['pred_margin_mu'] - (-df['market_spread_home'])
print(df['pred_minus_threshold'].describe())
print()

print("="*80)
print("SYSTEMATIC BIAS CHECK")
print("="*80)
print()

# The model should predict roughly 50/50 if unbiased (or close to market)
# Predicting 100% home covers suggests:
# 1. Model is systematically more bullish on home teams than market
# 2. Or there's a bug in prediction logic

print("If model predicts home to cover 100% of the time:")
print("  → Model is ALWAYS more bullish on home than the market")
print("  → This suggests a systematic HOME BIAS")
print()

# Check: what's the average difference between model and market?
df['model_vs_market'] = df['pred_margin_mu'] - (-df['market_spread_home'])
print(f"Average (model prediction) - (market expectation): {df['model_vs_market'].mean():.2f} points")
print(f"  This means model is on average {df['model_vs_market'].mean():.2f} points MORE BULLISH on home than market")
print()

print("="*80)
print("GAMES CLOSEST TO NOT PREDICTING HOME COVER")
print("="*80)
print()

df_sorted = df.sort_values('pred_minus_threshold')
print("Top 10 games where model almost predicted AWAY to cover:")
print()
for idx, row in df_sorted.head(10).iterrows():
    print(f"{row['home_team']} vs {row['away_team']}")
    print(f"  Market: Home {row['market_spread_home']:+.1f} | Model: {row['pred_margin_mu']:+.1f}")
    print(f"  Home covers if: margin > {-row['market_spread_home']:.1f}")
    print(f"  Model says: {row['pred_margin_mu']:.1f} > {-row['market_spread_home']:.1f}? Gap: {row['pred_minus_threshold']:.2f}")
    print()

