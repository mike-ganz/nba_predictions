import pandas as pd

df = pd.read_csv('predictions/logistic_2425_test.csv')

print("Home Cover Probability Distribution")
print("="*50)
print(f"Range: {df['cover_prob_home'].min():.3f} to {df['cover_prob_home'].max():.3f}")
print(f"\nDistribution:")
print(f"  Below 50%: {(df['cover_prob_home'] < 0.5).sum()} games ({(df['cover_prob_home'] < 0.5).sum()/len(df)*100:.1f}%)")
print(f"  Above 50%: {(df['cover_prob_home'] > 0.5).sum()} games ({(df['cover_prob_home'] > 0.5).sum()/len(df)*100:.1f}%)")
print(f"  Exactly 50%: {(df['cover_prob_home'] == 0.5).sum()} games")
print(f"\nMean: {df['cover_prob_home'].mean():.3f}")
print(f"Median: {df['cover_prob_home'].median():.3f}")

# Show some examples of low probabilities
print("\n" + "="*50)
print("Sample of games where model predicts AWAY to cover:")
print("="*50)
low_prob = df[df['cover_prob_home'] < 0.45].sort_values('cover_prob_home').head(10)
if len(low_prob) > 0:
    print(f"\n{'Date':<12} {'Away':<6} {'Home':<6} {'Spread':<8} {'Home Cover Prob':<18} {'Actual'}")
    print("-"*80)
    for _, row in low_prob.iterrows():
        actual = "Home" if row.get('actual_home_covers', -1) == 1 else "Away" if row.get('actual_home_covers', -1) == 0 else "?"
        print(f"{row['date']:<12} {row['away_team']:<6} {row['home_team']:<6} {row['market_spread_home']:>7.1f} {row['cover_prob_home']:>17.1%} {actual:>6}")
else:
    print("No games with home cover prob < 45%")

print("\n" + "="*50)
print("Histogram of probabilities:")
print("="*50)
bins = [0, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 1.0]
labels = ['<40%', '40-45%', '45-48%', '48-50%', '50-52%', '52-55%', '55-60%', '>60%']
df['prob_bin'] = pd.cut(df['cover_prob_home'], bins=bins, labels=labels)
print(df['prob_bin'].value_counts().sort_index())

