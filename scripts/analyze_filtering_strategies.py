"""
Comprehensive filtering strategy analysis

Tests combinations of:
- Home vs away favored (by market)
- Model prediction (home vs away to cover)
- Spread size buckets
"""

import pandas as pd
import numpy as np
from itertools import product

print("=" * 80)
print("FILTERING STRATEGY ANALYSIS")
print("=" * 80)

# Load uncalibrated predictions
df = pd.read_csv('predictions/test_2425_normalized_predictions.csv')

# Calculate outcomes
df['actual_margin'] = df['actual_home'] - df['actual_away']
df['actual_home_covers'] = (df['actual_margin'] > -df['market_spread_home']).astype(int)
df['pred_home_covers'] = (df['cover_prob_home'] > 0.5).astype(int)
df['ats_correct'] = (df['pred_home_covers'] == df['actual_home_covers']).astype(int)

# Create filtering variables
df['home_favored'] = df['market_spread_home'] > 0
df['away_favored'] = df['market_spread_home'] < 0
df['pick_n_toss'] = df['market_spread_home'] == 0

df['model_picks_home'] = df['pred_home_covers'] == 1
df['model_picks_away'] = df['pred_home_covers'] == 0

df['spread_abs'] = df['market_spread_home'].abs()
df['spread_tiny'] = df['spread_abs'] < 2.5
df['spread_small'] = (df['spread_abs'] >= 2.5) & (df['spread_abs'] < 5.5)
df['spread_medium'] = (df['spread_abs'] >= 5.5) & (df['spread_abs'] < 8.5)
df['spread_large'] = df['spread_abs'] >= 8.5

# Confidence levels
df['confidence'] = df['cover_prob_home'].apply(lambda p: max(p, 1-p))
df['conf_low'] = (df['confidence'] >= 0.50) & (df['confidence'] < 0.55)
df['conf_medium'] = (df['confidence'] >= 0.55) & (df['confidence'] < 0.65)
df['conf_high'] = (df['confidence'] >= 0.65) & (df['confidence'] < 0.75)
df['conf_very_high'] = df['confidence'] >= 0.75

print(f"\nTotal games: {len(df)}")
print(f"Overall ATS: {df['ats_correct'].mean()*100:.2f}%")
print(f"Overall ROI: {df.apply(lambda r: 0.91 if r['ats_correct'] else -1.0, axis=1).mean()*100:+.2f}%")

# Helper function to evaluate a filter
def evaluate_filter(filtered_df, name):
    if len(filtered_df) == 0:
        return None
    
    n_games = len(filtered_df)
    ats_acc = filtered_df['ats_correct'].mean() * 100
    roi = filtered_df.apply(lambda r: 0.91 if r['ats_correct'] else -1.0, axis=1).mean() * 100
    
    # Statistical significance (vs 50%)
    n_correct = filtered_df['ats_correct'].sum()
    expected = n_games * 0.5
    z_score = (n_correct - expected) / np.sqrt(n_games * 0.5 * 0.5)
    
    return {
        'filter': name,
        'n_games': n_games,
        'pct_of_total': n_games / len(df) * 100,
        'ats_acc': ats_acc,
        'roi': roi,
        'z_score': z_score,
        'profit_per_100': roi * n_games / 100  # Total profit if betting $100/game
    }

results = []

print("\n" + "=" * 80)
print("1. FAVORITE TYPE ANALYSIS")
print("=" * 80)

for fav_type, label in [('home_favored', 'Home Favored'), 
                         ('away_favored', 'Away Favored'),
                         ('pick_n_toss', 'Pick\'em')]:
    filtered = df[df[fav_type]]
    result = evaluate_filter(filtered, label)
    if result:
        results.append(result)
        print(f"\n{label}:")
        print(f"  Games: {result['n_games']} ({result['pct_of_total']:.1f}%)")
        print(f"  ATS: {result['ats_acc']:.2f}%")
        print(f"  ROI: {result['roi']:+.2f}%")
        print(f"  Profit ($100/game): ${result['profit_per_100']:,.0f}")

print("\n" + "=" * 80)
print("2. MODEL PREDICTION ANALYSIS")
print("=" * 80)

for pred_type, label in [('model_picks_home', 'Model Picks Home'),
                          ('model_picks_away', 'Model Picks Away')]:
    filtered = df[df[pred_type]]
    result = evaluate_filter(filtered, label)
    if result:
        results.append(result)
        print(f"\n{label}:")
        print(f"  Games: {result['n_games']} ({result['pct_of_total']:.1f}%)")
        print(f"  ATS: {result['ats_acc']:.2f}%")
        print(f"  ROI: {result['roi']:+.2f}%")

print("\n" + "=" * 80)
print("3. SPREAD SIZE ANALYSIS")
print("=" * 80)

for spread_type, label in [('spread_tiny', 'Tiny (<2.5)'),
                            ('spread_small', 'Small (2.5-5.5)'),
                            ('spread_medium', 'Medium (5.5-8.5)'),
                            ('spread_large', 'Large (8.5+)')]:
    filtered = df[df[spread_type]]
    result = evaluate_filter(filtered, label)
    if result:
        results.append(result)
        print(f"\n{label}:")
        print(f"  Games: {result['n_games']} ({result['pct_of_total']:.1f}%)")
        print(f"  ATS: {result['ats_acc']:.2f}%")
        print(f"  ROI: {result['roi']:+.2f}%")

print("\n" + "=" * 80)
print("4. CONFIDENCE LEVEL ANALYSIS")
print("=" * 80)

for conf_type, label in [('conf_low', 'Low (50-55%)'),
                          ('conf_medium', 'Medium (55-65%)'),
                          ('conf_high', 'High (65-75%)'),
                          ('conf_very_high', 'Very High (75%+)')]:
    filtered = df[df[conf_type]]
    result = evaluate_filter(filtered, label)
    if result:
        results.append(result)
        print(f"\n{label}:")
        print(f"  Games: {result['n_games']} ({result['pct_of_total']:.1f}%)")
        print(f"  ATS: {result['ats_acc']:.2f}%")
        print(f"  ROI: {result['roi']:+.2f}%")

print("\n" + "=" * 80)
print("5. TWO-WAY COMBINATIONS")
print("=" * 80)

print("\n--- FAVORITE TYPE × MODEL PREDICTION ---")
for fav in ['home_favored', 'away_favored']:
    for pred in ['model_picks_home', 'model_picks_away']:
        filtered = df[df[fav] & df[pred]]
        fav_label = 'Home Fav' if fav == 'home_favored' else 'Away Fav'
        pred_label = 'Pick Home' if pred == 'model_picks_home' else 'Pick Away'
        result = evaluate_filter(filtered, f"{fav_label} + {pred_label}")
        if result and result['n_games'] >= 20:
            results.append(result)
            print(f"\n{fav_label} + {pred_label}:")
            print(f"  Games: {result['n_games']} ({result['pct_of_total']:.1f}%)")
            print(f"  ATS: {result['ats_acc']:.2f}%")
            print(f"  ROI: {result['roi']:+.2f}%")
            print(f"  Profit: ${result['profit_per_100']:,.0f}")

print("\n--- FAVORITE TYPE × SPREAD SIZE ---")
for fav in ['home_favored', 'away_favored']:
    for spread in ['spread_small', 'spread_medium', 'spread_large']:
        filtered = df[df[fav] & df[spread]]
        fav_label = 'Home Fav' if fav == 'home_favored' else 'Away Fav'
        spread_labels = {'spread_small': 'Small Spread', 
                        'spread_medium': 'Med Spread',
                        'spread_large': 'Large Spread'}
        result = evaluate_filter(filtered, f"{fav_label} + {spread_labels[spread]}")
        if result and result['n_games'] >= 20:
            results.append(result)
            print(f"\n{fav_label} + {spread_labels[spread]}:")
            print(f"  Games: {result['n_games']} ({result['pct_of_total']:.1f}%)")
            print(f"  ATS: {result['ats_acc']:.2f}%")
            print(f"  ROI: {result['roi']:+.2f}%")

print("\n--- MODEL PREDICTION × CONFIDENCE ---")
for pred in ['model_picks_home', 'model_picks_away']:
    for conf in ['conf_high', 'conf_very_high']:
        filtered = df[df[pred] & df[conf]]
        pred_label = 'Pick Home' if pred == 'model_picks_home' else 'Pick Away'
        conf_labels = {'conf_high': 'High Conf', 'conf_very_high': 'Very High Conf'}
        result = evaluate_filter(filtered, f"{pred_label} + {conf_labels[conf]}")
        if result and result['n_games'] >= 10:
            results.append(result)
            print(f"\n{pred_label} + {conf_labels[conf]}:")
            print(f"  Games: {result['n_games']} ({result['pct_of_total']:.1f}%)")
            print(f"  ATS: {result['ats_acc']:.2f}%")
            print(f"  ROI: {result['roi']:+.2f}%")

print("\n" + "=" * 80)
print("6. THREE-WAY COMBINATIONS (High-Confidence Only)")
print("=" * 80)

for fav in ['home_favored', 'away_favored']:
    for pred in ['model_picks_home', 'model_picks_away']:
        for spread in ['spread_small', 'spread_medium', 'spread_large']:
            # Only look at medium+ confidence
            filtered = df[df[fav] & df[pred] & df[spread] & (df['confidence'] >= 0.60)]
            
            fav_label = 'Home Fav' if fav == 'home_favored' else 'Away Fav'
            pred_label = 'Pick Home' if pred == 'model_picks_home' else 'Pick Away'
            spread_labels = {'spread_small': 'Small', 'spread_medium': 'Med', 'spread_large': 'Large'}
            
            result = evaluate_filter(filtered, f"{fav_label} + {pred_label} + {spread_labels[spread]} Spread")
            
            if result and result['n_games'] >= 15 and result['roi'] > 5:  # Only show promising ones
                results.append(result)
                print(f"\n{fav_label} + {pred_label} + {spread_labels[spread]} Spread:")
                print(f"  Games: {result['n_games']} ({result['pct_of_total']:.1f}%)")
                print(f"  ATS: {result['ats_acc']:.2f}%")
                print(f"  ROI: {result['roi']:+.2f}%")
                print(f"  Profit: ${result['profit_per_100']:,.0f}")
                print(f"  Z-score: {result['z_score']:.2f}")

print("\n" + "=" * 80)
print("TOP 10 STRATEGIES BY ROI")
print("=" * 80)

results_df = pd.DataFrame(results)
results_df = results_df[results_df['n_games'] >= 20]  # Min 20 games
top_10 = results_df.nlargest(10, 'roi')

print(f"\n{'Rank':<6} {'Strategy':<45} {'Games':<8} {'ATS%':<8} {'ROI%':<8} {'Profit':<12}")
print("-" * 95)
for i, row in enumerate(top_10.itertuples(), 1):
    print(f"{i:<6} {row.filter[:43]:<45} {row.n_games:<8} {row.ats_acc:>6.2f}%  {row.roi:>+6.2f}%  ${row.profit_per_100:>9,.0f}")

print("\n" + "=" * 80)
print("TOP 10 STRATEGIES BY TOTAL PROFIT")
print("=" * 80)

top_10_profit = results_df.nlargest(10, 'profit_per_100')

print(f"\n{'Rank':<6} {'Strategy':<45} {'Games':<8} {'ATS%':<8} {'ROI%':<8} {'Profit':<12}")
print("-" * 95)
for i, row in enumerate(top_10_profit.itertuples(), 1):
    print(f"{i:<6} {row.filter[:43]:<45} {row.n_games:<8} {row.ats_acc:>6.2f}%  {row.roi:>+6.2f}%  ${row.profit_per_100:>9,.0f}")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

# Find best strategy balancing volume and ROI
best_overall = results_df[(results_df['roi'] > 5) & (results_df['n_games'] >= 50)].nlargest(1, 'profit_per_100')

if len(best_overall) > 0:
    best = best_overall.iloc[0]
    print(f"\nBest Strategy: {best['filter']}")
    print(f"  Games: {int(best['n_games'])} ({best['pct_of_total']:.1f}% of season)")
    print(f"  ATS Accuracy: {best['ats_acc']:.2f}%")
    print(f"  ROI: {best['roi']:+.2f}%")
    print(f"  Expected Profit ($100/game): ${best['profit_per_100']:,.0f}")
    print(f"  Statistical Significance: z={best['z_score']:.2f}")
else:
    print("\nNo single strategy with 50+ games and 5%+ ROI found.")
    print("Consider combining multiple filters or accepting smaller sample sizes.")

# Save results
results_df.to_csv('analysis/filtering_strategies.csv', index=False)
print(f"\nFull results saved to: analysis/filtering_strategies.csv")

print("\n" + "=" * 80)

