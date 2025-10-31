"""
Comprehensive strategy analysis on OLD model's 24-25 performance,
then validate the best strategy on 25-26.
"""
import pandas as pd
import numpy as np
from scipy import stats

print("="*80)
print("FINAL STRATEGY ANALYSIS - OLD MODEL")
print("="*80)
print()

# Load predictions
df_2425 = pd.read_csv('predictions/OLD_model_2425_predictions.csv')
df_2526 = pd.read_csv('predictions/OLD_model_2526_predictions.csv')

# Calculate ATS outcomes
for df in [df_2425, df_2526]:
    df['actual_home_covers'] = (df['actual_margin'] > -df['market_spread_home']).astype(int)
    df['model_predicts_home_covers'] = (df['pred_margin_mu'] > -df['market_spread_home']).astype(int)
    df['model_correct'] = (df['actual_home_covers'] == df['model_predicts_home_covers']).astype(int)
    
    # Market context
    df['home_favored'] = df['market_spread_home'] < 0
    df['away_favored'] = df['market_spread_home'] > 0
    df['spread_abs'] = df['market_spread_home'].abs()
    
    # Spread buckets
    df['spread_small'] = df['spread_abs'] <= 3.5
    df['spread_medium'] = (df['spread_abs'] > 3.5) & (df['spread_abs'] <= 7.5)
    df['spread_large'] = df['spread_abs'] > 7.5
    
    # Model confidence
    df['model_confidence'] = df[['cover_prob_home', 'cover_prob_away']].max(axis=1)
    df['high_confidence'] = df['model_confidence'] > 0.55
    
    # What model picks
    df['model_picks_home'] = df['model_predicts_home_covers'] == 1
    df['model_picks_away'] = df['model_predicts_home_covers'] == 0

def calculate_metrics(df, name):
    """Calculate ATS and ROI metrics"""
    n = len(df)
    if n == 0:
        return None
    
    correct = df['model_correct'].sum()
    accuracy = (correct / n) * 100
    
    # ROI calculation (assuming -110 odds)
    profit = (correct * 0.909) - (n - correct)
    roi = (profit / n) * 100
    
    # Statistical significance
    p_expected = 0.524  # Breakeven at -110
    z_score = (accuracy/100 - p_expected) / np.sqrt(p_expected * (1-p_expected) / n)
    
    return {
        'name': name,
        'games': n,
        'wins': correct,
        'ats_pct': accuracy,
        'roi': roi,
        'z_score': z_score
    }

# =============================================================================
# PHASE 1: COMPREHENSIVE ANALYSIS ON 24-25
# =============================================================================
print("="*80)
print("PHASE 1: STRATEGY ANALYSIS ON 24-25 SEASON")
print("="*80)
print()

strategies_2425 = []

# Overall
strategies_2425.append(calculate_metrics(df_2425, "Overall"))

# By favorite type
strategies_2425.append(calculate_metrics(
    df_2425[df_2425['home_favored']], "Home Favorites"
))
strategies_2425.append(calculate_metrics(
    df_2425[df_2425['away_favored']], "Away Favorites"
))

# By what model picks
strategies_2425.append(calculate_metrics(
    df_2425[df_2425['model_picks_home']], "When Model Picks Home"
))
strategies_2425.append(calculate_metrics(
    df_2425[df_2425['model_picks_away']], "When Model Picks Away"
))

# By spread size
strategies_2425.append(calculate_metrics(
    df_2425[df_2425['spread_small']], "Small Spread (≤3.5)"
))
strategies_2425.append(calculate_metrics(
    df_2425[df_2425['spread_medium']], "Medium Spread (3.5-7.5)"
))
strategies_2425.append(calculate_metrics(
    df_2425[df_2425['spread_large']], "Large Spread (>7.5)"
))

# Combined filters - Home Favorites
strategies_2425.append(calculate_metrics(
    df_2425[df_2425['home_favored'] & df_2425['model_picks_home']], 
    "Home Fav + Pick Home"
))
strategies_2425.append(calculate_metrics(
    df_2425[df_2425['home_favored'] & df_2425['model_picks_away']], 
    "Home Fav + Pick Away"
))

# Combined filters - Away Favorites
strategies_2425.append(calculate_metrics(
    df_2425[df_2425['away_favored'] & df_2425['model_picks_home']], 
    "Away Fav + Pick Home"
))
strategies_2425.append(calculate_metrics(
    df_2425[df_2425['away_favored'] & df_2425['model_picks_away']], 
    "Away Fav + Pick Away"
))

# Spread size combinations
strategies_2425.append(calculate_metrics(
    df_2425[df_2425['home_favored'] & df_2425['spread_medium']], 
    "Home Fav + Medium Spread"
))
strategies_2425.append(calculate_metrics(
    df_2425[df_2425['home_favored'] & df_2425['spread_large']], 
    "Home Fav + Large Spread"
))
strategies_2425.append(calculate_metrics(
    df_2425[df_2425['away_favored'] & df_2425['spread_medium']], 
    "Away Fav + Medium Spread"
))
strategies_2425.append(calculate_metrics(
    df_2425[df_2425['away_favored'] & df_2425['spread_large']], 
    "Away Fav + Large Spread"
))

# High confidence
strategies_2425.append(calculate_metrics(
    df_2425[df_2425['high_confidence']], 
    "High Confidence (>55%)"
))

# Convert to DataFrame
results_2425 = pd.DataFrame([s for s in strategies_2425 if s is not None])
results_2425 = results_2425.sort_values('roi', ascending=False)

print("TOP 10 STRATEGIES ON 24-25:")
print()
print(f"{'Strategy':<35} {'Games':<8} {'ATS%':<10} {'ROI%':<10} {'Z-Score':<10}")
print("-"*80)

for _, row in results_2425.head(10).iterrows():
    stars = "***" if row['z_score'] > 2.58 else ("**" if row['z_score'] > 1.96 else ("*" if row['z_score'] > 1.65 else ""))
    print(f"{row['name']:<35} {row['games']:<8.0f} {row['ats_pct']:>8.2f}% {row['roi']:>9.2f}% {row['z_score']:>9.2f} {stars}")

print()
print("Statistical significance: *** p<0.01, ** p<0.05, * p<0.10")
print()

# Identify best strategy
best_strategy = results_2425.iloc[0]
print(f"🏆 BEST STRATEGY ON 24-25: {best_strategy['name']}")
print(f"   {best_strategy['games']:.0f} games | {best_strategy['ats_pct']:.2f}% ATS | {best_strategy['roi']:.2f}% ROI")
print()

# =============================================================================
# PHASE 2: VALIDATE ON 25-26
# =============================================================================
print("="*80)
print("PHASE 2: VALIDATE BEST STRATEGY ON 25-26")
print("="*80)
print()

# Apply same filters to 25-26
strategies_2526 = []

for _, row_2425 in results_2425.head(5).iterrows():
    strategy_name = row_2425['name']
    
    # Determine filter based on name
    if strategy_name == "Overall":
        filtered_2526 = df_2526
    elif strategy_name == "Home Favorites":
        filtered_2526 = df_2526[df_2526['home_favored']]
    elif strategy_name == "Away Favorites":
        filtered_2526 = df_2526[df_2526['away_favored']]
    elif strategy_name == "When Model Picks Home":
        filtered_2526 = df_2526[df_2526['model_picks_home']]
    elif strategy_name == "When Model Picks Away":
        filtered_2526 = df_2526[df_2526['model_picks_away']]
    elif strategy_name == "Home Fav + Pick Home":
        filtered_2526 = df_2526[df_2526['home_favored'] & df_2526['model_picks_home']]
    elif strategy_name == "Home Fav + Pick Away":
        filtered_2526 = df_2526[df_2526['home_favored'] & df_2526['model_picks_away']]
    elif strategy_name == "Away Fav + Pick Home":
        filtered_2526 = df_2526[df_2526['away_favored'] & df_2526['model_picks_home']]
    elif strategy_name == "Away Fav + Pick Away":
        filtered_2526 = df_2526[df_2526['away_favored'] & df_2526['model_picks_away']]
    elif strategy_name == "Home Fav + Medium Spread":
        filtered_2526 = df_2526[df_2526['home_favored'] & df_2526['spread_medium']]
    elif strategy_name == "Home Fav + Large Spread":
        filtered_2526 = df_2526[df_2526['home_favored'] & df_2526['spread_large']]
    elif strategy_name == "Away Fav + Medium Spread":
        filtered_2526 = df_2526[df_2526['away_favored'] & df_2526['spread_medium']]
    elif strategy_name == "Away Fav + Large Spread":
        filtered_2526 = df_2526[df_2526['away_favored'] & df_2526['spread_large']]
    elif strategy_name == "High Confidence (>55%)":
        filtered_2526 = df_2526[df_2526['high_confidence']]
    elif "Small Spread" in strategy_name:
        filtered_2526 = df_2526[df_2526['spread_small']]
    elif "Medium Spread" in strategy_name:
        filtered_2526 = df_2526[df_2526['spread_medium']]
    elif "Large Spread" in strategy_name:
        filtered_2526 = df_2526[df_2526['spread_large']]
    else:
        continue
    
    result_2526 = calculate_metrics(filtered_2526, strategy_name)
    if result_2526:
        strategies_2526.append(result_2526)

print("TOP 5 STRATEGIES: 24-25 vs 25-26 COMPARISON")
print()
print(f"{'Strategy':<35} {'24-25 ATS%':<12} {'25-26 ATS%':<12} {'25-26 ROI%':<12} {'25-26 Games':<12}")
print("-"*80)

for i, row_2425 in results_2425.head(5).iterrows():
    strategy_name = row_2425['name']
    
    # Find matching 25-26 result
    matching_2526 = [s for s in strategies_2526 if s['name'] == strategy_name]
    
    if matching_2526:
        row_2526 = matching_2526[0]
        print(f"{strategy_name:<35} {row_2425['ats_pct']:>10.2f}% {row_2526['ats_pct']:>10.2f}% {row_2526['roi']:>10.2f}% {row_2526['games']:>11.0f}")
    else:
        print(f"{strategy_name:<35} {row_2425['ats_pct']:>10.2f}% {'N/A':>10} {'N/A':>10} {'0':>11}")

print()

# =============================================================================
# FINAL RECOMMENDATION
# =============================================================================
print("="*80)
print("FINAL RECOMMENDATION")
print("="*80)
print()

print(f"RECOMMENDED STRATEGY: {best_strategy['name']}")
print()

# Find 25-26 performance for best strategy
best_on_2526 = [s for s in strategies_2526 if s['name'] == best_strategy['name']]

if best_on_2526:
    perf_2526 = best_on_2526[0]
    
    print(f"24-25 Performance:")
    print(f"  Games: {best_strategy['games']:.0f}")
    print(f"  ATS%: {best_strategy['ats_pct']:.2f}%")
    print(f"  ROI: {best_strategy['roi']:.2f}%")
    print(f"  Z-score: {best_strategy['z_score']:.2f}")
    print()
    
    print(f"25-26 Performance (Validation):")
    print(f"  Games: {perf_2526['games']:.0f}")
    print(f"  Wins: {perf_2526['wins']:.0f}")
    print(f"  ATS%: {perf_2526['ats_pct']:.2f}%")
    print(f"  ROI: {perf_2526['roi']:.2f}%")
    print()
    
    # Calculate consistency
    diff = abs(perf_2526['ats_pct'] - best_strategy['ats_pct'])
    
    if diff < 5:
        print("✅ STRONG VALIDATION: Performance is consistent across both seasons!")
        print(f"   Difference: Only {diff:.1f} pp")
    elif diff < 10:
        print("✓ GOOD VALIDATION: Performance is reasonably consistent")
        print(f"   Difference: {diff:.1f} pp")
    else:
        print("⚠️  WEAK VALIDATION: Large performance difference between seasons")
        print(f"   Difference: {diff:.1f} pp (may be due to small 25-26 sample)")
    
    print()
    
    # Dollar analysis
    if perf_2526['games'] > 0:
        print("DOLLAR ANALYSIS (betting $100 per game):")
        print(f"  Total wagered: ${perf_2526['games'] * 100:.0f}")
        print(f"  Total profit: ${perf_2526['roi'] * perf_2526['games']:.0f}")
        print(f"  Return: ${100 + perf_2526['roi']:.2f} per $100 wagered")
        
        # 95% confidence interval
        n = perf_2526['games']
        p = perf_2526['ats_pct'] / 100
        se = np.sqrt(p * (1-p) / n)
        ci_lower = (p - 1.96 * se) * 100
        ci_upper = (p + 1.96 * se) * 100
        
        print()
        print(f"⚠️  SMALL SAMPLE WARNING:")
        print(f"   With only {n:.0f} games, the 95% confidence interval is:")
        print(f"   {ci_lower:.1f}% to {ci_upper:.1f}% ATS")
        print(f"   Need 300+ games for stable estimates")

else:
    print("⚠️  No 25-26 games match this strategy (may be too specific)")

print()
print("="*80)

