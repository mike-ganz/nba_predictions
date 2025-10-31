"""
Strategy analysis using USER's spread bucket definitions:
VERY LOW (0-2), LOW (2-5), MEDIUM (5-8), HIGH (8-12), VERY HIGH (12+)
"""
import pandas as pd
import numpy as np

print("="*80)
print("STRATEGY ANALYSIS WITH USER'S SPREAD BUCKETS")
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
    
    # USER'S SPREAD BUCKETS
    df['spread_bucket'] = 'UNKNOWN'
    df.loc[df['spread_abs'] >= 12, 'spread_bucket'] = '1. VERY HIGH'
    df.loc[(df['spread_abs'] >= 8) & (df['spread_abs'] < 12), 'spread_bucket'] = '2. HIGH'
    df.loc[(df['spread_abs'] >= 5) & (df['spread_abs'] < 8), 'spread_bucket'] = '3. MEDIUM'
    df.loc[(df['spread_abs'] >= 2) & (df['spread_abs'] < 5), 'spread_bucket'] = '4. LOW'
    df.loc[df['spread_abs'] < 2, 'spread_bucket'] = '5. VERY LOW'

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
    
    return {
        'name': name,
        'games': n,
        'wins': correct,
        'ats_pct': accuracy,
        'roi': roi,
    }

print("="*80)
print("COMPARISON: MY BUCKETS vs YOUR BUCKETS")
print("="*80)
print()

print("MY 3-BUCKET SYSTEM:")
print("  Small: ≤3.5 pts")
print("  Medium: 3.5-7.5 pts") 
print("  Large: >7.5 pts")
print()

print("YOUR 5-BUCKET SYSTEM:")
print("  VERY LOW: 0-2 pts")
print("  LOW: 2-5 pts")
print("  MEDIUM: 5-8 pts")
print("  HIGH: 8-12 pts")
print("  VERY HIGH: 12+ pts")
print()

print("MAPPING:")
print("  My 'Small' ≈ Your 'VERY LOW' + half of 'LOW'")
print("  My 'Medium' ≈ Half of your 'LOW' + 'MEDIUM' + part of 'HIGH'")
print("  My 'Large' ≈ Most of your 'HIGH' + all 'VERY HIGH'")
print()

# =============================================================================
# 24-25 ANALYSIS WITH USER'S BUCKETS
# =============================================================================
print("="*80)
print("24-25 SEASON: YOUR BUCKET PERFORMANCE")
print("="*80)
print()

print(f"{'Bucket':<20} {'Games':<10} {'ATS%':<12} {'ROI%':<12}")
print("-"*80)

for bucket in ['5. VERY LOW', '4. LOW', '3. MEDIUM', '2. HIGH', '1. VERY HIGH']:
    result = calculate_metrics(df_2425[df_2425['spread_bucket'] == bucket], bucket)
    if result:
        print(f"{bucket:<20} {result['games']:<10} {result['ats_pct']:>10.2f}% {result['roi']:>10.2f}%")

print()

# By favorite type AND bucket
print("="*80)
print("24-25: HOME FAVORITES BY YOUR BUCKETS")
print("="*80)
print()

print(f"{'Bucket':<20} {'Games':<10} {'ATS%':<12} {'ROI%':<12}")
print("-"*80)

for bucket in ['5. VERY LOW', '4. LOW', '3. MEDIUM', '2. HIGH', '1. VERY HIGH']:
    result = calculate_metrics(
        df_2425[(df_2425['home_favored']) & (df_2425['spread_bucket'] == bucket)], 
        bucket
    )
    if result:
        print(f"{bucket:<20} {result['games']:<10} {result['ats_pct']:>10.2f}% {result['roi']:>10.2f}%")

print()

print("="*80)
print("24-25: AWAY FAVORITES BY YOUR BUCKETS")
print("="*80)
print()

print(f"{'Bucket':<20} {'Games':<10} {'ATS%':<12} {'ROI%':<12}")
print("-"*80)

for bucket in ['5. VERY LOW', '4. LOW', '3. MEDIUM', '2. HIGH', '1. VERY HIGH']:
    result = calculate_metrics(
        df_2425[(df_2425['away_favored']) & (df_2425['spread_bucket'] == bucket)], 
        bucket
    )
    if result:
        print(f"{bucket:<20} {result['games']:<10} {result['ats_pct']:>10.2f}% {result['roi']:>10.2f}%")

print()

# =============================================================================
# 25-26 VALIDATION
# =============================================================================
print("="*80)
print("25-26 SEASON: YOUR BUCKET PERFORMANCE")
print("="*80)
print()

print(f"{'Bucket':<20} {'Games':<10} {'ATS%':<12} {'ROI%':<12}")
print("-"*80)

for bucket in ['5. VERY LOW', '4. LOW', '3. MEDIUM', '2. HIGH', '1. VERY HIGH']:
    result = calculate_metrics(df_2526[df_2526['spread_bucket'] == bucket], bucket)
    if result:
        print(f"{bucket:<20} {result['games']:<10} {result['ats_pct']:>10.2f}% {result['roi']:>10.2f}%")

print()

print("="*80)
print("25-26: HOME FAVORITES BY YOUR BUCKETS")
print("="*80)
print()

print(f"{'Bucket':<20} {'Games':<10} {'ATS%':<12} {'ROI%':<12}")
print("-"*80)

results_2526_home = []
for bucket in ['5. VERY LOW', '4. LOW', '3. MEDIUM', '2. HIGH', '1. VERY HIGH']:
    result = calculate_metrics(
        df_2526[(df_2526['home_favored']) & (df_2526['spread_bucket'] == bucket)], 
        bucket
    )
    if result:
        results_2526_home.append(result)
        print(f"{bucket:<20} {result['games']:<10} {result['ats_pct']:>10.2f}% {result['roi']:>10.2f}%")

print()

# =============================================================================
# IDENTIFY BEST STRATEGY WITH USER'S BUCKETS
# =============================================================================
print("="*80)
print("BEST STRATEGIES WITH YOUR BUCKET DEFINITIONS")
print("="*80)
print()

# Collect all home favorite strategies by bucket for 24-25
strategies_2425 = []
for bucket in ['5. VERY LOW', '4. LOW', '3. MEDIUM', '2. HIGH', '1. VERY HIGH']:
    result = calculate_metrics(
        df_2425[(df_2425['home_favored']) & (df_2425['spread_bucket'] == bucket)], 
        f"Home Fav + {bucket}"
    )
    if result:
        strategies_2425.append(result)

# Sort by ROI
strategies_2425.sort(key=lambda x: x['roi'], reverse=True)

print("TOP 3 HOME FAVORITE STRATEGIES (24-25):")
print()
print(f"{'Strategy':<30} {'Games':<10} {'ATS%':<12} {'ROI%':<12}")
print("-"*80)

for result in strategies_2425[:3]:
    print(f"{result['name']:<30} {result['games']:<10} {result['ats_pct']:>10.2f}% {result['roi']:>10.2f}%")

print()

# Find matching 25-26 performance
best_strategy = strategies_2425[0]
bucket_name = best_strategy['name'].replace('Home Fav + ', '')

result_2526 = calculate_metrics(
    df_2526[(df_2526['home_favored']) & (df_2526['spread_bucket'] == bucket_name)],
    bucket_name
)

print("="*80)
print("RECOMMENDED STRATEGY (WITH YOUR BUCKETS)")
print("="*80)
print()

print(f"STRATEGY: {best_strategy['name']}")
print()
print(f"24-25 Performance:")
print(f"  Games: {best_strategy['games']}")
print(f"  ATS%: {best_strategy['ats_pct']:.2f}%")
print(f"  ROI: {best_strategy['roi']:.2f}%")
print()

if result_2526:
    print(f"25-26 Validation:")
    print(f"  Games: {result_2526['games']}")
    print(f"  Wins: {result_2526['wins']}")
    print(f"  ATS%: {result_2526['ats_pct']:.2f}%")
    print(f"  ROI: {result_2526['roi']:.2f}%")
    print()
    
    diff = abs(result_2526['ats_pct'] - best_strategy['ats_pct'])
    if diff < 10:
        print(f"✓ Consistent performance (diff: {diff:.1f} pp)")
    else:
        print(f"⚠️  Performance varies (diff: {diff:.1f} pp - small sample)")
else:
    print("25-26 Validation: No games in this bucket")

print()
print("="*80)
print("KEY INSIGHT")
print("="*80)
print()
print("Your 5-bucket system provides MORE GRANULARITY:")
print("  ✓ Can identify specific sweet spots (e.g., 5-8 pt favorites)")
print("  ✓ Separates blowouts from competitive games better")
print("  ✓ More actionable for bet selection")
print()
print("My 3-bucket system is SIMPLER:")
print("  ✓ Larger sample sizes per bucket")
print("  ✓ Less risk of overfitting to noise")
print("  ✓ Easier to remember and apply")
print()
print("Both are valid - yours is better for detailed analysis!")
print("="*80)

