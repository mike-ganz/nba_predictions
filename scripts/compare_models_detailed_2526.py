"""
Detailed comparison of old vs new model on 2025-26 season.
Shows performance across all key segments.
"""
import pandas as pd
import numpy as np

def calculate_segment_metrics(df, segment_name):
    """Calculate ATS metrics for a segment."""
    if len(df) == 0:
        return None
    
    n_games = len(df)
    ats_acc = df['ats_correct'].mean() * 100
    
    # ROI
    wins = df['ats_correct'].sum()
    losses = n_games - wins
    profit = wins * 0.909 - losses * 1.1
    total_wagered = n_games * 1.1
    roi = (profit / total_wagered * 100) if total_wagered > 0 else 0.0
    
    return {
        'segment': segment_name,
        'n_games': n_games,
        'ats_acc': ats_acc,
        'roi': roi,
        'profit': profit,
    }

def analyze_model(pred_df, model_name):
    """Analyze a model's predictions with all cuts."""
    df = pred_df.copy()
    
    # Calculate ATS correctness
    df['actual_home_covers'] = (df['actual_margin'] > -df['market_spread_home']).astype(int)
    df['predicted_home_covers'] = (df['pred_margin_mu'] > -df['market_spread_home']).astype(int)
    df['ats_correct'] = (df['actual_home_covers'] == df['predicted_home_covers']).astype(int)
    
    # Create filtering variables
    df['home_favored'] = df['market_spread_home'] < 0
    df['away_favored'] = df['market_spread_home'] > 0
    
    df['model_picks_home'] = df['predicted_home_covers'] == 1
    df['model_picks_away'] = df['predicted_home_covers'] == 0
    
    df['spread_abs'] = df['market_spread_home'].abs()
    df['spread_small'] = df['spread_abs'] <= 3.5
    df['spread_medium'] = (df['spread_abs'] > 3.5) & (df['spread_abs'] <= 7.5)
    df['spread_large'] = df['spread_abs'] > 7.5
    
    results = []
    
    # Overall
    results.append(calculate_segment_metrics(df, 'Overall'))
    
    # By favorite type
    results.append(calculate_segment_metrics(df[df['home_favored']], 'Home Favored'))
    results.append(calculate_segment_metrics(df[df['away_favored']], 'Away Favored'))
    
    # By model prediction
    results.append(calculate_segment_metrics(df[df['model_picks_home']], 'Pick Home'))
    results.append(calculate_segment_metrics(df[df['model_picks_away']], 'Pick Away'))
    
    # By spread size
    results.append(calculate_segment_metrics(df[df['spread_small']], 'Small Spread (≤3.5)'))
    results.append(calculate_segment_metrics(df[df['spread_medium']], 'Medium Spread (3.5-7.5)'))
    results.append(calculate_segment_metrics(df[df['spread_large']], 'Large Spread (>7.5)'))
    
    # Favorite × Model Prediction
    results.append(calculate_segment_metrics(
        df[df['home_favored'] & df['model_picks_home']], 
        'Home Fav + Pick Home'
    ))
    results.append(calculate_segment_metrics(
        df[df['home_favored'] & df['model_picks_away']], 
        'Home Fav + Pick Away'
    ))
    results.append(calculate_segment_metrics(
        df[df['away_favored'] & df['model_picks_home']], 
        'Away Fav + Pick Home'
    ))
    results.append(calculate_segment_metrics(
        df[df['away_favored'] & df['model_picks_away']], 
        'Away Fav + Pick Away'
    ))
    
    # Favorite × Spread Size
    results.append(calculate_segment_metrics(
        df[df['home_favored'] & df['spread_small']], 
        'Home Fav + Small Spread'
    ))
    results.append(calculate_segment_metrics(
        df[df['home_favored'] & df['spread_medium']], 
        'Home Fav + Med Spread'
    ))
    results.append(calculate_segment_metrics(
        df[df['home_favored'] & df['spread_large']], 
        'Home Fav + Large Spread'
    ))
    results.append(calculate_segment_metrics(
        df[df['away_favored'] & df['spread_small']], 
        'Away Fav + Small Spread'
    ))
    results.append(calculate_segment_metrics(
        df[df['away_favored'] & df['spread_medium']], 
        'Away Fav + Med Spread'
    ))
    results.append(calculate_segment_metrics(
        df[df['away_favored'] & df['spread_large']], 
        'Away Fav + Large Spread'
    ))
    
    # Model Prediction × Spread Size
    results.append(calculate_segment_metrics(
        df[df['model_picks_home'] & df['spread_small']], 
        'Pick Home + Small Spread'
    ))
    results.append(calculate_segment_metrics(
        df[df['model_picks_home'] & df['spread_medium']], 
        'Pick Home + Med Spread'
    ))
    results.append(calculate_segment_metrics(
        df[df['model_picks_home'] & df['spread_large']], 
        'Pick Home + Large Spread'
    ))
    results.append(calculate_segment_metrics(
        df[df['model_picks_away'] & df['spread_small']], 
        'Pick Away + Small Spread'
    ))
    results.append(calculate_segment_metrics(
        df[df['model_picks_away'] & df['spread_medium']], 
        'Pick Away + Med Spread'
    ))
    results.append(calculate_segment_metrics(
        df[df['model_picks_away'] & df['spread_large']], 
        'Pick Away + Large Spread'
    ))
    
    # Filter out None results
    results = [r for r in results if r is not None]
    
    # Add model name to each result
    for r in results:
        r['model'] = model_name
    
    return pd.DataFrame(results)

# Load predictions
old_pred = pd.read_csv('predictions/current_season_2025_2026_predictions.csv')
new_pred = pd.read_csv('predictions/current_season_2025_2026_predictions_NEW_MODEL.csv')

print("="*100)
print("DETAILED MODEL COMPARISON: OLD (21-24) vs NEW (21-25)")
print("2025-26 Season (72 games)")
print("="*100)
print()

# Analyze both models
old_results = analyze_model(old_pred, 'Old (21-24)')
new_results = analyze_model(new_pred, 'New (21-25)')

# Merge results for comparison
merged = old_results.merge(new_results, on='segment', suffixes=('_old', '_new'))

# Sort by old model ROI
merged = merged.sort_values('roi_old', ascending=False)

print("="*100)
print(f"{'Segment':<35} {'Games':<8} {'Old ATS%':<12} {'Old ROI%':<12} {'New ATS%':<12} {'New ROI%':<12} {'Δ ATS%':<10}")
print("="*100)

for _, row in merged.iterrows():
    segment = row['segment']
    n_games = row['n_games_old']
    old_ats = row['ats_acc_old']
    old_roi = row['roi_old']
    new_ats = row['ats_acc_new']
    new_roi = row['roi_new']
    delta_ats = new_ats - old_ats
    
    # Color coding (text only)
    winner = "→" if abs(delta_ats) < 1 else ("🟢" if delta_ats > 0 else "🔴")
    
    print(f"{segment:<35} {n_games:<8} {old_ats:>10.2f}% {old_roi:>11.2f}% {new_ats:>11.2f}% {new_roi:>11.2f}% {delta_ats:>9.2f}%")

print("="*100)
print()

# Highlight key differences
print("="*100)
print("KEY DIFFERENCES (Segments where models diverge most)")
print("="*100)
print()

merged['ats_diff'] = (merged['ats_acc_new'] - merged['ats_acc_old']).abs()
top_diffs = merged.nlargest(10, 'ats_diff')[['segment', 'n_games_old', 'ats_acc_old', 'ats_acc_new', 'ats_diff']]

print(f"{'Segment':<35} {'Games':<8} {'Old ATS%':<12} {'New ATS%':<12} {'Difference':<12}")
print("-"*100)
for _, row in top_diffs.iterrows():
    print(f"{row['segment']:<35} {row['n_games_old']:<8} {row['ats_acc_old']:>10.2f}% {row['ats_acc_new']:>10.2f}% {row['ats_diff']:>10.2f}%")

print()

# Best strategies for each model
print("="*100)
print("TOP 5 STRATEGIES FOR EACH MODEL (Min 10 games)")
print("="*100)
print()

min_games = 10
qualified = merged[merged['n_games_old'] >= min_games]

print("OLD MODEL (21-24) - Best ROI:")
print("-"*100)
old_top5 = qualified.nlargest(5, 'roi_old')[['segment', 'n_games_old', 'ats_acc_old', 'roi_old']]
for idx, row in old_top5.iterrows():
    print(f"{row['segment']:<35} {row['n_games_old']:<8} {row['ats_acc_old']:>10.2f}% {row['roi_old']:>11.2f}%")

print()
print("NEW MODEL (21-25) - Best ROI:")
print("-"*100)
new_top5 = qualified.nlargest(5, 'roi_new')[['segment', 'n_games_new', 'ats_acc_new', 'roi_new']]
for idx, row in new_top5.iterrows():
    print(f"{row['segment']:<35} {row['n_games_new']:<8} {row['ats_acc_new']:>10.2f}% {row['roi_new']:>11.2f}%")

print()

# Save full results
merged.to_csv('predictions/model_comparison_detailed_2526.csv', index=False)
print("="*100)
print("✅ Full results saved to: predictions/model_comparison_detailed_2526.csv")
print("="*100)

