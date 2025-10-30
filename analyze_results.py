"""Comprehensive analysis of model evaluation results."""
import pandas as pd
import numpy as np

def analyze_predictions(csv_path, name):
    """Analyze predictions from a per_game_predictions.csv file."""
    df = pd.read_csv(csv_path)
    
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}\n")
    
    # Basic info
    print(f"Total Games: {len(df)}")
    print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
    if 'season' in df.columns:
        print(f"Seasons: {', '.join(df['season'].unique())}")
    print()
    
    # 1. MONEYLINE (Winner Prediction)
    df['pred_winner'] = (df['pred_home'] > df['pred_away']).astype(int)
    df['actual_winner'] = (df['actual_home'] > df['actual_away']).astype(int)
    df['ml_correct'] = (df['pred_winner'] == df['actual_winner'])
    ml_acc = df['ml_correct'].mean()
    
    print(f"MONEYLINE (Winner Prediction)")
    print(f"  Accuracy: {ml_acc:.1%} ({df['ml_correct'].sum()}/{len(df)})")
    print()
    
    # 2. AGAINST THE SPREAD (ATS)
    df['home_margin'] = df['actual_home'] - df['actual_away']
    df['home_covers'] = df['home_margin'] + df['market_spread_home'] > 0
    df['away_covers'] = df['home_margin'] + df['market_spread_home'] < 0
    df['push'] = df['home_margin'] + df['market_spread_home'] == 0
    
    # Model picks the side with higher cover probability
    df['model_pick_home'] = df['cover_prob_home'] > df['cover_prob_away']
    df['ats_correct'] = np.where(
        df['model_pick_home'],
        df['home_covers'],
        df['away_covers']
    )
    
    ats_acc = df['ats_correct'].mean()
    ats_count = df['ats_correct'].sum()
    
    print(f"AGAINST THE SPREAD (ATS)")
    print(f"  Accuracy: {ats_acc:.1%} ({ats_count}/{len(df)})")
    print(f"  Breakeven: 52.4% (need to beat -110 odds)")
    print(f"  Edge over breakeven: {(ats_acc - 0.524)*100:+.1f}%")
    
    # Calculate ROI assuming -110 odds (risk 1.1 to win 1.0)
    df['ats_profit'] = np.where(df['ats_correct'], 1.0, -1.1)
    total_wagered = len(df) * 1.1
    total_profit = df['ats_profit'].sum()
    roi = (total_profit / total_wagered) * 100
    
    print(f"  ROI (flat betting): {roi:+.2f}%")
    print(f"  Profit on ${total_wagered:.0f} wagered: ${total_profit:+.2f}")
    print()
    
    # 3. TOTALS (Over/Under)
    df['predicted_total'] = df['pred_home'] + df['pred_away']
    df['actual_total'] = df['actual_home'] + df['actual_away']
    
    df['model_pick_over'] = df['predicted_total'] > df['market_total']
    df['actual_over'] = df['actual_total'] > df['market_total']
    
    df['total_correct'] = df['model_pick_over'] == df['actual_over']
    total_acc = df['total_correct'].mean()
    
    print(f"TOTALS (Over/Under)")
    print(f"  Accuracy: {total_acc:.1%} ({df['total_correct'].sum()}/{len(df)})")
    
    # Total ROI
    df['total_profit'] = np.where(df['total_correct'], 1.0, -1.1)
    total_profit_ou = df['total_profit'].sum()
    roi_ou = (total_profit_ou / total_wagered) * 100
    print(f"  ROI (flat betting): {roi_ou:+.2f}%")
    print()
    
    # 4. PREDICTION ERRORS
    df['margin_error'] = abs(df['home_margin'] - (df['pred_home'] - df['pred_away']))
    df['total_error'] = abs(df['actual_total'] - df['predicted_total'])
    
    print(f"PREDICTION ERRORS")
    print(f"  Mean Margin Error: {df['margin_error'].mean():.2f} points")
    print(f"  Median Margin Error: {df['margin_error'].median():.2f} points")
    print(f"  Mean Total Error: {df['total_error'].mean():.2f} points")
    print(f"  Median Total Error: {df['total_error'].median():.2f} points")
    print()
    
    # 5. CALIBRATION / CONFIDENCE
    if 'cover_prob_home' in df.columns:
        # High confidence picks (>60% cover probability)
        high_conf = df[df['cover_prob_home'].apply(lambda x: max(x, 1-x) > 0.6)]
        if len(high_conf) > 0:
            high_conf_acc = high_conf['ats_correct'].mean()
            print(f"HIGH CONFIDENCE PICKS (>60% cover prob)")
            print(f"  Count: {len(high_conf)}")
            print(f"  ATS Accuracy: {high_conf_acc:.1%}")
            
            hc_profit = high_conf['ats_profit'].sum()
            hc_wagered = len(high_conf) * 1.1
            hc_roi = (hc_profit / hc_wagered) * 100
            print(f"  ROI: {hc_roi:+.2f}%")
            print()
    
    # 6. HOME vs AWAY PERFORMANCE
    home_picks = df[df['model_pick_home']]
    away_picks = df[~df['model_pick_home']]
    
    print(f"PICK DISTRIBUTION")
    print(f"  Home picks: {len(home_picks)} ({len(home_picks)/len(df)*100:.1f}%)")
    print(f"    Accuracy: {home_picks['ats_correct'].mean():.1%}")
    print(f"  Away picks: {len(away_picks)} ({len(away_picks)/len(df)*100:.1f}%)")
    print(f"    Accuracy: {away_picks['ats_correct'].mean():.1%}")
    print()
    
    # 7. BY SEASON (if multiple seasons)
    if 'season' in df.columns and df['season'].nunique() > 1:
        print(f"PERFORMANCE BY SEASON")
        for season in sorted(df['season'].unique()):
            season_df = df[df['season'] == season]
            season_ats = season_df['ats_correct'].mean()
            season_ml = season_df['ml_correct'].mean()
            print(f"  {season}:")
            print(f"    Games: {len(season_df)}")
            print(f"    ATS: {season_ats:.1%}, ML: {season_ml:.1%}")
        print()
    
    return df

# Analyze both reports
print("\n" + "="*70)
print("NBA PREDICTIONS MODEL EVALUATION RESULTS")
print("="*70)

val_df = analyze_predictions('reports/val_with_players/per_game_predictions.csv', 
                              'VALIDATION SET (Historical: 2021-2024)')

pred_df = analyze_predictions('reports/2425_with_players/per_game_predictions.csv',
                               '2024-2025 SEASON (Current/Future)')

# Summary comparison
print("\n" + "="*70)
print("SUMMARY COMPARISON")
print("="*70 + "\n")

val_ats = val_df['ats_correct'].mean()
val_ml = val_df['ml_correct'].mean()
val_total = val_df['total_correct'].mean()

pred_ats = pred_df['ats_correct'].mean()
pred_ml = pred_df['ml_correct'].mean()
pred_total = pred_df['total_correct'].mean()

print(f"{'Metric':<25} {'Validation':<15} {'2024-2025':<15} {'Change'}")
print(f"{'-'*70}")
print(f"{'ATS Accuracy':<25} {val_ats:>14.1%} {pred_ats:>14.1%} {pred_ats-val_ats:>+14.1%}")
print(f"{'Moneyline Accuracy':<25} {val_ml:>14.1%} {pred_ml:>14.1%} {pred_ml-val_ml:>+14.1%}")
print(f"{'Total Accuracy':<25} {val_total:>14.1%} {pred_total:>14.1%} {pred_total-val_total:>+14.1%}")

val_roi = ((val_df['ats_profit'].sum() / (len(val_df) * 1.1)) * 100)
pred_roi = ((pred_df['ats_profit'].sum() / (len(pred_df) * 1.1)) * 100)
print(f"{'ATS ROI':<25} {val_roi:>13.2f}% {pred_roi:>13.2f}% {pred_roi-val_roi:>+13.2f}%")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70 + "\n")

if pred_ats > 0.524:
    print("✓ Model beats breakeven (52.4%) on 2024-2025 season")
    print(f"  Edge: {(pred_ats - 0.524)*100:.1f} percentage points")
else:
    print("✗ Model below breakeven (52.4%) on 2024-2025 season")
    print(f"  Shortfall: {(pred_ats - 0.524)*100:.1f} percentage points")

print()

if pred_roi > 0:
    print(f"✓ Positive ROI on 2024-2025: {pred_roi:+.2f}%")
else:
    print(f"✗ Negative ROI on 2024-2025: {pred_roi:+.2f}%")

print()

if abs(pred_ats - val_ats) < 0.02:
    print("✓ Performance stable between validation and 2024-2025 (<2% difference)")
else:
    print(f"⚠ Performance shift: {(pred_ats - val_ats)*100:+.1f}% between validation and 2024-2025")
    if pred_ats > val_ats:
        print("  Model performing BETTER on 2024-2025 (possible overfitting or market shift)")
    else:
        print("  Model performing WORSE on 2024-2025 (possible underfitting or regime change)")

print()
print("="*70)

