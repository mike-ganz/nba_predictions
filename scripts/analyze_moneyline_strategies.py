"""
Comprehensive Moneyline (ML) Strategy Analysis

Analyzes ML betting performance across various filters.
Unlike ATS, ROI is calculated using actual moneyline odds.
"""

import pandas as pd
import numpy as np

print("=" * 100)
print("MONEYLINE BETTING STRATEGY ANALYSIS")
print("=" * 100)

# Load predictions
df = pd.read_csv('predictions/test_2425_normalized_predictions.csv')

# Load moneyline odds from raw JSONL
import json
ml_data = []
with open('data/games_predict_2024_2025_with_players_norm.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        game = json.loads(line)
        ml_data.append({
            'game_id': game['game_id'],
            'moneyline_home': game['market']['moneyline_home'],
            'moneyline_away': game['market']['moneyline_away']
        })

df_ml = pd.DataFrame(ml_data)
df = df.merge(df_ml, on='game_id', how='left')

# Calculate outcomes
df['actual_margin'] = df['actual_home'] - df['actual_away']
df['actual_home_wins'] = (df['actual_margin'] > 0).astype(int)
df['pred_home_wins'] = (df['win_prob_home'] > 0.5).astype(int)
df['ml_correct'] = (df['pred_home_wins'] == df['actual_home_wins']).astype(int)

# Calculate ML profit function
def calc_ml_profit(row):
    """Calculate profit/loss for a $100 ML bet based on model's pick"""
    if row['pred_home_wins']:
        # Model picks home
        if row['actual_home_wins']:
            # Win: pay based on home ML odds
            if row['moneyline_home'] > 0:
                return row['moneyline_home']  # Underdog win: +150 = win $150
            else:
                return 100 / abs(row['moneyline_home']) * 100  # Favorite win: -150 = win $66.67
        else:
            return -100  # Loss
    else:
        # Model picks away
        if not row['actual_home_wins']:
            # Win: pay based on away ML odds
            if row['moneyline_away'] > 0:
                return row['moneyline_away']  # Underdog win
            else:
                return 100 / abs(row['moneyline_away']) * 100  # Favorite win
        else:
            return -100  # Loss

df['ml_profit'] = df.apply(calc_ml_profit, axis=1)

# Create filter variables
df['home_favored'] = df['market_spread_home'] > 0
df['away_favored'] = df['market_spread_home'] < 0
df['pick_em'] = df['market_spread_home'] == 0

df['model_picks_home'] = df['pred_home_wins'] == 1
df['model_picks_away'] = df['pred_home_wins'] == 0

# ML odds buckets (based on model's pick)
def get_ml_bucket(row):
    ml = row['moneyline_home'] if row['pred_home_wins'] else row['moneyline_away']
    if ml <= -200:
        return 'heavy_fav'
    elif ml < -150:
        return 'solid_fav'
    elif ml < -110:
        return 'slight_fav'
    elif ml < 110:
        return 'pick_em'
    elif ml < 150:
        return 'slight_dog'
    elif ml < 200:
        return 'solid_dog'
    else:
        return 'heavy_dog'

df['ml_bucket'] = df.apply(get_ml_bucket, axis=1)

# Implied probability buckets
def get_implied_prob_bucket(row):
    prob = row['win_prob_home'] if row['pred_home_wins'] else (1 - row['win_prob_home'])
    if prob >= 0.70:
        return '70%+'
    elif prob >= 0.60:
        return '60-70%'
    elif prob >= 0.55:
        return '55-60%'
    elif prob >= 0.50:
        return '50-55%'
    else:
        return '<50%'

df['implied_prob_bucket'] = df.apply(get_implied_prob_bucket, axis=1)

def calc_ml_metrics(filtered_df):
    """Calculate ML betting metrics"""
    if len(filtered_df) == 0:
        return None
    
    n = len(filtered_df)
    wins = filtered_df['ml_correct'].sum()
    win_pct = wins / n * 100
    total_profit = filtered_df['ml_profit'].sum()
    roi = (total_profit / (n * 100)) * 100
    avg_odds = filtered_df.apply(
        lambda r: r['moneyline_home'] if r['pred_home_wins'] else r['moneyline_away'], 
        axis=1
    ).mean()
    
    return {
        'n': n,
        'wins': wins,
        'win_pct': win_pct,
        'total_profit': total_profit,
        'roi': roi,
        'avg_odds': avg_odds
    }

print(f"\n{'='*100}")
print("BASELINE: OVERALL ML PERFORMANCE")
print("="*100)
baseline = calc_ml_metrics(df)
print(f"Total Games: {baseline['n']}")
print(f"Wins: {baseline['wins']} ({baseline['win_pct']:.2f}%)")
print(f"Total Profit ($100/game): ${baseline['total_profit']:,.0f}")
print(f"ROI: {baseline['roi']:+.2f}%")
print(f"Average ML Odds: {baseline['avg_odds']:+.0f}")

# TABLE 1: BY FAVORITE TYPE
print(f"\n{'='*100}")
print("TABLE 1: BY FAVORITE TYPE (Market)")
print("="*100)
print(f"{'Favorite Type':<20} {'Games':<10} {'Wins':<10} {'Win %':<10} {'Total Profit':<15} {'ROI%':<10}")
print("-"*100)

for fav_type, label in [('home_favored', 'Home Favored'), 
                         ('away_favored', 'Away Favored')]:
    filtered = df[df[fav_type]]
    metrics = calc_ml_metrics(filtered)
    if metrics:
        print(f"{label:<20} {metrics['n']:<10} {metrics['wins']:<10} {metrics['win_pct']:<10.2f} ${metrics['total_profit']:<14,.0f} {metrics['roi']:<+10.2f}")

# TABLE 2: BY MODEL PREDICTION
print(f"\n{'='*100}")
print("TABLE 2: BY MODEL PREDICTION")
print("="*100)
print(f"{'Model Picks':<20} {'Games':<10} {'Wins':<10} {'Win %':<10} {'Total Profit':<15} {'ROI%':<10}")
print("-"*100)

for pred_type, label in [('model_picks_home', 'Home to Win'),
                          ('model_picks_away', 'Away to Win')]:
    filtered = df[df[pred_type]]
    metrics = calc_ml_metrics(filtered)
    if metrics:
        print(f"{label:<20} {metrics['n']:<10} {metrics['wins']:<10} {metrics['win_pct']:<10.2f} ${metrics['total_profit']:<14,.0f} {metrics['roi']:<+10.2f}")

# TABLE 3: BY ML ODDS BUCKET
print(f"\n{'='*100}")
print("TABLE 3: BY MONEYLINE ODDS BUCKET (Model's Pick)")
print("="*100)
print(f"{'ML Odds Bucket':<20} {'Games':<10} {'Wins':<10} {'Win %':<10} {'Avg Odds':<12} {'Total Profit':<15} {'ROI%':<10}")
print("-"*100)

ml_buckets = [
    ('heavy_fav', 'Heavy Fav (<-200)'),
    ('solid_fav', 'Solid Fav (-150 to -200)'),
    ('slight_fav', 'Slight Fav (-110 to -150)'),
    ('pick_em', 'Pick\'em (-110 to +110)'),
    ('slight_dog', 'Slight Dog (+110 to +150)'),
    ('solid_dog', 'Solid Dog (+150 to +200)'),
    ('heavy_dog', 'Heavy Dog (>+200)')
]

for bucket, label in ml_buckets:
    filtered = df[df['ml_bucket'] == bucket]
    metrics = calc_ml_metrics(filtered)
    if metrics and metrics['n'] >= 5:
        print(f"{label:<20} {metrics['n']:<10} {metrics['wins']:<10} {metrics['win_pct']:<10.2f} {metrics['avg_odds']:<+12.0f} ${metrics['total_profit']:<14,.0f} {metrics['roi']:<+10.2f}")

# TABLE 4: BY IMPLIED PROBABILITY
print(f"\n{'='*100}")
print("TABLE 4: BY IMPLIED WIN PROBABILITY (Model's Confidence)")
print("="*100)
print(f"{'Implied Prob':<20} {'Games':<10} {'Wins':<10} {'Win %':<10} {'Total Profit':<15} {'ROI%':<10}")
print("-"*100)

for bucket in ['70%+', '60-70%', '55-60%', '50-55%', '<50%']:
    filtered = df[df['implied_prob_bucket'] == bucket]
    metrics = calc_ml_metrics(filtered)
    if metrics and metrics['n'] >= 5:
        print(f"{bucket:<20} {metrics['n']:<10} {metrics['wins']:<10} {metrics['win_pct']:<10.2f} ${metrics['total_profit']:<14,.0f} {metrics['roi']:<+10.2f}")

# TABLE 5: FAVORITE TYPE × MODEL PREDICTION
print(f"\n{'='*100}")
print("TABLE 5: FAVORITE TYPE × MODEL PREDICTION")
print("="*100)
print(f"{'Strategy':<35} {'Games':<10} {'Wins':<10} {'Win %':<10} {'Total Profit':<15} {'ROI%':<10}")
print("-"*100)

combos = [
    ('home_favored', 'model_picks_home', 'Home Fav + Pick Home'),
    ('home_favored', 'model_picks_away', 'Home Fav + Pick Away'),
    ('away_favored', 'model_picks_home', 'Away Fav + Pick Home'),
    ('away_favored', 'model_picks_away', 'Away Fav + Pick Away'),
]

for fav, pred, label in combos:
    filtered = df[df[fav] & df[pred]]
    metrics = calc_ml_metrics(filtered)
    if metrics:
        print(f"{label:<35} {metrics['n']:<10} {metrics['wins']:<10} {metrics['win_pct']:<10.2f} ${metrics['total_profit']:<14,.0f} {metrics['roi']:<+10.2f}")

# TABLE 6: FAVORITE TYPE × ML ODDS
print(f"\n{'='*100}")
print("TABLE 6: FAVORITE TYPE × ML ODDS BUCKET")
print("="*100)
print(f"{'Strategy':<40} {'Games':<10} {'Wins':<10} {'Win %':<10} {'Total Profit':<15} {'ROI%':<10}")
print("-"*100)

for fav, fav_label in [('home_favored', 'Home Fav'), ('away_favored', 'Away Fav')]:
    for bucket, bucket_label in [('heavy_fav', 'Heavy Fav'), ('solid_fav', 'Solid Fav'), 
                                  ('slight_fav', 'Slight Fav'), ('slight_dog', 'Slight Dog'),
                                  ('solid_dog', 'Solid Dog'), ('heavy_dog', 'Heavy Dog')]:
        filtered = df[df[fav] & (df['ml_bucket'] == bucket)]
        metrics = calc_ml_metrics(filtered)
        if metrics and metrics['n'] >= 10:
            label = f"{fav_label} + Pick {bucket_label}"
            print(f"{label:<40} {metrics['n']:<10} {metrics['wins']:<10} {metrics['win_pct']:<10.2f} ${metrics['total_profit']:<14,.0f} {metrics['roi']:<+10.2f}")

# TABLE 7: MODEL PREDICTION × ML ODDS BUCKET
print(f"\n{'='*100}")
print("TABLE 7: MODEL PREDICTION × ML ODDS BUCKET")
print("="*100)
print(f"{'Strategy':<50} {'Games':<10} {'Wins':<10} {'Win %':<10} {'ROI%':<10} {'Profit':<12}")
print("-"*100)

for pred, pred_label in [('model_picks_home', 'Pick Home'), ('model_picks_away', 'Pick Away')]:
    for bucket, bucket_label in ml_buckets:
        filtered = df[df[pred] & (df['ml_bucket'] == bucket)]
        metrics = calc_ml_metrics(filtered)
        if metrics and metrics['n'] >= 10:
            label = f"{pred_label} + {bucket_label}"
            print(f"{label:<50} {metrics['n']:<10} {metrics['wins']:<10} {metrics['win_pct']:<10.2f} {metrics['roi']:<+10.2f} ${metrics['total_profit']:<11,.0f}")

# TABLE 8: THREE-WAY COMBINATIONS (Fav Type × Model Pick × ML Odds)
print(f"\n{'='*100}")
print("TABLE 8: THREE-WAY COMBINATIONS (Favorite Type × Model Pick × ML Odds)")
print("="*100)
print(f"{'Strategy':<60} {'Games':<10} {'Wins':<10} {'Win %':<10} {'ROI%':<10} {'Profit':<12}")
print("-"*100)

for fav, fav_label in [('home_favored', 'Home Fav'), ('away_favored', 'Away Fav')]:
    for pred, pred_label in [('model_picks_home', 'Pick Home'), ('model_picks_away', 'Pick Away')]:
        for bucket, bucket_label in [('heavy_fav', 'Heavy Fav'), ('solid_fav', 'Solid Fav'), 
                                      ('slight_fav', 'Slight Fav'), ('pick_em', "Pick'em"),
                                      ('slight_dog', 'Slight Dog'), ('solid_dog', 'Solid Dog')]:
            filtered = df[df[fav] & df[pred] & (df['ml_bucket'] == bucket)]
            metrics = calc_ml_metrics(filtered)
            if metrics and metrics['n'] >= 5:
                label = f"{fav_label} + {pred_label} + {bucket_label}"
                print(f"{label:<60} {metrics['n']:<10} {metrics['wins']:<10} {metrics['win_pct']:<10.2f} {metrics['roi']:<+10.2f} ${metrics['total_profit']:<11,.0f}")

# TABLE 9: CONFIDENCE × FAVORITE TYPE
print(f"\n{'='*100}")
print("TABLE 9: CONFIDENCE LEVEL × FAVORITE TYPE")
print("="*100)
print(f"{'Strategy':<45} {'Games':<10} {'Wins':<10} {'Win %':<10} {'ROI%':<10} {'Profit':<12}")
print("-"*100)

for prob_bucket in ['70%+', '60-70%', '55-60%', '50-55%']:
    for fav, fav_label in [('home_favored', 'Home Fav'), ('away_favored', 'Away Fav')]:
        filtered = df[(df['implied_prob_bucket'] == prob_bucket) & df[fav]]
        metrics = calc_ml_metrics(filtered)
        if metrics and metrics['n'] >= 10:
            label = f"{prob_bucket} Conf + {fav_label}"
            print(f"{label:<45} {metrics['n']:<10} {metrics['wins']:<10} {metrics['win_pct']:<10.2f} {metrics['roi']:<+10.2f} ${metrics['total_profit']:<11,.0f}")

# TABLE 10: CONFIDENCE × MODEL PREDICTION
print(f"\n{'='*100}")
print("TABLE 10: CONFIDENCE LEVEL × MODEL PREDICTION")
print("="*100)
print(f"{'Strategy':<45} {'Games':<10} {'Wins':<10} {'Win %':<10} {'ROI%':<10} {'Profit':<12}")
print("-"*100)

for prob_bucket in ['70%+', '60-70%', '55-60%', '50-55%']:
    for pred, pred_label in [('model_picks_home', 'Pick Home'), ('model_picks_away', 'Pick Away')]:
        filtered = df[(df['implied_prob_bucket'] == prob_bucket) & df[pred]]
        metrics = calc_ml_metrics(filtered)
        if metrics and metrics['n'] >= 10:
            label = f"{prob_bucket} Conf + {pred_label}"
            print(f"{label:<45} {metrics['n']:<10} {metrics['wins']:<10} {metrics['win_pct']:<10.2f} {metrics['roi']:<+10.2f} ${metrics['total_profit']:<11,.0f}")

# TABLE 11: CONFIDENCE × ML ODDS BUCKET
print(f"\n{'='*100}")
print("TABLE 11: CONFIDENCE LEVEL × ML ODDS BUCKET")
print("="*100)
print(f"{'Strategy':<50} {'Games':<10} {'Wins':<10} {'Win %':<10} {'ROI%':<10} {'Profit':<12}")
print("-"*100)

for prob_bucket in ['70%+', '60-70%', '55-60%', '50-55%']:
    for bucket, bucket_label in ml_buckets:
        filtered = df[(df['implied_prob_bucket'] == prob_bucket) & (df['ml_bucket'] == bucket)]
        metrics = calc_ml_metrics(filtered)
        if metrics and metrics['n'] >= 10:
            label = f"{prob_bucket} Conf + {bucket_label}"
            print(f"{label:<50} {metrics['n']:<10} {metrics['wins']:<10} {metrics['win_pct']:<10.2f} {metrics['roi']:<+10.2f} ${metrics['total_profit']:<11,.0f}")

# TABLE 12: TOP 50 BY ROI (Comprehensive)
print(f"\n{'='*100}")
print("TABLE 12: TOP 50 ML STRATEGIES BY ROI (Min 10 Games)")
print("="*100)
print(f"{'Rank':<6} {'Strategy':<65} {'Games':<10} {'Wins':<10} {'Win %':<10} {'ROI%':<10} {'Profit':<12}")
print("-"*100)

# Collect all strategies
all_strategies = []

# Single variable
for fav_type, label in [('home_favored', 'Home Favored'), ('away_favored', 'Away Favored')]:
    filtered = df[df[fav_type]]
    metrics = calc_ml_metrics(filtered)
    if metrics and metrics['n'] >= 10:
        all_strategies.append((label, metrics))

for pred_type, label in [('model_picks_home', 'Pick Home'), ('model_picks_away', 'Pick Away')]:
    filtered = df[df[pred_type]]
    metrics = calc_ml_metrics(filtered)
    if metrics and metrics['n'] >= 10:
        all_strategies.append((label, metrics))

for bucket, bucket_label in ml_buckets:
    filtered = df[df['ml_bucket'] == bucket]
    metrics = calc_ml_metrics(filtered)
    if metrics and metrics['n'] >= 10:
        all_strategies.append((bucket_label, metrics))

for prob_bucket in ['70%+', '60-70%', '55-60%', '50-55%', '<50%']:
    filtered = df[df['implied_prob_bucket'] == prob_bucket]
    metrics = calc_ml_metrics(filtered)
    if metrics and metrics['n'] >= 10:
        all_strategies.append((f"{prob_bucket} Confidence", metrics))

# Two-way combinations
for fav, fav_label in [('home_favored', 'Home Fav'), ('away_favored', 'Away Fav')]:
    for pred, pred_label in [('model_picks_home', 'Pick Home'), ('model_picks_away', 'Pick Away')]:
        filtered = df[df[fav] & df[pred]]
        metrics = calc_ml_metrics(filtered)
        if metrics and metrics['n'] >= 10:
            all_strategies.append((f"{fav_label} + {pred_label}", metrics))

for fav, fav_label in [('home_favored', 'Home Fav'), ('away_favored', 'Away Fav')]:
    for bucket, bucket_label in ml_buckets:
        filtered = df[df[fav] & (df['ml_bucket'] == bucket)]
        metrics = calc_ml_metrics(filtered)
        if metrics and metrics['n'] >= 10:
            all_strategies.append((f"{fav_label} + {bucket_label}", metrics))

for pred, pred_label in [('model_picks_home', 'Pick Home'), ('model_picks_away', 'Pick Away')]:
    for bucket, bucket_label in ml_buckets:
        filtered = df[df[pred] & (df['ml_bucket'] == bucket)]
        metrics = calc_ml_metrics(filtered)
        if metrics and metrics['n'] >= 10:
            all_strategies.append((f"{pred_label} + {bucket_label}", metrics))

for prob_bucket in ['70%+', '60-70%', '55-60%', '50-55%']:
    for fav, fav_label in [('home_favored', 'Home Fav'), ('away_favored', 'Away Fav')]:
        filtered = df[(df['implied_prob_bucket'] == prob_bucket) & df[fav]]
        metrics = calc_ml_metrics(filtered)
        if metrics and metrics['n'] >= 10:
            all_strategies.append((f"{prob_bucket} Conf + {fav_label}", metrics))

for prob_bucket in ['70%+', '60-70%', '55-60%', '50-55%']:
    for pred, pred_label in [('model_picks_home', 'Pick Home'), ('model_picks_away', 'Pick Away')]:
        filtered = df[(df['implied_prob_bucket'] == prob_bucket) & df[pred]]
        metrics = calc_ml_metrics(filtered)
        if metrics and metrics['n'] >= 10:
            all_strategies.append((f"{prob_bucket} Conf + {pred_label}", metrics))

for prob_bucket in ['70%+', '60-70%', '55-60%', '50-55%']:
    for bucket, bucket_label in ml_buckets:
        filtered = df[(df['implied_prob_bucket'] == prob_bucket) & (df['ml_bucket'] == bucket)]
        metrics = calc_ml_metrics(filtered)
        if metrics and metrics['n'] >= 10:
            all_strategies.append((f"{prob_bucket} Conf + {bucket_label}", metrics))

# Three-way combinations
for fav, fav_label in [('home_favored', 'Home Fav'), ('away_favored', 'Away Fav')]:
    for pred, pred_label in [('model_picks_home', 'Pick Home'), ('model_picks_away', 'Pick Away')]:
        for bucket, bucket_label in [('heavy_fav', 'Heavy Fav'), ('solid_fav', 'Solid Fav'), 
                                      ('slight_fav', 'Slight Fav'), ('pick_em', "Pick'em"),
                                      ('slight_dog', 'Slight Dog'), ('solid_dog', 'Solid Dog')]:
            filtered = df[df[fav] & df[pred] & (df['ml_bucket'] == bucket)]
            metrics = calc_ml_metrics(filtered)
            if metrics and metrics['n'] >= 10:
                all_strategies.append((f"{fav_label} + {pred_label} + {bucket_label}", metrics))

# Sort by ROI
all_strategies.sort(key=lambda x: x[1]['roi'], reverse=True)
top_50 = all_strategies[:50]

for i, (label, metrics) in enumerate(top_50, 1):
    print(f"{i:<6} {label:<65} {metrics['n']:<10} {metrics['wins']:<10} {metrics['win_pct']:<10.2f} {metrics['roi']:<+10.2f} ${metrics['total_profit']:<11,.0f}")

# TABLE 13: TOP 30 BY TOTAL PROFIT
print(f"\n{'='*100}")
print("TABLE 13: TOP 30 ML STRATEGIES BY TOTAL PROFIT (Min 10 Games)")
print("="*100)
print(f"{'Rank':<6} {'Strategy':<65} {'Games':<10} {'Wins':<10} {'Win %':<10} {'ROI%':<10} {'Profit':<12}")
print("-"*100)

all_strategies.sort(key=lambda x: x[1]['total_profit'], reverse=True)
top_30_profit = all_strategies[:30]

for i, (label, metrics) in enumerate(top_30_profit, 1):
    print(f"{i:<6} {label:<65} {metrics['n']:<10} {metrics['wins']:<10} {metrics['win_pct']:<10.2f} {metrics['roi']:<+10.2f} ${metrics['total_profit']:<11,.0f}")

print(f"\n{'='*100}")
print("KEY INSIGHTS")
print("="*100)
print("\n1. ML betting is very different from ATS betting")
print("   - Favorites need high win % to be profitable (due to low payouts)")
print("   - Underdogs can be profitable with <50% win rate (due to high payouts)")
print("\n2. Look for positive ROI, not just high win %")
print("   - 70% wins on heavy favorites might = negative ROI")
print("   - 45% wins on big underdogs might = positive ROI")
print("\n3. ML odds already incorporate market efficiency")
print("   - Harder to find edges than ATS betting")
print("   - Model needs to find mispriced games")

print("\n" + "="*100)

