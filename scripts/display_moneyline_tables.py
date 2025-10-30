"""
Display all moneyline analysis tables in readable format
"""

import pandas as pd
import json

# Load predictions
df = pd.read_csv('predictions/test_2425_normalized_predictions.csv')

# Load moneyline odds from raw JSONL
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

# Calculate ML profit
def calc_ml_profit(row):
    """Calculate profit/loss for a $100 ML bet based on model's pick"""
    if row['pred_home_wins']:
        if row['actual_home_wins']:
            if row['moneyline_home'] > 0:
                return row['moneyline_home']
            else:
                return 100 / abs(row['moneyline_home']) * 100
        else:
            return -100
    else:
        if not row['actual_home_wins']:
            if row['moneyline_away'] > 0:
                return row['moneyline_away']
            else:
                return 100 / abs(row['moneyline_away']) * 100
        else:
            return -100

df['ml_profit'] = df.apply(calc_ml_profit, axis=1)

# Create filter variables
df['home_favored'] = df['market_spread_home'] > 0
df['away_favored'] = df['market_spread_home'] < 0
df['pick_em'] = df['market_spread_home'] == 0
df['model_picks_home'] = df['pred_home_wins'] == 1
df['model_picks_away'] = df['pred_home_wins'] == 0

# ML odds buckets
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
    
    return {
        'n': n,
        'wins': wins,
        'win_pct': win_pct,
        'total_profit': total_profit,
        'roi': roi
    }

def print_table(title, data, headers):
    """Print a nicely formatted table"""
    print("\n" + "="*120)
    print(title)
    print("="*120)
    
    # Print headers
    header_line = ""
    for header, width in headers:
        header_line += f"{header:<{width}}"
    print(header_line)
    print("-"*120)
    
    # Print data
    for row in data:
        row_line = ""
        for value, (_, width) in zip(row, headers):
            if isinstance(value, float):
                if 'ROI' in str(value) or '%' in str(headers[row.index(value)][0]):
                    row_line += f"{value:+{width-1}.2f}%"
                else:
                    row_line += f"{value:<{width}.2f}"
            elif isinstance(value, int) and value > 1000:
                row_line += f"${value:<{width-1},}"
            else:
                row_line += f"{str(value):<{width}}"
        print(row_line)

print("\n" + "#"*120)
print("MONEYLINE BETTING ANALYSIS - COMPLETE TABLES")
print("#"*120)

print("\n" + "="*120)
print("DEFINITIONS & TERMINOLOGY")
print("="*120)
print("""
FAVORITE TYPE (Based on Market Spread):
  - Home Favored: Market spread > 0 (e.g., home is -5.5 point favorite)
  - Away Favored: Market spread < 0 (e.g., home is +5.5 point underdog, away favored)
  - Pick'em: Market spread = 0 (even matchup)

MODEL PREDICTION:
  - Pick Home: Model predicts home team will win (win_prob_home > 50%)
  - Pick Away: Model predicts away team will win (win_prob_away > 50%)

MONEYLINE ODDS BUCKETS (for the team the model picks):
  - Heavy Fav: Odds <= -200 (need to bet $200 to win $100; ~67% implied probability)
  - Solid Fav: Odds -150 to -200 (need to bet $150-200 to win $100; ~60-67% implied)
  - Slight Fav: Odds -110 to -150 (need to bet $110-150 to win $100; ~52-60% implied)
  - Pick'em: Odds -110 to +110 (roughly even money; ~48-52% implied)
  - Slight Dog: Odds +110 to +150 (bet $100 to win $110-150; ~40-48% implied)
  - Solid Dog: Odds +150 to +200 (bet $100 to win $150-200; ~33-40% implied)
  - Heavy Dog: Odds > +200 (bet $100 to win $200+; <33% implied)

MODEL CONFIDENCE (implied probability for the team model picks):
  - 70%+: Model is very confident in its pick
  - 60-70%: Model is moderately confident
  - 55-60%: Model is somewhat confident
  - 50-55%: Model is barely confident (close to 50/50)
  - <50%: Should not happen (model wouldn't pick this team)

METRICS:
  - Games: Number of games in this category
  - Wins: Number of games where model's ML pick was correct
  - Win %: Percentage of correct picks (Wins / Games * 100)
  - Total Profit: Total profit/loss betting $100 per game
  - ROI%: Return on Investment percentage = (Total Profit / Total Wagered) * 100
          Positive ROI = profitable, Negative ROI = losing money

IMPORTANT: 
  - High win % does NOT guarantee profit in ML betting!
  - Favorites require very high accuracy due to low payouts (e.g., -400 needs 80% to break even)
  - Underdogs can be profitable with <50% win rate due to high payouts (e.g., +200 needs 34% to break even)
""")

# BASELINE
print("\n" + "="*120)
print("BASELINE: OVERALL ML PERFORMANCE")
print("="*120)
baseline = calc_ml_metrics(df)
print(f"Total Games: {baseline['n']}")
print(f"Wins: {baseline['wins']} ({baseline['win_pct']:.2f}%)")
print(f"Total Profit ($100/game): ${baseline['total_profit']:,.0f}")
print(f"ROI: {baseline['roi']:+.2f}%")

# TABLE 1: By Favorite Type
data = []
for fav_type, label in [('home_favored', 'Home Favored'), ('away_favored', 'Away Favored')]:
    filtered = df[df[fav_type]]
    metrics = calc_ml_metrics(filtered)
    if metrics:
        data.append([label, metrics['n'], metrics['wins'], metrics['win_pct'], 
                    f"${metrics['total_profit']:,.0f}", metrics['roi']])

print_table(
    "TABLE 1: BY FAVORITE TYPE (Market)",
    data,
    [("Favorite Type", 25), ("Games", 12), ("Wins", 12), ("Win %", 12), 
     ("Total Profit", 18), ("ROI%", 12)]
)

# TABLE 2: By Model Prediction
data = []
for pred_type, label in [('model_picks_home', 'Home to Win'), ('model_picks_away', 'Away to Win')]:
    filtered = df[df[pred_type]]
    metrics = calc_ml_metrics(filtered)
    if metrics:
        data.append([label, metrics['n'], metrics['wins'], metrics['win_pct'], 
                    f"${metrics['total_profit']:,.0f}", metrics['roi']])

print_table(
    "TABLE 2: BY MODEL PREDICTION",
    data,
    [("Model Picks", 25), ("Games", 12), ("Wins", 12), ("Win %", 12), 
     ("Total Profit", 18), ("ROI%", 12)]
)

# TABLE 3: By ML Odds Bucket
ml_buckets = [
    ('heavy_fav', 'Heavy Fav (< -200)'),
    ('solid_fav', 'Solid Fav (-150 to -200)'),
    ('slight_fav', 'Slight Fav (-110 to -150)'),
    ('pick_em', "Pick'em (-110 to +110)"),
    ('slight_dog', 'Slight Dog (+110 to +150)'),
    ('solid_dog', 'Solid Dog (+150 to +200)'),
    ('heavy_dog', 'Heavy Dog (> +200)')
]

data = []
for bucket, label in ml_buckets:
    filtered = df[df['ml_bucket'] == bucket]
    metrics = calc_ml_metrics(filtered)
    if metrics and metrics['n'] >= 5:
        data.append([label, metrics['n'], metrics['wins'], metrics['win_pct'], 
                    f"${metrics['total_profit']:,.0f}", metrics['roi']])

print_table(
    "TABLE 3: BY MONEYLINE ODDS BUCKET (Model's Pick)",
    data,
    [("ML Odds Bucket", 35), ("Games", 12), ("Wins", 12), ("Win %", 12), 
     ("Total Profit", 18), ("ROI%", 12)]
)

# TABLE 4: By Confidence Level
data = []
for bucket in ['70%+', '60-70%', '55-60%', '50-55%']:
    filtered = df[df['implied_prob_bucket'] == bucket]
    metrics = calc_ml_metrics(filtered)
    if metrics and metrics['n'] >= 5:
        data.append([f"{bucket} Confidence", metrics['n'], metrics['wins'], metrics['win_pct'], 
                    f"${metrics['total_profit']:,.0f}", metrics['roi']])

print_table(
    "TABLE 4: BY MODEL CONFIDENCE LEVEL",
    data,
    [("Confidence Level", 30), ("Games", 12), ("Wins", 12), ("Win %", 12), 
     ("Total Profit", 18), ("ROI%", 12)]
)

# TABLE 5: Favorite Type x Model Prediction
data = []
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
        data.append([label, metrics['n'], metrics['wins'], metrics['win_pct'], 
                    f"${metrics['total_profit']:,.0f}", metrics['roi']])

print_table(
    "TABLE 5: FAVORITE TYPE x MODEL PREDICTION",
    data,
    [("Strategy", 35), ("Games", 12), ("Wins", 12), ("Win %", 12), 
     ("Total Profit", 18), ("ROI%", 12)]
)

# TABLE 6: Favorite Type x ML Odds
data = []
for fav, fav_label in [('home_favored', 'Home Fav'), ('away_favored', 'Away Fav')]:
    for bucket, bucket_label in [('heavy_fav', 'Heavy Fav'), ('solid_fav', 'Solid Fav'), 
                                  ('slight_fav', 'Slight Fav'), ('slight_dog', 'Slight Dog'),
                                  ('solid_dog', 'Solid Dog')]:
        filtered = df[df[fav] & (df['ml_bucket'] == bucket)]
        metrics = calc_ml_metrics(filtered)
        if metrics and metrics['n'] >= 10:
            label = f"{fav_label} + {bucket_label}"
            data.append([label, metrics['n'], metrics['wins'], metrics['win_pct'], 
                        f"${metrics['total_profit']:,.0f}", metrics['roi']])

print_table(
    "TABLE 6: FAVORITE TYPE x ML ODDS BUCKET",
    data,
    [("Strategy", 40), ("Games", 12), ("Wins", 12), ("Win %", 12), 
     ("Total Profit", 18), ("ROI%", 12)]
)

# TABLE 7: Confidence x Favorite Type
data = []
for prob_bucket in ['70%+', '60-70%', '55-60%', '50-55%']:
    for fav, fav_label in [('home_favored', 'Home Fav'), ('away_favored', 'Away Fav')]:
        filtered = df[(df['implied_prob_bucket'] == prob_bucket) & df[fav]]
        metrics = calc_ml_metrics(filtered)
        if metrics and metrics['n'] >= 10:
            label = f"{prob_bucket} Conf + {fav_label}"
            data.append([label, metrics['n'], metrics['wins'], metrics['win_pct'], 
                        f"${metrics['total_profit']:,.0f}", metrics['roi']])

print_table(
    "TABLE 7: CONFIDENCE LEVEL x FAVORITE TYPE",
    data,
    [("Strategy", 40), ("Games", 12), ("Wins", 12), ("Win %", 12), 
     ("Total Profit", 18), ("ROI%", 12)]
)

# TABLE 8: Confidence x Model Prediction
data = []
for prob_bucket in ['70%+', '60-70%', '55-60%', '50-55%']:
    for pred, pred_label in [('model_picks_home', 'Pick Home'), ('model_picks_away', 'Pick Away')]:
        filtered = df[(df['implied_prob_bucket'] == prob_bucket) & df[pred]]
        metrics = calc_ml_metrics(filtered)
        if metrics and metrics['n'] >= 10:
            label = f"{prob_bucket} Conf + {pred_label}"
            data.append([label, metrics['n'], metrics['wins'], metrics['win_pct'], 
                        f"${metrics['total_profit']:,.0f}", metrics['roi']])

print_table(
    "TABLE 8: CONFIDENCE LEVEL x MODEL PREDICTION",
    data,
    [("Strategy", 40), ("Games", 12), ("Wins", 12), ("Win %", 12), 
     ("Total Profit", 18), ("ROI%", 12)]
)

# TABLE 9: TOP 30 BY ROI
all_strategies = []

# Collect all two-way combinations
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

for prob_bucket in ['70%+', '60-70%', '55-60%', '50-55%']:
    for fav, fav_label in [('home_favored', 'Home Fav'), ('away_favored', 'Away Fav')]:
        filtered = df[(df['implied_prob_bucket'] == prob_bucket) & df[fav]]
        metrics = calc_ml_metrics(filtered)
        if metrics and metrics['n'] >= 10:
            all_strategies.append((f"{prob_bucket} Conf + {fav_label}", metrics))

for prob_bucket in ['70%+', '60-70%', '55-60%']:
    for pred, pred_label in [('model_picks_home', 'Pick Home'), ('model_picks_away', 'Pick Away')]:
        filtered = df[(df['implied_prob_bucket'] == prob_bucket) & df[pred]]
        metrics = calc_ml_metrics(filtered)
        if metrics and metrics['n'] >= 10:
            all_strategies.append((f"{prob_bucket} Conf + {pred_label}", metrics))

# Add single variable strategies
for bucket in ['70%+', '60-70%', '55-60%', '50-55%']:
    filtered = df[df['implied_prob_bucket'] == bucket]
    metrics = calc_ml_metrics(filtered)
    if metrics and metrics['n'] >= 10:
        all_strategies.append((f"{bucket} Confidence", metrics))

# Sort by ROI and get top 30
all_strategies.sort(key=lambda x: x[1]['roi'], reverse=True)
data = []
for i, (label, metrics) in enumerate(all_strategies[:30], 1):
    data.append([i, label, metrics['n'], metrics['wins'], metrics['win_pct'], 
                f"${metrics['total_profit']:,.0f}", metrics['roi']])

print_table(
    "TABLE 9: TOP 30 STRATEGIES BY ROI (Min 10 Games)",
    data,
    [("Rank", 8), ("Strategy", 50), ("Games", 10), ("Wins", 10), ("Win %", 10), 
     ("Total Profit", 16), ("ROI%", 10)]
)

# TABLE 10: TOP 30 BY TOTAL PROFIT
all_strategies.sort(key=lambda x: x[1]['total_profit'], reverse=True)
data = []
for i, (label, metrics) in enumerate(all_strategies[:30], 1):
    data.append([i, label, metrics['n'], metrics['wins'], metrics['win_pct'], 
                f"${metrics['total_profit']:,.0f}", metrics['roi']])

print_table(
    "TABLE 10: TOP 30 STRATEGIES BY TOTAL PROFIT (Min 10 Games)",
    data,
    [("Rank", 8), ("Strategy", 50), ("Games", 10), ("Wins", 10), ("Win %", 10), 
     ("Total Profit", 16), ("ROI%", 10)]
)

print("\n" + "#"*120)
print("END OF ANALYSIS")
print("#"*120 + "\n")

