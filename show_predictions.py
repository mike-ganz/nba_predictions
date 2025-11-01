"""Display predictions in a nice format."""
import pandas as pd
import json
import sys

# Load predictions
predictions_file = sys.argv[1] if len(sys.argv) > 1 else 'predictions/predictions_2025-10-31.csv'
df = pd.read_csv(predictions_file)

# Try to load injury warnings from game data
game_data_file = predictions_file.replace('predictions/predictions_', 'data/games_future_').replace('.csv', '_norm.jsonl')
injury_warnings = {}
try:
    with open(game_data_file) as f:
        for line in f:
            game = json.loads(line)
            warnings = game.get('metadata', {}).get('injury_warnings')
            if warnings:
                injury_warnings[game['game_id']] = warnings
except:
    pass

print()
print("=" * 100)
print("NBA PREDICTIONS")
print("=" * 100)
print()

for _, row in df.iterrows():
    away = row['away_team']
    home = row['home_team']
    spread = row['market_spread_home']
    pred = row['pred_margin_mu']
    sigma = row['pred_margin_sigma']
    h_cover = row['cover_prob_home'] * 100
    a_cover = row['cover_prob_away'] * 100
    h_win = row['win_prob_home'] * 100
    a_win = row['win_prob_away'] * 100
    
    # Determine favorite and format spread
    if spread < 0:
        market_line = f"{home} {spread:.1f}"
    else:
        market_line = f"{away} -{spread:.1f}"
    
    # Prediction
    pred_winner = home if pred > 0 else away
    pred_margin = abs(pred)
    
    # Best bet
    if h_cover > 55:
        best_bet = f"[*] {home} {spread:+.1f} ({h_cover:.1f}%)"
    elif a_cover > 55:
        best_bet = f"[*] {away} {-spread:+.1f} ({a_cover:.1f}%)"
    else:
        best_bet = "No strong edge"
    
    print(f"{away} @ {home}")
    print(f"  Market Spread: {market_line}")
    print(f"  Predicted: {pred_winner} by {pred_margin:.1f} pts (uncertainty: ±{sigma:.1f})")
    print(f"  Win Prob: {home} {h_win:.1f}% | {away} {a_win:.1f}%")
    print(f"  Cover Prob: {home} {h_cover:.1f}% | {away} {a_cover:.1f}%")
    print(f"  Best Bet: {best_bet}")
    
    # Show injury warnings if available
    game_id = row.get('game_id', '')
    if game_id in injury_warnings:
        for warning in injury_warnings[game_id]:
            print(f"  [!] {warning}")
    
    print()

print("=" * 100)
print("Note: Cover probabilities > 55% suggest potential value bets")
print("=" * 100)

