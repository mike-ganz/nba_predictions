# OLD Model Prediction Files Guide

## Files Generated

### 1. **24-25 Season Predictions**
- **File:** `predictions/OLD_model_2425_predictions.csv`
- **Games:** 1,315
- **Performance:** 51.33% ATS, -2.01% ROI
- **MAE:** 10.64 points

### 2. **25-26 Season Predictions (Current)**
- **File:** `predictions/OLD_model_2526_predictions.csv`
- **Games:** 72
- **Performance:** 61.11% ATS, +16.66% ROI
- **MAE:** 10.68 points

---

## CSV Column Descriptions

### Game Identifiers
- `game_id`: Unique game identifier (format: YYYY-MM-DD-AWAY-HOME)
- `date`: Game date
- `away_team`: Away team abbreviation
- `home_team`: Home team abbreviation

### Market Data
- `market_spread_home`: Point spread from home team perspective
  - Negative = home favored (e.g., -6.5 means home favored by 6.5)
  - Positive = home underdog (e.g., +3.5 means home is 3.5 point dog)
- `baseline_margin`: Market expectation = -market_spread_home
  - This is what the market expects the margin to be

### Model Predictions
- `pred_margin_mu`: Model's predicted margin (home score - away score)
  - Positive = home team expected to win
  - Negative = away team expected to win
- `pred_margin_sigma`: Model's uncertainty (standard deviation)
  - Higher = less confident prediction
  - Typical range: 5.9 to 8.5 points

### Probabilities
- `cover_prob_home`: Probability home team covers the spread (0 to 1)
- `cover_prob_away`: Probability away team covers the spread (0 to 1)
- `win_prob_home`: Probability home team wins outright (0 to 1)
- `win_prob_away`: Probability away team wins outright (0 to 1)

### Actual Results
- `actual_home`: Home team final score
- `actual_away`: Away team final score
- `actual_margin`: Actual margin (home score - away score)

---

## How to Analyze

### 1. Check Model Prediction vs Market
```python
import pandas as pd
df = pd.read_csv('predictions/OLD_model_2425_predictions.csv')

# Model more bullish on home than market
df['model_edge'] = df['pred_margin_mu'] - df['baseline_margin']

# Games where model strongly disagrees with market
big_edges = df[abs(df['model_edge']) > 3]
print(big_edges[['date', 'home_team', 'away_team', 'market_spread_home', 
                 'pred_margin_mu', 'model_edge', 'actual_margin']])
```

### 2. Calculate Cover Results
```python
# Did home team cover?
df['actual_home_covers'] = (df['actual_margin'] > -df['market_spread_home']).astype(int)

# What did model predict?
df['model_predicts_home_covers'] = (df['pred_margin_mu'] > -df['market_spread_home']).astype(int)

# Was model correct?
df['model_correct'] = (df['actual_home_covers'] == df['model_predicts_home_covers']).astype(int)

# Win rate
print(f"ATS Accuracy: {df['model_correct'].mean() * 100:.2f}%")
```

### 3. High Confidence Picks
```python
# Games where model is very confident (high cover probability)
high_confidence = df[
    (df['cover_prob_home'] > 0.60) | (df['cover_prob_away'] > 0.60)
].copy()

print(f"High confidence games: {len(high_confidence)}")
print(f"High confidence accuracy: {high_confidence['model_correct'].mean() * 100:.2f}%")
```

### 4. Best Bets (Model Disagrees with Market)
```python
# When model says home covers but market doesn't favor them as much
df['home_favored'] = df['market_spread_home'] < 0
df['away_favored'] = df['market_spread_home'] > 0

# Away favorites strategy
away_fav = df[df['away_favored']].copy()
print(f"Away favorites: {len(away_fav)} games")
print(f"Accuracy: {away_fav['model_correct'].mean() * 100:.2f}%")
```

---

## Quick Analysis Examples

### Example 1: Most Confident Predictions
```python
df = pd.read_csv('predictions/OLD_model_2526_predictions.csv')
df['max_cover_prob'] = df[['cover_prob_home', 'cover_prob_away']].max(axis=1)
top_10 = df.nlargest(10, 'max_cover_prob')

print(top_10[['date', 'home_team', 'away_team', 'market_spread_home', 
              'pred_margin_mu', 'max_cover_prob', 'actual_margin']])
```

### Example 2: Biggest Upsets
```python
# Where model was very wrong
df['prediction_error'] = abs(df['pred_margin_mu'] - df['actual_margin'])
biggest_misses = df.nlargest(10, 'prediction_error')

print(biggest_misses[['date', 'home_team', 'away_team', 'pred_margin_mu', 
                      'actual_margin', 'prediction_error']])
```

### Example 3: Model vs Market Edge
```python
# When model strongly disagreed with market
df['model_edge'] = df['pred_margin_mu'] - df['baseline_margin']
big_disagreements = df[abs(df['model_edge']) > 5]

# How often was model right in these cases?
big_disagreements['model_correct'] = (
    (big_disagreements['actual_home_covers'] == 
     big_disagreements['model_predicts_home_covers']).astype(int)
)

print(f"Big disagreements: {len(big_disagreements)} games")
print(f"Model accuracy when edge > 5 pts: {big_disagreements['model_correct'].mean() * 100:.2f}%")
```

---

## Performance Summary

### 24-25 Season (1,315 games)
- **Overall:** 51.33% ATS
- **Home Favorites:** 53.56% ATS (786 games)
- **Away Favorites:** 48.02% ATS (529 games)

### 25-26 Season (72 games) ⚠️ Small Sample!
- **Overall:** 61.11% ATS
- **Home Favorites:** 48.78% ATS (41 games)
- **Away Favorites:** 67.74% ATS (31 games)

**Note:** 25-26 has very small sample size. The 67.74% on away favorites is based on only 31 games and has ±18pp confidence interval!

---

## Files Available

```
predictions/
├── OLD_model_2425_predictions.csv  (1,315 games, ready for analysis)
├── OLD_model_2526_predictions.csv  (72 games, ready for analysis)
└── coefficient_comparison.csv      (OLD vs NEW model weights)
```

---

## Next Steps

1. **Load the CSVs** in Excel, Python, or your analysis tool of choice
2. **Sort by confidence** to see where model is most certain
3. **Filter by favorite type** to test different strategies
4. **Check predictions vs actuals** to see where model was right/wrong
5. **Look for patterns** in successful vs unsuccessful predictions

The game-by-game data is now ready for your detailed analysis!

