import pandas as pd

df = pd.read_csv('predictions/logistic_2425_test.csv')
df = df[df['actual_home_covers'].notna()]

actual_home_rate = df['actual_home_covers'].mean()
predicted_home_rate = (df['cover_prob_home'] > 0.5).sum() / len(df)

print("="*60)
print("HOME vs AWAY BIAS ANALYSIS")
print("="*60)
print(f"\nActual home cover rate: {actual_home_rate:.1%}")
print(f"Model predicts home > 50%: {predicted_home_rate:.1%}")
print(f"\nBias: Model underestimates home by {(actual_home_rate - predicted_home_rate)*100:.1f} percentage points!")

# Check if model is systematically wrong
print("\n" + "="*60)
print("IF WE JUST BET AGAINST THE MODEL:")
print("="*60)

# Model says away (home < 50%), so bet home
away_picks = df[df['cover_prob_home'] < 0.5]
home_picks = df[df['cover_prob_home'] > 0.5]

print(f"\nModel picks away (home prob < 50%): {len(away_picks)} games")
if len(away_picks) > 0:
    away_correct = (away_picks['actual_home_covers'] == 0).mean()
    home_wins_when_model_says_away = (away_picks['actual_home_covers'] == 1).mean()
    print(f"  Model correct (away covered): {away_correct:.1%}")
    print(f"  Home actually covered: {home_wins_when_model_says_away:.1%}")
    print(f"  If we FADED and bet HOME: {home_wins_when_model_says_away:.1%} accuracy!")

print(f"\nModel picks home (home prob > 50%): {len(home_picks)} games")
if len(home_picks) > 0:
    home_correct = (home_picks['actual_home_covers'] == 1).mean()
    print(f"  Model correct (home covered): {home_correct:.1%}")

