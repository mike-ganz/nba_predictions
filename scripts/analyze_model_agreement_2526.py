"""
Analyze agreement between old and new models on 25-26 season.
Shows when models agree/disagree and who's right when they disagree.
"""
import pandas as pd
import numpy as np

print("="*100)
print("MODEL AGREEMENT ANALYSIS: OLD (21-24) vs NEW (21-25)")
print("2025-26 Season (72 games)")
print("="*100)
print()

# Load predictions from both models
old_pred = pd.read_csv('predictions/current_season_2025_2026_predictions.csv')
new_pred = pd.read_csv('predictions/current_season_2025_2026_predictions_NEW_MODEL.csv')

print(f"Total games: {len(old_pred)}")
print()

# Merge on game_id
merged = old_pred.merge(new_pred, on='game_id', suffixes=('_old', '_new'))

# Calculate ATS predictions for both
merged['actual_margin'] = merged['actual_home_old'] - merged['actual_away_old']
merged['actual_home_covers'] = (merged['actual_margin'] > -merged['market_spread_home_old']).astype(int)

merged['old_picks_home'] = (merged['pred_margin_mu_old'] > -merged['market_spread_home_old']).astype(int)
merged['new_picks_home'] = (merged['pred_margin_mu_new'] > -merged['market_spread_home_new']).astype(int)

merged['old_correct'] = (merged['old_picks_home'] == merged['actual_home_covers']).astype(int)
merged['new_correct'] = (merged['new_picks_home'] == merged['actual_home_covers']).astype(int)

# Calculate agreement
merged['models_agree'] = (merged['old_picks_home'] == merged['new_picks_home']).astype(int)
merged['models_disagree'] = 1 - merged['models_agree']

# Overall agreement rate
agreement_rate = merged['models_agree'].mean() * 100
disagreement_rate = 100 - agreement_rate

print("="*100)
print("OVERALL AGREEMENT")
print("="*100)
print()
print(f"Games where models AGREE:    {merged['models_agree'].sum()} ({agreement_rate:.2f}%)")
print(f"Games where models DISAGREE: {merged['models_disagree'].sum()} ({disagreement_rate:.2f}%)")
print()

# When they agree, how often are they right?
agree_games = merged[merged['models_agree'] == 1]
agree_correct_both = (agree_games['old_correct'] & agree_games['new_correct']).sum()
agree_wrong_both = ((1 - agree_games['old_correct']) & (1 - agree_games['new_correct'])).sum()

print("When models AGREE:")
print(f"  Both CORRECT: {agree_correct_both} ({agree_correct_both/len(agree_games)*100:.2f}%)")
print(f"  Both WRONG:   {agree_wrong_both} ({agree_wrong_both/len(agree_games)*100:.2f}%)")
print()

# When they disagree, who's right more often?
disagree_games = merged[merged['models_disagree'] == 1]
old_right_new_wrong = (disagree_games['old_correct'] & (1 - disagree_games['new_correct'])).sum()
new_right_old_wrong = (disagree_games['new_correct'] & (1 - disagree_games['old_correct'])).sum()
both_wrong = ((1 - disagree_games['old_correct']) & (1 - disagree_games['new_correct'])).sum()

print("When models DISAGREE:")
print(f"  Old RIGHT, New WRONG: {old_right_new_wrong} ({old_right_new_wrong/len(disagree_games)*100:.2f}%)")
print(f"  New RIGHT, Old WRONG: {new_right_old_wrong} ({new_right_old_wrong/len(disagree_games)*100:.2f}%)")
print(f"  Both WRONG:          {both_wrong} ({both_wrong/len(disagree_games)*100:.2f}%)")
print()

if old_right_new_wrong > new_right_old_wrong:
    print(f"✅ When they disagree, OLD MODEL is right more often (+{old_right_new_wrong - new_right_old_wrong} games)")
elif new_right_old_wrong > old_right_new_wrong:
    print(f"✅ When they disagree, NEW MODEL is right more often (+{new_right_old_wrong - old_right_new_wrong} games)")
else:
    print("🤝 When they disagree, both models are right equally often")
print()

# Profile disagreement games
print("="*100)
print("PROFILE OF DISAGREEMENT GAMES")
print("="*100)
print()

# Add categorical variables
disagree_games = disagree_games.copy()
disagree_games['home_favored'] = disagree_games['market_spread_home_old'] < 0
disagree_games['away_favored'] = disagree_games['market_spread_home_old'] > 0
disagree_games['spread_abs'] = disagree_games['market_spread_home_old'].abs()
disagree_games['spread_small'] = disagree_games['spread_abs'] <= 3.5
disagree_games['spread_medium'] = (disagree_games['spread_abs'] > 3.5) & (disagree_games['spread_abs'] <= 7.5)
disagree_games['spread_large'] = disagree_games['spread_abs'] > 7.5

def profile_segment(df, segment_name):
    if len(df) == 0:
        return None
    old_wins = df['old_correct'].sum()
    new_wins = df['new_correct'].sum()
    old_pct = old_wins / len(df) * 100
    new_pct = new_wins / len(df) * 100
    return {
        'segment': segment_name,
        'n_games': len(df),
        'pct_of_disagreements': len(df) / len(disagree_games) * 100,
        'old_wins': old_wins,
        'old_win_pct': old_pct,
        'new_wins': new_wins,
        'new_win_pct': new_pct,
        'winner': 'Old' if old_wins > new_wins else ('New' if new_wins > old_wins else 'Tie')
    }

results = []

# By favorite type
results.append(profile_segment(disagree_games[disagree_games['home_favored']], 'Home Favored'))
results.append(profile_segment(disagree_games[disagree_games['away_favored']], 'Away Favored'))

# By spread size
results.append(profile_segment(disagree_games[disagree_games['spread_small']], 'Small Spread (≤3.5)'))
results.append(profile_segment(disagree_games[disagree_games['spread_medium']], 'Medium Spread (3.5-7.5)'))
results.append(profile_segment(disagree_games[disagree_games['spread_large']], 'Large Spread (>7.5)'))

# Favorite × Spread
results.append(profile_segment(
    disagree_games[disagree_games['home_favored'] & disagree_games['spread_small']], 
    'Home Fav + Small'
))
results.append(profile_segment(
    disagree_games[disagree_games['home_favored'] & disagree_games['spread_medium']], 
    'Home Fav + Med'
))
results.append(profile_segment(
    disagree_games[disagree_games['home_favored'] & disagree_games['spread_large']], 
    'Home Fav + Large'
))
results.append(profile_segment(
    disagree_games[disagree_games['away_favored'] & disagree_games['spread_small']], 
    'Away Fav + Small'
))
results.append(profile_segment(
    disagree_games[disagree_games['away_favored'] & disagree_games['spread_medium']], 
    'Away Fav + Med'
))
results.append(profile_segment(
    disagree_games[disagree_games['away_favored'] & disagree_games['spread_large']], 
    'Away Fav + Large'
))

results = [r for r in results if r is not None]
results_df = pd.DataFrame(results)

print(f"{'Segment':<30} {'Games':<8} {'% of Disagree':<15} {'Old Wins':<10} {'Old %':<10} {'New Wins':<10} {'New %':<10} {'Winner':<8}")
print("-"*100)
for _, row in results_df.iterrows():
    print(f"{row['segment']:<30} {row['n_games']:<8} {row['pct_of_disagreements']:>13.1f}% "
          f"{row['old_wins']:>9} {row['old_win_pct']:>9.1f}% {row['new_wins']:>9} {row['new_win_pct']:>9.1f}% "
          f"{row['winner']:<8}")

print()

# Show where old model dominates
print("="*100)
print("SEGMENTS WHERE OLD MODEL DOMINATES (When They Disagree)")
print("="*100)
print()

old_dominant = results_df[results_df['old_win_pct'] - results_df['new_win_pct'] > 5].sort_values('old_win_pct', ascending=False)
if len(old_dominant) > 0:
    print(f"{'Segment':<30} {'Games':<8} {'Old Win %':<12} {'New Win %':<12} {'Gap':<10}")
    print("-"*100)
    for _, row in old_dominant.iterrows():
        gap = row['old_win_pct'] - row['new_win_pct']
        print(f"{row['segment']:<30} {row['n_games']:<8} {row['old_win_pct']:>10.1f}% {row['new_win_pct']:>10.1f}% {gap:>9.1f}%")
else:
    print("No segments where old model dominates by >5%")

print()

# Show where new model dominates
print("="*100)
print("SEGMENTS WHERE NEW MODEL DOMINATES (When They Disagree)")
print("="*100)
print()

new_dominant = results_df[results_df['new_win_pct'] - results_df['old_win_pct'] > 5].sort_values('new_win_pct', ascending=False)
if len(new_dominant) > 0:
    print(f"{'Segment':<30} {'Games':<8} {'Old Win %':<12} {'New Win %':<12} {'Gap':<10}")
    print("-"*100)
    for _, row in new_dominant.iterrows():
        gap = row['new_win_pct'] - row['old_win_pct']
        print(f"{row['segment']:<30} {row['n_games']:<8} {row['old_win_pct']:>10.1f}% {row['new_win_pct']:>10.1f}% {gap:>9.1f}%")
else:
    print("No segments where new model dominates by >5%")

print()

# Save detailed results
disagree_games_out = disagree_games[['game_id', 'date_old', 'home_team_old', 'away_team_old',
                                      'market_spread_home_old', 'actual_margin',
                                      'pred_margin_mu_old', 'pred_margin_mu_new',
                                      'old_picks_home', 'new_picks_home',
                                      'actual_home_covers', 'old_correct', 'new_correct',
                                      'home_favored', 'away_favored', 'spread_abs']]
disagree_games_out.columns = ['game_id', 'date', 'home_team', 'away_team', 'market_spread',
                               'actual_margin', 'old_pred', 'new_pred', 'old_picks_home',
                               'new_picks_home', 'actual_home_covers', 'old_correct',
                               'new_correct', 'home_favored', 'away_favored', 'spread_abs']
disagree_games_out.to_csv('predictions/model_disagreement_analysis_2526.csv', index=False)

print("="*100)
print("✅ Detailed disagreement games saved to: predictions/model_disagreement_analysis_2526.csv")
print("="*100)
print()

# Summary statistics
print("="*100)
print("SUMMARY")
print("="*100)
print()
print(f"Total Games:              {len(merged)}")
print(f"Agreement Rate:           {agreement_rate:.2f}%")
print(f"Disagreement Rate:        {disagreement_rate:.2f}%")
print()
print(f"When Disagree:")
print(f"  Old Model Wins:         {old_right_new_wrong} ({old_right_new_wrong/len(disagree_games)*100:.2f}%)")
print(f"  New Model Wins:         {new_right_old_wrong} ({new_right_old_wrong/len(disagree_games)*100:.2f}%)")
print(f"  Net Advantage (Old):    +{old_right_new_wrong - new_right_old_wrong} games")
print()
print(f"Overall Accuracy:")
print(f"  Old Model:              {merged['old_correct'].sum()} / {len(merged)} ({merged['old_correct'].mean()*100:.2f}%)")
print(f"  New Model:              {merged['new_correct'].sum()} / {len(merged)} ({merged['new_correct'].mean()*100:.2f}%)")
print()
print("="*100)

