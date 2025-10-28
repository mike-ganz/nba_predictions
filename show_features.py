"""Display current features with a real game example."""
import json

# Read one game from the training data
with open("data/games_train.jsonl", "r") as f:
    game = json.loads(f.readline())

print("=" * 80)
print("CURRENT FEATURES (Example Game)")
print("=" * 80)
print(f"\nGame: {game['teams']['A']['team_name']} @ {game['teams']['H']['team_name']}")
print(f"Date: {game['date']}")
print(f"Final Score: {game['teams']['A']['team_name']} {game['outcome']['away_final']}, {game['teams']['H']['team_name']} {game['outcome']['home_final']}")
print(f"Spread: {game['teams']['H']['team_name']} {game['market']['spread_home']}")

print("\n" + "=" * 80)
print("TEAM-LEVEL FEATURES (per team)")
print("=" * 80)

away_team = game['teams']['A']
home_team = game['teams']['H']

features_list = [
    ("off_rating", "Offensive Rating (pts per 100 poss)", "higher = better offense"),
    ("def_rating", "Defensive Rating (pts allowed per 100)", "lower = better defense"),
    ("pace", "Pace (possessions per 48 min)", "how fast they play"),
    ("three_pt_rate", "3-Point Rate (3PA / FGA)", "how often they shoot 3s"),
    ("free_throw_rate", "Free Throw Rate (FTA / FGA)", "how often they get to the line"),
    ("off_reb_rate", "Offensive Rebound Rate", "% of available off rebounds"),
    ("def_reb_rate", "Defensive Rebound Rate", "% of available def rebounds"),
    ("assist_rate", "Assist Rate (AST / FGM)", "% of made FGs assisted"),
    ("turnover_rate", "Turnover Rate (TOV / possessions)", "how often they turn it over"),
    ("rest_days", "Rest Days", "days since last game"),
]

print(f"\n{away_team['team_name']} (Away):")
for key, desc, meaning in features_list:
    value = away_team.get(key, "N/A")
    print(f"  {desc:40s}: {value:8} ({meaning})")

print(f"\n{home_team['team_name']} (Home):")
for key, desc, meaning in features_list:
    value = home_team.get(key, "N/A")
    print(f"  {desc:40s}: {value:8} ({meaning})")

print("\n" + "=" * 80)
print("DERIVED MATCHUP FEATURES (computed from team features)")
print("=" * 80)
print("\nThese are calculated in features/matchup.py:")
print("  • edge = (team_off_rating - opp_def_rating) - (opp_off_rating - team_def_rating)")
print("  • orb_edge = team_off_reb_rate - opp_def_reb_rate")
print("  • tov_edge = opp_turnover_rate - team_turnover_rate")
print("  • tpar = team's 3-point rate")
print("  • ftr = team's free throw rate")
print("  • rest_days = team's rest days")
print("  • pace_mean = average of both teams' pace")

# Calculate example edges for this game
away_edge = (away_team['off_rating'] - home_team['def_rating']) - (home_team['off_rating'] - away_team['def_rating'])
home_edge = (home_team['off_rating'] - away_team['def_rating']) - (away_team['off_rating'] - home_team['def_rating'])

print(f"\nExample calculated for this game:")
print(f"  {away_team['team_name']} edge: {away_edge:.2f}")
print(f"  {home_team['team_name']} edge: {home_edge:.2f}")

print("\n" + "=" * 80)
print("MARKET FEATURES (from betting markets)")
print("=" * 80)
print(f"  Spread (home): {game['market']['spread_home']}")
print(f"  Total: {game['market']['total']}")
print(f"  Moneyline (home): {game['market']['moneyline_home']}")
print(f"  Moneyline (away): {game['market']['moneyline_away']}")
print("\nDerived:")
print("  • implied_home_winprob = converted from moneyline")
print("  • implied_away_winprob = converted from moneyline")

print("\n" + "=" * 80)
print("AVAILABILITY FEATURES (player-specific, when available)")
print("=" * 80)
print("  • minutes_missing_top2 = minutes from top 2 players who are out")
print("  • star_out = 1 if star player out, 0 otherwise")
print("  • usage_share_top2 = combined usage % of top 2 available players")
print("  • team_weighted_ts = team's weighted true shooting %")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\nPer-Team Features (9 each for home/away):")
print("  1. edge (net rating advantage)")
print("  2. orb_edge (offensive rebounding advantage)")
print("  3. tov_edge (turnover differential advantage)")
print("  4. tpar (3-point rate)")
print("  5. ftr (free throw rate)")
print("  6. rest_days")
print("  7. minutes_missing_top2")
print("  8. star_out")
print("  9. usage_share_top2")
print("\nShared Features (5 total):")
print("  1. pace_mean (game pace)")
print("  2. implied_home_winprob (from market)")
print("  3. implied_away_winprob (from market)")
print("  4. team_weighted_ts_home (shooting efficiency)")
print("  5. team_weighted_ts_away (shooting efficiency)")
print("\nTotal: 23 features")
print("=" * 80)

