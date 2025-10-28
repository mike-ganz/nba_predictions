import pandas as pd, numpy as np
p = r"reports/run_2425_val/per_game_predictions_2024_2025.csv"
df = pd.read_csv(p)
# Actual cover indicator (home), push=0.5 for calibration scoring
margin = df["actual_home"] - df["actual_away"]
s = df["market_spread_home"].astype(float)
actual = np.where(margin > -s, 1.0, np.where(np.isclose(margin, -s), 0.5, 0.0))
pred = df["cover_prob_home"].astype(float)
brier = np.mean((pred - actual)**2)
print("ATS Brier (home side):", round(brier, 4))

# Reliability by decile
bins = pd.cut(pred, np.linspace(0,1,11), include_lowest=True)
rel = pd.DataFrame({'pred':pred, 'actual':actual}).groupby(bins).agg(pred=('pred','mean'), actual=('actual','mean'), n=('pred','size'))
print(rel)