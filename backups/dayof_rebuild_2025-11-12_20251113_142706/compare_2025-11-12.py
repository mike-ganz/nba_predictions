import pandas as pd
from pathlib import Path

rebuilt = pd.read_csv('predictions/predictions_champion_2025-11-12_rebuilt.csv')
lookup  = pd.read_csv('predictions/current_season_champion_2025_2026_predictions.csv')

rebuilt = rebuilt.sort_values('game_id')
lookup  = lookup[lookup['date']=='2025-11-12'].sort_values('game_id')

merged = rebuilt.merge(lookup[['game_id','market_spread_home','baseline_margin','pred_margin_mu']].rename(columns={
    'market_spread_home':'look_spread','baseline_margin':'look_base','pred_margin_mu':'look_mu' 
}), on='game_id', how='left')

merged['spread_diff'] = merged['market_spread_home'] - merged['look_spread']
merged['base_diff']   = merged['baseline_margin']   - merged['look_base']
merged['mu_diff']     = merged['pred_margin_mu']    - merged['look_mu']

print('\n==== REBUILT vs LOOKBACK (2025-11-12) ====')
cols = ['game_id','market_spread_home','look_spread','baseline_margin','look_base','pred_margin_mu','look_mu','spread_diff','base_diff','mu_diff']
print(merged[cols].to_string(index=False))

mismatch = merged[merged['mu_diff'].abs()>0.25]
print('\nMismatches (|mu_diff|>0.25):', len(mismatch))
if len(mismatch):
    print(mismatch[['game_id','mu_diff']].to_string(index=False))
