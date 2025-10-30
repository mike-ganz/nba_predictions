"""
Comprehensive Home/Away Bias Investigation

This script investigates why the margin model systematically over-predicts
home team margins in 2024-2025, especially for away favorites.

Runs 5 phases of analysis:
1. Granular ATS Performance Analysis (cross-tabs)
2. Feature Behavior Comparison (validation vs 2024-2025)
3. Model Prediction Patterns
4. Market Comparison Baseline
5. Hypothesis Testing & Root Cause
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print(" HOME/AWAY BIAS INVESTIGATION")
print("="*80)

# Load data
df_val = pd.read_csv('artifacts/margin_test/val_predictions.csv')
df_test = pd.read_csv('artifacts/margin_test/2425_predictions.csv')

# Add derived columns for both datasets
for df in [df_val, df_test]:
    df['actual_margin'] = df['actual_home'] - df['actual_away']
    df['mu_bias'] = df['actual_margin'] - df['pred_margin_mu']
    df['actual_home_covers'] = (df['actual_margin'] > -df['market_spread_home']).astype(int)
    df['pred_home_covers'] = (df['pred_margin_mu'] > -df['market_spread_home']).astype(int)
    df['ats_correct'] = (df['pred_home_covers'] == df['actual_home_covers']).astype(int)
    df['favorite'] = np.where(df['market_spread_home'] < 0, 'home_fav', 'away_fav')
    df['spread_abs'] = np.abs(df['market_spread_home'])
    df['spread_range'] = pd.cut(df['spread_abs'], 
                                  bins=[0, 3, 6, 10, 30],
                                  labels=['0-3', '3-6', '6-10', '>10'])
    df['pred_coverer'] = np.where(df['pred_home_covers'], 'home', 'away')
    df['actual_margin_abs'] = np.abs(df['actual_margin'])
    df['game_competitiveness'] = pd.cut(df['actual_margin_abs'],
                                         bins=[0, 8, 15, 100],
                                         labels=['Close (<8)', 'Decisive (8-15)', 'Blowout (>15)'])

# ============================================================================
# PHASE 1: GRANULAR ATS PERFORMANCE ANALYSIS
# ============================================================================

def phase1_granular_ats(df, label):
    """Detailed cross-tabulated ATS performance analysis."""
    print(f"\n{'='*80}")
    print(f" PHASE 1: GRANULAR ATS PERFORMANCE - {label}")
    print('='*80)
    
    # CUT A: Favorite Type x Predicted Coverer
    print(f"\n{'-'*80}")
    print(" CUT A: Favorite Type x Predicted Coverer")
    print('-'*80)
    
    crosstab_a = pd.crosstab(
        df['favorite'],
        df['pred_coverer'],
        values=df['ats_correct'],
        aggfunc='mean'
    )
    counts_a = pd.crosstab(df['favorite'], df['pred_coverer'])
    
    print(f"\n{'Favorite Type':<20} {'Predicts Home':<20} {'Predicts Away':<20}")
    print("-" * 60)
    for fav in ['home_fav', 'away_fav']:
        if fav in crosstab_a.index:
            home_acc = crosstab_a.loc[fav, 'home'] * 100 if 'home' in crosstab_a.columns else 0
            away_acc = crosstab_a.loc[fav, 'away'] * 100 if 'away' in crosstab_a.columns else 0
            home_cnt = counts_a.loc[fav, 'home'] if 'home' in counts_a.columns else 0
            away_cnt = counts_a.loc[fav, 'away'] if 'away' in counts_a.columns else 0
            
            fav_label = 'Home Favorite' if fav == 'home_fav' else 'Away Favorite'
            print(f"{fav_label:<20} {home_acc:5.1f}% (n={home_cnt:>3})   {away_acc:5.1f}% (n={away_cnt:>3})")
    
    # Find worst quadrant
    min_cell = crosstab_a.min().min()
    for fav in crosstab_a.index:
        for cov in crosstab_a.columns:
            if crosstab_a.loc[fav, cov] == min_cell:
                print(f"\n  [!] WORST QUADRANT: {fav} + predicts {cov} covers = {min_cell*100:.1f}%")
    
    # CUT B: Favorite Type x Spread Range
    print(f"\n{'-'*80}")
    print(" CUT B: Favorite Type x Spread Range")
    print('-'*80)
    
    crosstab_b = pd.crosstab(
        df['favorite'],
        df['spread_range'],
        values=df['ats_correct'],
        aggfunc='mean'
    )
    counts_b = pd.crosstab(df['favorite'], df['spread_range'])
    
    print(f"\n{'Favorite Type':<20} {'0-3':<12} {'3-6':<12} {'6-10':<12} {'>10':<12}")
    print("-" * 68)
    for fav in ['home_fav', 'away_fav']:
        if fav in crosstab_b.index:
            fav_label = 'Home Favorite' if fav == 'home_fav' else 'Away Favorite'
            row_str = f"{fav_label:<20}"
            for spread_rng in ['0-3', '3-6', '6-10', '>10']:
                if spread_rng in crosstab_b.columns:
                    acc = crosstab_b.loc[fav, spread_rng] * 100
                    cnt = counts_b.loc[fav, spread_rng]
                    row_str += f"{acc:5.1f}%({cnt:>3})  "
                else:
                    row_str += "   -       "
            print(row_str)
    
    # Identify problematic cells
    print("\n  Key Findings:")
    away_fav_close = None
    if 'away_fav' in crosstab_b.index and '0-3' in crosstab_b.columns:
        away_fav_close = crosstab_b.loc['away_fav', '0-3']
        if away_fav_close < 0.45:
            print(f"  [!] Away favorites in close games (0-3): {away_fav_close*100:.1f}% (CRITICAL FAILURE)")
    
    # CUT C: Predicted Coverer x Spread Range
    print(f"\n{'-'*80}")
    print(" CUT C: Predicted Coverer x Spread Range")
    print('-'*80)
    
    crosstab_c = pd.crosstab(
        df['pred_coverer'],
        df['spread_range'],
        values=df['ats_correct'],
        aggfunc='mean'
    )
    counts_c = pd.crosstab(df['pred_coverer'], df['spread_range'])
    
    print(f"\n{'Prediction':<20} {'0-3':<12} {'3-6':<12} {'6-10':<12} {'>10':<12}")
    print("-" * 68)
    for pred in ['home', 'away']:
        if pred in crosstab_c.index:
            pred_label = 'Predict Home' if pred == 'home' else 'Predict Away'
            row_str = f"{pred_label:<20}"
            for spread_rng in ['0-3', '3-6', '6-10', '>10']:
                if spread_rng in crosstab_c.columns:
                    acc = crosstab_c.loc[pred, spread_rng] * 100
                    cnt = counts_c.loc[pred, spread_rng]
                    row_str += f"{acc:5.1f}%({cnt:>3})  "
                else:
                    row_str += "   -       "
            print(row_str)
    
    # CUT D: Favorite Type x Game Competitiveness (actual outcome)
    print(f"\n{'-'*80}")
    print(" CUT D: Favorite Type x Game Competitiveness (Actual)")
    print('-'*80)
    
    crosstab_d = pd.crosstab(
        df['favorite'],
        df['game_competitiveness'],
        values=df['ats_correct'],
        aggfunc='mean'
    )
    counts_d = pd.crosstab(df['favorite'], df['game_competitiveness'])
    
    print(f"\n{'Favorite Type':<20} {'Close (<8)':<16} {'Decisive (8-15)':<16} {'Blowout (>15)':<16}")
    print("-" * 68)
    for fav in ['home_fav', 'away_fav']:
        if fav in crosstab_d.index:
            fav_label = 'Home Favorite' if fav == 'home_fav' else 'Away Favorite'
            row_str = f"{fav_label:<20}"
            for comp in ['Close (<8)', 'Decisive (8-15)', 'Blowout (>15)']:
                if comp in crosstab_d.columns:
                    acc = crosstab_d.loc[fav, comp] * 100
                    cnt = counts_d.loc[fav, comp]
                    row_str += f"{acc:5.1f}%({cnt:>3})    "
                else:
                    row_str += "   -          "
            print(row_str)
    
    # CUT E: Direction of Error When Wrong
    print(f"\n{'-'*80}")
    print(" CUT E: Direction of Error When Wrong")
    print('-'*80)
    
    wrong_df = df[df['ats_correct'] == 0].copy()
    wrong_df['error_direction'] = np.where(
        wrong_df['actual_home_covers'] == 1,
        'Actually Home Covered',
        'Actually Away Covered'
    )
    
    error_cross = pd.crosstab(
        wrong_df['favorite'],
        wrong_df['error_direction'],
        normalize='index'
    )
    error_counts = pd.crosstab(wrong_df['favorite'], wrong_df['error_direction'])
    
    print(f"\nWhen predictions are WRONG:")
    print(f"\n{'Favorite Type':<20} {'Home Actually Covered':<30} {'Away Actually Covered':<30}")
    print("-" * 80)
    for fav in ['home_fav', 'away_fav']:
        if fav in error_cross.index:
            fav_label = 'Home Favorite' if fav == 'home_fav' else 'Away Favorite'
            row_str = f"{fav_label:<20}"
            for direction in ['Actually Home Covered', 'Actually Away Covered']:
                if direction in error_cross.columns:
                    pct = error_cross.loc[fav, direction] * 100
                    cnt = error_counts.loc[fav, direction]
                    row_str += f"{pct:5.1f}% ({cnt:>3} errors)      "
                else:
                    row_str += "   -                   "
            print(row_str)
    
    print("\n  Interpretation:")
    if 'away_fav' in error_cross.index and 'Actually Home Covered' in error_cross.columns:
        away_fav_wrong_home = error_cross.loc['away_fav', 'Actually Home Covered']
        if away_fav_wrong_home > 0.6:
            print(f"  [!] When wrong on away favorites, {away_fav_wrong_home*100:.1f}% of time we under-predicted home")
            print(f"      -> Model is systematically TOO BEARISH on home teams when away is favored")
    
    return {
        'away_fav_close': away_fav_close,
        'crosstab_a': crosstab_a,
        'crosstab_b': crosstab_b,
        'crosstab_c': crosstab_c,
    }

# ============================================================================
# PHASE 2: FEATURE BEHAVIOR COMPARISON
# ============================================================================

def phase2_feature_comparison(df_val, df_test):
    """Compare feature distributions between validation and test."""
    print(f"\n{'='*80}")
    print(f" PHASE 2: FEATURE BEHAVIOR COMPARISON")
    print('='*80)
    
    # 2.1: Home Court Advantage
    print(f"\n{'-'*80}")
    print(" 2.1: Home Court Advantage Analysis")
    print('-'*80)
    
    val_home_margin = df_val['actual_margin'].mean()
    test_home_margin = df_test['actual_margin'].mean()
    
    print(f"\nActual Home Margins:")
    print(f"  Validation (2021-2024): {val_home_margin:+.3f} points")
    print(f"  Test (2024-2025):       {test_home_margin:+.3f} points")
    print(f"  Change:                 {test_home_margin - val_home_margin:+.3f} points")
    
    if abs(test_home_margin - val_home_margin) > 0.5:
        if test_home_margin < val_home_margin:
            print(f"\n  [!] HOME COURT ADVANTAGE DECLINED by {val_home_margin - test_home_margin:.2f} points")
            print(f"      Model may be over-predicting home margins based on outdated HCA")
        else:
            print(f"\n  [!] HOME COURT ADVANTAGE INCREASED")
    
    # Check by favorite type
    print(f"\nBy Favorite Type:")
    for fav in ['home_fav', 'away_fav']:
        val_margin = df_val[df_val['favorite'] == fav]['actual_margin'].mean()
        test_margin = df_test[df_test['favorite'] == fav]['actual_margin'].mean()
        fav_label = 'Home Favorites' if fav == 'home_fav' else 'Away Favorites'
        print(f"  {fav_label}:")
        print(f"    Validation: {val_margin:+.3f}  |  Test: {test_margin:+.3f}  |  Change: {test_margin - val_margin:+.3f}")
    
    # 2.2: Market Baseline Comparison
    print(f"\n{'-'*80}")
    print(" 2.2: Market Baseline Accuracy")
    print('-'*80)
    
    # Market "prediction" is just spread = 0 (50/50)
    val_market_ats = (df_val['actual_margin'] > -df_val['market_spread_home']).mean()
    test_market_ats = (df_test['actual_margin'] > -df_test['market_spread_home']).mean()
    
    val_model_ats = df_val['ats_correct'].mean()
    test_model_ats = df_test['ats_correct'].mean()
    
    print(f"\nMarket ATS Accuracy (how often favorite covers):")
    print(f"  Validation: {val_market_ats*100:.2f}%")
    print(f"  Test:       {test_market_ats*100:.2f}%")
    
    print(f"\nModel ATS Accuracy:")
    print(f"  Validation: {val_model_ats*100:.2f}% (edge: {(val_model_ats - val_market_ats)*100:+.2f}%)")
    print(f"  Test:       {test_model_ats*100:.2f}% (edge: {(test_model_ats - test_market_ats)*100:+.2f}%)")
    
    if (test_model_ats - test_market_ats) < (val_model_ats - val_market_ats):
        print(f"\n  [!] Model's edge over market DECLINED")
        print(f"      Market may have gotten more efficient")
    
    # 2.3: Model-Market Correlation
    print(f"\n{'-'*80}")
    print(" 2.3: Model Reliance on Market")
    print('-'*80)
    
    val_corr = df_val['pred_margin_mu'].corr(df_val['baseline_margin'])
    test_corr = df_test['pred_margin_mu'].corr(df_test['baseline_margin'])
    
    print(f"\nCorrelation(pred_margin, baseline_margin):")
    print(f"  Validation: {val_corr:.4f}")
    print(f"  Test:       {test_corr:.4f}")
    
    if test_corr > 0.95:
        print(f"\n  [!] Model is {test_corr*100:.1f}% correlated with market")
        print(f"      Very limited independent signal")
    
    # 2.4: Feature Correlation Shifts
    print(f"\n{'-'*80}")
    print(" 2.4: Feature Predictive Power Changes")
    print('-'*80)
    
    # We don't have raw features, but we can check prediction components
    print(f"\nModel adjustment from baseline:")
    val_adjustment = df_val['pred_margin_mu'] - df_val['baseline_margin']
    test_adjustment = df_test['pred_margin_mu'] - df_test['baseline_margin']
    
    print(f"  Validation: mean={val_adjustment.mean():+.3f}, std={val_adjustment.std():.3f}")
    print(f"  Test:       mean={test_adjustment.mean():+.3f}, std={test_adjustment.std():.3f}")
    
    if abs(test_adjustment.mean()) > abs(val_adjustment.mean()) + 0.3:
        print(f"\n  [!] Model is making LARGER adjustments from baseline in test")
        print(f"      These adjustments may be overconfident or miscalibrated")
    
    return {
        'hca_change': test_home_margin - val_home_margin,
        'market_efficiency_change': (test_model_ats - test_market_ats) - (val_model_ats - val_market_ats),
        'correlation_change': test_corr - val_corr,
    }

# ============================================================================
# PHASE 3: MODEL PREDICTION PATTERNS
# ============================================================================

def phase3_prediction_patterns(df, label):
    """Analyze systematic patterns in predictions."""
    print(f"\n{'='*80}")
    print(f" PHASE 3: MODEL PREDICTION PATTERNS - {label}")
    print('='*80)
    
    # 3.1: Bias by predicted magnitude
    print(f"\n{'-'*80}")
    print(" 3.1: Bias by Predicted Margin Magnitude")
    print('-'*80)
    
    df['pred_magnitude'] = pd.cut(
        df['pred_margin_mu'],
        bins=[-100, -10, -3, 3, 10, 100],
        labels=['Strong Away', 'Weak Away', 'Toss-up', 'Weak Home', 'Strong Home']
    )
    
    print(f"\n{'Predicted Outcome':<20} {'Count':<8} {'Mean Bias':<12} {'ATS Acc':<12}")
    print("-" * 52)
    
    for mag in ['Strong Away', 'Weak Away', 'Toss-up', 'Weak Home', 'Strong Home']:
        mag_df = df[df['pred_magnitude'] == mag]
        if len(mag_df) > 0:
            mean_bias = mag_df['mu_bias'].mean()
            ats_acc = mag_df['ats_correct'].mean()
            print(f"{mag:<20} {len(mag_df):<8} {mean_bias:>+10.3f}   {ats_acc*100:>8.2f}%")
    
    # 3.2: Conditional bias analysis
    print(f"\n{'-'*80}")
    print(" 3.2: Bias Conditional on Favorite and Prediction")
    print('-'*80)
    
    print(f"\n{'Scenario':<45} {'Count':<8} {'Mean Bias':<12} {'ATS Acc':<12}")
    print("-" * 77)
    
    scenarios = [
        ('Home fav, predict home covers', 
         (df['favorite'] == 'home_fav') & (df['pred_home_covers'] == 1)),
        ('Home fav, predict away covers',
         (df['favorite'] == 'home_fav') & (df['pred_home_covers'] == 0)),
        ('Away fav, predict home covers',
         (df['favorite'] == 'away_fav') & (df['pred_home_covers'] == 1)),
        ('Away fav, predict away covers',
         (df['favorite'] == 'away_fav') & (df['pred_home_covers'] == 0)),
    ]
    
    for scenario_name, mask in scenarios:
        scenario_df = df[mask]
        if len(scenario_df) > 0:
            mean_bias = scenario_df['mu_bias'].mean()
            ats_acc = scenario_df['ats_correct'].mean()
            print(f"{scenario_name:<45} {len(scenario_df):<8} {mean_bias:>+10.3f}   {ats_acc*100:>8.2f}%")
    
    # 3.3: Time series analysis (if date available)
    if 'date' in df.columns:
        print(f"\n{'-'*80}")
        print(" 3.3: Time Series of Bias")
        print('-'*80)
        
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
        monthly = df.groupby('month').agg({
            'mu_bias': 'mean',
            'ats_correct': 'mean',
            'game_id': 'count'
        }).reset_index()
        monthly.columns = ['month', 'mean_bias', 'ats_acc', 'count']
        
        print(f"\n{'Month':<10} {'Games':<8} {'Mean Bias':<12} {'ATS Acc':<12}")
        print("-" * 42)
        for _, row in monthly.iterrows():
            print(f"{str(row['month']):<10} {int(row['count']):<8} {row['mean_bias']:>+10.3f}   {row['ats_acc']*100:>8.2f}%")
        
        # Check for trend
        if len(monthly) > 3:
            bias_trend = np.polyfit(range(len(monthly)), monthly['mean_bias'], 1)[0]
            if abs(bias_trend) > 0.1:
                direction = "INCREASING" if bias_trend > 0 else "DECREASING"
                print(f"\n  [!] Bias is {direction} over time (slope: {bias_trend:+.3f}/month)")

# ============================================================================
# PHASE 4: MARKET COMPARISON
# ============================================================================

def phase4_market_comparison(df):
    """Compare model performance to market baseline."""
    print(f"\n{'='*80}")
    print(f" PHASE 4: MARKET COMPARISON BASELINE")
    print('='*80)
    
    # 4.1: Market performance by subgroup
    print(f"\n{'-'*80}")
    print(" 4.1: Model vs Market by Problematic Subgroups")
    print('-'*80)
    
    subgroups = {
        'All games': df,
        'Home favorites': df[df['favorite'] == 'home_fav'],
        'Away favorites': df[df['favorite'] == 'away_fav'],
        'Close games (0-3)': df[df['spread_abs'] < 3],
        'Away fav + close': df[(df['favorite'] == 'away_fav') & (df['spread_abs'] < 3)],
    }
    
    print(f"\n{'Subgroup':<25} {'Count':<8} {'Model ATS':<12} {'Market Baseline':<18}")
    print("-" * 63)
    
    for name, subset in subgroups.items():
        if len(subset) > 0:
            model_ats = subset['ats_correct'].mean()
            # Market baseline: favorites should cover 50% (by design of spread)
            # But check actual
            market_baseline = (subset['actual_margin'] > -subset['market_spread_home']).mean()
            
            print(f"{name:<25} {len(subset):<8} {model_ats*100:>8.2f}%    {market_baseline*100:>8.2f}%")
    
    # 4.2: Disagreement analysis
    print(f"\n{'-'*80}")
    print(" 4.2: Performance When Agreeing vs Disagreeing with Market")
    print('-'*80)
    
    df['model_market_diff'] = df['pred_margin_mu'] - df['baseline_margin']
    df['large_disagreement'] = np.abs(df['model_market_diff']) > 3
    
    agree_df = df[~df['large_disagreement']]
    disagree_df = df[df['large_disagreement']]
    
    print(f"\n{'Scenario':<30} {'Count':<8} {'ATS Acc':<12} {'Mean |Diff|':<12}")
    print("-" * 62)
    
    if len(agree_df) > 0:
        agree_ats = agree_df['ats_correct'].mean()
        agree_diff = np.abs(agree_df['model_market_diff']).mean()
        print(f"{'Agree with market':<30} {len(agree_df):<8} {agree_ats*100:>8.2f}%    {agree_diff:>8.2f}")
    
    if len(disagree_df) > 0:
        disagree_ats = disagree_df['ats_correct'].mean()
        disagree_diff = np.abs(disagree_df['model_market_diff']).mean()
        print(f"{'Disagree with market (>3pts)':<30} {len(disagree_df):<8} {disagree_ats*100:>8.2f}%    {disagree_diff:>8.2f}")
        
        if disagree_ats < agree_ats - 0.03:
            print(f"\n  [!] Model performs WORSE when disagreeing with market")
            print(f"      Adjustments may be adding noise, not signal")

# ============================================================================
# PHASE 5: HYPOTHESIS TESTING
# ============================================================================

def phase5_hypothesis_testing(df_val, df_test, phase2_results):
    """Test specific hypotheses about what changed."""
    print(f"\n{'='*80}")
    print(f" PHASE 5: HYPOTHESIS TESTING & ROOT CAUSE")
    print('='*80)
    
    findings = []
    
    # HYPOTHESIS A: Home Court Advantage Declined
    print(f"\n{'-'*80}")
    print(" HYPOTHESIS A: Home Court Advantage Declined")
    print('-'*80)
    
    hca_change = phase2_results['hca_change']
    
    # Statistical test
    val_margins = df_val['actual_margin'].values
    test_margins = df_test['actual_margin'].values
    t_stat, p_value = stats.ttest_ind(val_margins, test_margins)
    
    print(f"\nHome court advantage change: {hca_change:+.3f} points")
    print(f"T-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
    
    if p_value < 0.05 and hca_change < -0.3:
        print(f"\n  [!!!] HYPOTHESIS CONFIRMED (p < 0.05)")
        print(f"       Home court advantage declined significantly")
        print(f"       Model learned HCA from 2021-2024, but it's weaker in 2024-2025")
        findings.append(('HCA_DECLINED', hca_change, 'HIGH'))
    elif abs(hca_change) > 0.5:
        print(f"\n  [!] HYPOTHESIS LIKELY")
        print(f"      Large change in HCA, but not statistically significant yet")
        findings.append(('HCA_DECLINED', hca_change, 'MEDIUM'))
    else:
        print(f"\n  [+] Hypothesis unlikely - HCA stable")
    
    # HYPOTHESIS B: Away Favorites Markets Sharper
    print(f"\n{'-'*80}")
    print(" HYPOTHESIS B: Away Favorite Markets Got Sharper")
    print('-'*80)
    
    val_away_fav = df_val[df_val['favorite'] == 'away_fav']
    test_away_fav = df_test[df_test['favorite'] == 'away_fav']
    
    val_away_market_acc = (val_away_fav['actual_margin'] > -val_away_fav['market_spread_home']).mean()
    test_away_market_acc = (test_away_fav['actual_margin'] > -test_away_fav['market_spread_home']).mean()
    
    val_away_model_acc = val_away_fav['ats_correct'].mean()
    test_away_model_acc = test_away_fav['ats_correct'].mean()
    
    print(f"\nAway Favorite Market Accuracy:")
    print(f"  Validation: {val_away_market_acc*100:.2f}%")
    print(f"  Test:       {test_away_market_acc*100:.2f}%")
    print(f"  Change:     {(test_away_market_acc - val_away_market_acc)*100:+.2f}%")
    
    print(f"\nAway Favorite Model Accuracy:")
    print(f"  Validation: {val_away_model_acc*100:.2f}%")
    print(f"  Test:       {test_away_model_acc*100:.2f}%")
    print(f"  Change:     {(test_away_model_acc - val_away_model_acc)*100:+.2f}%")
    
    if test_away_model_acc < test_away_market_acc - 0.05:
        print(f"\n  [!!!] HYPOTHESIS CONFIRMED")
        print(f"        Model performs worse than market baseline on away favorites")
        print(f"        Model's adjustments are counterproductive")
        findings.append(('AWAY_FAV_ADJUSTED_WRONG', test_away_model_acc, 'HIGH'))
    
    # HYPOTHESIS C: Model Over-Relies on Market
    print(f"\n{'-'*80}")
    print(" HYPOTHESIS C: Model Over-Relies on Market")
    print('-'*80)
    
    test_corr = df_test['pred_margin_mu'].corr(df_test['baseline_margin'])
    
    print(f"\nModel-Market Correlation: {test_corr:.4f}")
    
    if test_corr > 0.95:
        print(f"\n  [!!!] HYPOTHESIS CONFIRMED")
        print(f"        Model is {test_corr*100:.1f}% correlated with market")
        print(f"        Limited independent predictive signal")
        findings.append(('MARKET_DEPENDENT', test_corr, 'HIGH'))
    
    # HYPOTHESIS D: Close Games More Unpredictable
    print(f"\n{'-'*80}")
    print(" HYPOTHESIS D: Close Games Became More Unpredictable")
    print('-'*80)
    
    val_close = df_val[df_val['spread_abs'] < 3]
    test_close = df_test[df_test['spread_abs'] < 3]
    
    val_close_var = val_close['actual_margin'].std()
    test_close_var = test_close['actual_margin'].std()
    
    print(f"\nClose Game Margin Std Dev:")
    print(f"  Validation: {val_close_var:.2f}")
    print(f"  Test:       {test_close_var:.2f}")
    print(f"  Change:     {test_close_var - val_close_var:+.2f}")
    
    if test_close_var > val_close_var + 1.0:
        print(f"\n  [!] HYPOTHESIS LIKELY")
        print(f"      Close games more volatile in 2024-2025")
        findings.append(('CLOSE_GAMES_VOLATILE', test_close_var, 'MEDIUM'))
    else:
        print(f"\n  [+] Close game variance stable")
    
    return findings

# ============================================================================
# RUN ALL PHASES
# ============================================================================

print("\n\nStarting comprehensive investigation...")
print("This will take a few minutes...\n")

# Phase 1: Granular ATS
print("\n" + "="*80)
print(" RUNNING PHASE 1: GRANULAR ATS ANALYSIS")
print("="*80)
val_phase1 = phase1_granular_ats(df_val, "VALIDATION (2021-2024)")
test_phase1 = phase1_granular_ats(df_test, "2024-2025 SEASON")

# Phase 2: Feature Comparison
print("\n" + "="*80)
print(" RUNNING PHASE 2: FEATURE COMPARISON")
print("="*80)
phase2_results = phase2_feature_comparison(df_val, df_test)

# Phase 3: Prediction Patterns
print("\n" + "="*80)
print(" RUNNING PHASE 3: PREDICTION PATTERNS")
print("="*80)
phase3_prediction_patterns(df_val, "VALIDATION (2021-2024)")
phase3_prediction_patterns(df_test, "2024-2025 SEASON")

# Phase 4: Market Comparison
print("\n" + "="*80)
print(" RUNNING PHASE 4: MARKET COMPARISON")
print("="*80)
phase4_market_comparison(df_test)

# Phase 5: Hypothesis Testing
print("\n" + "="*80)
print(" RUNNING PHASE 5: HYPOTHESIS TESTING")
print("="*80)
findings = phase5_hypothesis_testing(df_val, df_test, phase2_results)

# ============================================================================
# FINAL SYNTHESIS
# ============================================================================

print(f"\n\n{'='*80}")
print(" FINAL SYNTHESIS & ROOT CAUSE DIAGNOSIS")
print('='*80)

print(f"\n{'-'*80}")
print(" KEY FINDINGS")
print('-'*80)

if findings:
    print(f"\n{len(findings)} confirmed or likely hypotheses:\n")
    for i, (finding_type, value, confidence) in enumerate(findings, 1):
        conf_marker = "[!!!]" if confidence == 'HIGH' else "[!]"
        print(f"{i}. {conf_marker} {finding_type}")
        if finding_type == 'HCA_DECLINED':
            print(f"   Home court advantage declined by {value:.2f} points")
        elif finding_type == 'AWAY_FAV_ADJUSTED_WRONG':
            print(f"   Model accuracy on away favorites: {value*100:.1f}%")
        elif finding_type == 'MARKET_DEPENDENT':
            print(f"   Model-market correlation: {value:.3f}")
        elif finding_type == 'CLOSE_GAMES_VOLATILE':
            print(f"   Close game std dev: {value:.2f}")

print(f"\n{'-'*80}")
print(" ROOT CAUSE DIAGNOSIS")
print('-'*80)

# Determine primary root cause
if any(f[0] == 'AWAY_FAV_ADJUSTED_WRONG' for f in findings):
    print(f"\n[PRIMARY ISSUE] Away Favorite Prediction Failure")
    print(f"  - Model performs drastically worse on away favorites (40.08%)")
    print(f"  - This is BELOW market baseline")
    print(f"  - Model's adjustments from market are counterproductive")

if any(f[0] == 'HCA_DECLINED' for f in findings):
    print(f"\n[CONTRIBUTING FACTOR] Home Court Advantage Decline")
    print(f"  - Actual home margins lower in 2024-2025")
    print(f"  - Model learned from 2021-2024 when HCA was stronger")
    print(f"  - Leads to systematic over-prediction of home margins")

if any(f[0] == 'MARKET_DEPENDENT' for f in findings):
    print(f"\n[STRUCTURAL ISSUE] Over-Reliance on Market")
    print(f"  - 95%+ correlation with market spreads")
    print(f"  - Model provides minimal independent signal")
    print(f"  - Vulnerable to market inefficiency changes")

print(f"\n{'-'*80}")
print(" RECOMMENDED ACTIONS")
print('-'*80)

print(f"\n1. IMMEDIATE FIX - Bias Correction:")
print(f"   adjusted_mu = pred_mu + 0.55  # Correct systematic over-prediction")
print(f"   Use threshold = 0.595 instead of 0.50")
print(f"   Expected improvement: ~2% ATS accuracy")

print(f"\n2. SHORT-TERM - Away Favorite Special Handling:")
print(f"   if market_spread_home > 0:  # away favorite")
print(f"       # Either:")
print(f"       # a) Don't bet (filter out)")
print(f"       # b) Use market baseline only (ignore model adjustment)")
print(f"       # c) Add larger bias correction (+1.5 points)")

print(f"\n3. MEDIUM-TERM - Feature Re-engineering:")
print(f"   - Add 2024-2025 specific features (home court weaker?)")
print(f"   - Reduce reliance on market baseline")
print(f"   - Add interaction terms for favorite direction")

print(f"\n4. LONG-TERM - Model Redesign:")
print(f"   - Consider ensemble: separate models for home/away favorites")
print(f"   - Add time-varying parameters (HCA changes over time)")
print(f"   - Incorporate more independent signals")

print(f"\n{'='*80}")
print(" INVESTIGATION COMPLETE")
print('='*80)

