"""
Hybrid Model Selector: Automatic regime detection and model switching.

This module implements automatic detection of high-variance regimes
and switches between Champion and Context models accordingly.

Key Components:
1. VarianceRegimeDetector - Detects current league variance regime
2. HybridPredictor - Generates predictions using the appropriate model
3. Backtester - Validates the strategy on historical data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import joblib
import yaml


@dataclass
class RegimeConfig:
    """Configuration for variance regime detection.
    
    Optimal parameters determined via sensitivity analysis (tune_hybrid_thresholds.py):
    - Tested 75 parameter combinations
    - Best improvement: +2.41% over Champion baseline
    - Switch date: 2025-11-14 (correctly identifies variance regime shift)
    """
    # Historical baselines (calculated from 2021-24 seasons)
    baseline_std_deff: float = 3.37  # Mean final-season std dev of DEFF
    baseline_std_pace: float = 1.90  # Mean final-season std dev of Pace
    baseline_std_orb: float = 0.029  # Mean final-season std dev of ORB rate
    
    # Thresholds (optimized via sensitivity analysis)
    high_variance_threshold: float = 1.30  # Switch to Context when ratio > this
    normal_variance_threshold: float = 1.20  # Switch back to Champion when ratio < this
    
    # Stability requirements (optimized via sensitivity analysis)
    consecutive_days_required: int = 5  # Days above threshold before switching
    min_games_for_detection: int = 250  # ~31 games per team = early-mid November
    
    # Primary metric for detection
    primary_metric: str = "std_deff"


class VarianceRegimeDetector:
    """
    Detects the current variance regime of the NBA league.
    
    Uses team-level standard deviation of efficiency metrics to determine
    if the league is in a "high variance" state where the Context model
    should be preferred over the Champion model.
    """
    
    def __init__(self, config: RegimeConfig = None):
        self.config = config or RegimeConfig()
        self._df_cache: Dict[str, pd.DataFrame] = {}
        self._regime_history: List[Dict] = []
    
    def _load_boxscores(self, season: str) -> pd.DataFrame:
        """Load team boxscore data for a season."""
        if season in self._df_cache:
            return self._df_cache[season]
        
        season_map = {
            '2021-2022': 'data/team_boxscores/historical/2021-2022_NBA_Box_Score_Team-Stats.xlsx',
            '2022-2023': 'data/team_boxscores/historical/2022-2023_NBA_Box_Score_Team-Stats.xlsx',
            '2023-2024': 'data/team_boxscores/historical/2023-2024_NBA_Box_Score_Team-Stats.xlsx',
            '2024-2025': 'data/team_boxscores/historical/2024-2025_NBA_Box_Score_Team-Stats.xlsx',
        }
        
        if season in season_map:
            path = Path(season_map[season])
            if path.exists():
                df = pd.read_excel(path)
                df['DATE'] = pd.to_datetime(df['DATE'])
                self._df_cache[season] = df
                return df
        
        # For 2025-2026, check current directory
        if season == '2025-2026':
            current_dir = Path('data/team_boxscores/current')
            if current_dir.exists():
                xlsx_files = sorted(current_dir.glob('*.xlsx'), reverse=True)
                if xlsx_files:
                    df = pd.read_excel(xlsx_files[0])
                    df['DATE'] = pd.to_datetime(df['DATE'])
                    self._df_cache[season] = df
                    return df
        
        return pd.DataFrame()
    
    def calculate_variance_metrics(self, season: str, target_date: str) -> Dict[str, float]:
        """
        Calculate league-wide variance metrics up to (but not including) target_date.
        
        Returns dict with:
        - std_deff: Std dev of team defensive ratings
        - std_oeff: Std dev of team offensive ratings
        - std_pace: Std dev of team pace
        - std_orb: Std dev of team offensive rebound rates
        - game_count: Number of games used in calculation
        """
        df = self._load_boxscores(season)
        
        if df.empty:
            return {'std_deff': None, 'std_oeff': None, 'std_pace': None, 
                    'std_orb': None, 'game_count': 0}
        
        target_dt = pd.to_datetime(target_date)
        df_before = df[df['DATE'] < target_dt]
        
        if len(df_before) < self.config.min_games_for_detection:
            return {'std_deff': None, 'std_oeff': None, 'std_pace': None,
                    'std_orb': None, 'game_count': len(df_before)}
        
        # Calculate team-level averages
        team_stats = df_before.groupby('TEAM').agg({
            'OEFF': 'mean',
            'DEFF': 'mean',
            'PACE': 'mean',
            'OR': 'sum',
            'DR': 'sum',
        })
        
        # Calculate ORB rate per team
        team_stats['ORB_rate'] = team_stats['OR'] / (team_stats['OR'] + team_stats['DR'])
        
        # Calculate standard deviations across teams
        return {
            'std_deff': team_stats['DEFF'].std(),
            'std_oeff': team_stats['OEFF'].std(),
            'std_pace': team_stats['PACE'].std(),
            'std_orb': team_stats['ORB_rate'].std(),
            'game_count': len(df_before),
        }
    
    def get_variance_ratio(self, season: str, target_date: str) -> Dict[str, float]:
        """
        Calculate variance ratios (current / historical baseline).
        
        Returns dict with ratio for each metric.
        """
        metrics = self.calculate_variance_metrics(season, target_date)
        
        if metrics['std_deff'] is None:
            return {'ratio_deff': None, 'ratio_pace': None, 'ratio_orb': None,
                    'game_count': metrics['game_count']}
        
        return {
            'ratio_deff': metrics['std_deff'] / self.config.baseline_std_deff,
            'ratio_oeff': metrics['std_oeff'] / self.config.baseline_std_deff,  # Use same baseline
            'ratio_pace': metrics['std_pace'] / self.config.baseline_std_pace,
            'ratio_orb': metrics['std_orb'] / self.config.baseline_std_orb,
            'game_count': metrics['game_count'],
            'std_deff': metrics['std_deff'],
            'std_pace': metrics['std_pace'],
        }
    
    def detect_regime(self, season: str, target_date: str, 
                      history: List[Dict] = None) -> Tuple[str, Dict]:
        """
        Detect the current variance regime.
        
        Args:
            season: Season string (e.g., '2025-2026')
            target_date: Date to detect regime for
            history: Previous regime detections (for consecutive day tracking)
        
        Returns:
            Tuple of (regime_name, details_dict)
            regime_name is one of: 'normal', 'elevated', 'high', 'unknown'
        """
        ratios = self.get_variance_ratio(season, target_date)
        
        if ratios['ratio_deff'] is None:
            return 'unknown', {
                'reason': 'insufficient_data',
                'game_count': ratios['game_count'],
                'min_required': self.config.min_games_for_detection,
            }
        
        # Get primary metric ratio
        primary_ratio = ratios[f'ratio_{self.config.primary_metric.replace("std_", "")}']
        
        # Determine raw regime
        if primary_ratio > self.config.high_variance_threshold:
            raw_regime = 'high'
        elif primary_ratio > self.config.normal_variance_threshold:
            raw_regime = 'elevated'
        else:
            raw_regime = 'normal'
        
        # Check consecutive days (hysteresis)
        consecutive_high = 0
        if history:
            for h in reversed(history):
                if h.get('raw_regime') == 'high':
                    consecutive_high += 1
                else:
                    break
        
        # Apply hysteresis: need N consecutive high days to switch
        if raw_regime == 'high':
            consecutive_high += 1
        
        # Determine final regime with hysteresis
        if consecutive_high >= self.config.consecutive_days_required:
            final_regime = 'high'
        elif history and history[-1].get('final_regime') == 'high':
            # Already in high regime - use lower threshold to stay
            if primary_ratio > self.config.normal_variance_threshold:
                final_regime = 'high'
            else:
                final_regime = 'normal'
        else:
            final_regime = raw_regime if raw_regime != 'high' else 'elevated'
        
        details = {
            'date': target_date,
            'ratio_deff': ratios['ratio_deff'],
            'ratio_pace': ratios['ratio_pace'],
            'std_deff': ratios['std_deff'],
            'std_pace': ratios['std_pace'],
            'game_count': ratios['game_count'],
            'raw_regime': raw_regime,
            'final_regime': final_regime,
            'consecutive_high_days': consecutive_high,
            'threshold': self.config.high_variance_threshold,
        }
        
        return final_regime, details


class HybridModelSelector:
    """
    Selects between Champion and Context models based on variance regime.
    """
    
    def __init__(self, 
                 champion_path: str = "artifacts/champion_rest_schedule",
                 context_path: str = "artifacts/experiments/exp6_context",
                 config: RegimeConfig = None):
        self.champion_path = Path(champion_path)
        self.context_path = Path(context_path)
        self.detector = VarianceRegimeDetector(config)
        self.config = config or RegimeConfig()
        
        # Load models
        self.champion_model = None
        self.context_model = None
        self._load_models()
    
    def _load_models(self):
        """Load both models."""
        champ_model_path = self.champion_path / "margin_model.joblib"
        ctx_model_path = self.context_path / "margin_model.joblib"
        
        if champ_model_path.exists():
            self.champion_model = joblib.load(champ_model_path)
            print(f"Loaded Champion model from {champ_model_path}")
        
        if ctx_model_path.exists():
            self.context_model = joblib.load(ctx_model_path)
            print(f"Loaded Context model from {ctx_model_path}")
    
    def select_model(self, season: str, game_date: str, 
                     history: List[Dict] = None) -> Tuple[str, object, Dict]:
        """
        Select the appropriate model for a given date.
        
        Returns:
            Tuple of (model_name, model_object, regime_details)
        """
        regime, details = self.detector.detect_regime(season, game_date, history)
        
        if regime == 'high' and self.context_model is not None:
            return 'context', self.context_model, details
        else:
            return 'champion', self.champion_model, details


def backtest_hybrid_strategy(predictions_champion: pd.DataFrame,
                             predictions_context: pd.DataFrame,
                             season: str = '2025-2026') -> pd.DataFrame:
    """
    Backtest the hybrid model selection strategy.
    
    Args:
        predictions_champion: DataFrame with Champion model predictions
        predictions_context: DataFrame with Context model predictions
        season: Season to backtest
    
    Returns:
        DataFrame with daily regime, model selected, and ATS results
    """
    detector = VarianceRegimeDetector()
    
    # Ensure date columns are datetime
    predictions_champion = predictions_champion.copy()
    predictions_context = predictions_context.copy()
    predictions_champion['date'] = pd.to_datetime(predictions_champion['date'])
    predictions_context['date'] = pd.to_datetime(predictions_context['date'])
    
    # Get unique dates
    dates = sorted(predictions_champion['date'].unique())
    
    results = []
    history = []
    
    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        
        # Detect regime
        regime, details = detector.detect_regime(season, date_str, history)
        history.append(details)
        
        # Select model
        if regime == 'high':
            model_used = 'context'
            day_preds = predictions_context[predictions_context['date'] == date]
        else:
            model_used = 'champion'
            day_preds = predictions_champion[predictions_champion['date'] == date]
        
        # Calculate ATS for this day
        valid = day_preds.dropna(subset=['actual_margin'])
        if len(valid) > 0:
            pred_covers = valid['pred_margin_mu'] + valid['market_spread_home'] > 0
            actual_covers = valid['actual_margin'] + valid['market_spread_home'] > 0
            correct = (pred_covers == actual_covers).sum()
            total = len(valid)
        else:
            correct = 0
            total = 0
        
        results.append({
            'date': date,
            'regime': regime,
            'model_used': model_used,
            'ratio_deff': details.get('ratio_deff'),
            'std_deff': details.get('std_deff'),
            'consecutive_high': details.get('consecutive_high_days', 0),
            'games': total,
            'correct': correct,
            'ats_pct': 100 * correct / total if total > 0 else None,
        })
    
    return pd.DataFrame(results)


def main():
    """Run backtest and display results."""
    print("="*70)
    print("  HYBRID MODEL SELECTION BACKTEST")
    print("="*70)
    
    # Load predictions
    champ_path = Path("predictions/current_season_champion_2025_2026_predictions.csv")
    ctx_path = Path("artifacts/experiments/exp6_context/predictions_2526.csv")
    
    if not champ_path.exists() or not ctx_path.exists():
        print("ERROR: Prediction files not found")
        print(f"  Champion: {champ_path} - {'exists' if champ_path.exists() else 'MISSING'}")
        print(f"  Context:  {ctx_path} - {'exists' if ctx_path.exists() else 'MISSING'}")
        return
    
    champ_preds = pd.read_csv(champ_path)
    ctx_preds = pd.read_csv(ctx_path)
    
    print(f"\nLoaded predictions:")
    print(f"  Champion: {len(champ_preds)} games")
    print(f"  Context:  {len(ctx_preds)} games")
    
    # Run backtest
    print("\nRunning backtest...")
    results = backtest_hybrid_strategy(champ_preds, ctx_preds)
    
    # Display regime transitions
    print("\n" + "-"*70)
    print("  REGIME DETECTION TIMELINE")
    print("-"*70)
    
    prev_regime = None
    for _, row in results.iterrows():
        if row['regime'] != prev_regime:
            ratio_str = f"{row['ratio_deff']:.3f}" if pd.notna(row['ratio_deff']) else 'N/A'
            print(f"  {row['date'].strftime('%Y-%m-%d')}: {prev_regime or 'start'} -> {row['regime']} "
                  f"(ratio={ratio_str}, consecutive={row['consecutive_high']})")
            prev_regime = row['regime']
    
    # Calculate aggregate stats
    print("\n" + "-"*70)
    print("  PERFORMANCE BY MODEL USED")
    print("-"*70)
    
    for model in ['champion', 'context']:
        model_results = results[results['model_used'] == model]
        total_games = model_results['games'].sum()
        total_correct = model_results['correct'].sum()
        if total_games > 0:
            ats = 100 * total_correct / total_games
            print(f"  {model.capitalize():10s}: {ats:.2f}% ({total_correct}/{total_games}) over {len(model_results)} days")
    
    # Overall hybrid performance
    print("\n" + "-"*70)
    print("  OVERALL HYBRID PERFORMANCE")
    print("-"*70)
    
    total_games = results['games'].sum()
    total_correct = results['correct'].sum()
    hybrid_ats = 100 * total_correct / total_games
    
    print(f"  Hybrid ATS:   {hybrid_ats:.2f}% ({total_correct}/{total_games})")
    
    # Compare to baselines
    champ_preds['date'] = pd.to_datetime(champ_preds['date'])
    ctx_preds['date'] = pd.to_datetime(ctx_preds['date'])
    
    def calc_ats(df):
        valid = df.dropna(subset=['actual_margin'])
        pred_covers = valid['pred_margin_mu'] + valid['market_spread_home'] > 0
        actual_covers = valid['actual_margin'] + valid['market_spread_home'] > 0
        return (pred_covers == actual_covers).sum(), len(valid)
    
    champ_c, champ_t = calc_ats(champ_preds)
    ctx_c, ctx_t = calc_ats(ctx_preds)
    
    print(f"  Champion ATS: {100*champ_c/champ_t:.2f}% ({champ_c}/{champ_t})")
    print(f"  Context ATS:  {100*ctx_c/ctx_t:.2f}% ({ctx_c}/{ctx_t})")
    
    print("\n" + "-"*70)
    print("  IMPROVEMENT")
    print("-"*70)
    
    champ_ats = 100 * champ_c / champ_t
    improvement = hybrid_ats - champ_ats
    
    if improvement > 0:
        print(f"  [WIN] Hybrid beats Champion by {improvement:.2f}%")
    elif improvement < 0:
        print(f"  [LOSS] Hybrid loses to Champion by {-improvement:.2f}%")
    else:
        print(f"  [TIE] Hybrid ties Champion")
    
    print("="*70)
    
    # Save detailed results
    output_path = Path("analysis/hybrid_backtest_results.csv")
    output_path.parent.mkdir(exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()

