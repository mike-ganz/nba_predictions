#!/usr/bin/env python
"""
NBA Metrics Monitoring Script

This script monitors key NBA metrics across the current season and compares them
to historical baselines to detect regime changes and trends.

Metrics Tracked (aligned with production model features):
- OEFF (Offensive Efficiency): Points per 100 possessions [MODEL FEATURE]
- DEFF (Defensive Efficiency): Points allowed per 100 possessions [MODEL FEATURE]
- ORr (Offensive Rebound Rate): ORB / (ORB + DRB) [MODEL FEATURE]
- DRr (Defensive Rebound Rate): DRB / (ORB + DRB) [MODEL FEATURE]
- ASTr (Assist Rate): AST / FGM [MODEL FEATURE]
- TOr (Turnover Rate): TOV / (FGA + 0.44 * FTA + TOV) [MODEL FEATURE]
- 3PAr (3-Point Attempt Rate): 3PA / FGA [MODEL FEATURE]
- Pace: Possessions per game [MODEL FEATURE]
- FTR (Free Throw Rate): FTA / FGA [EXCLUDED from champion model - regime-specific]
- eFG% (Effective Field Goal %): (FGM + 0.5 * 3PM) / FGA [Context metric]
- TS% (True Shooting %): PTS / (2 * (FGA + 0.44 * FTA)) [Context metric]
- 3P%: 3PM / 3PA [Context metric]
- PPG: Points per game [Context metric]

Note: All team features in the production model are league-relative normalized.
This script tracks raw league averages to detect baseline shifts that would
affect the normalization process.

Usage:
    python monitor_nba_metrics.py
    
    # Show detailed stats
    python monitor_nba_metrics.py --verbose
    
    # Custom z-score threshold for alerts
    python monitor_nba_metrics.py --alert-threshold 1.5
"""

import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class MetricCalculator:
    """Calculate various NBA metrics from box score data."""
    
    @staticmethod
    def calculate_ftr(df: pd.DataFrame) -> Dict[str, float]:
        """Free Throw Rate: FTA / FGA"""
        fta = df['FTA'].sum()
        fga = df['FGA'].sum()
        return {
            'value': (fta / fga * 100) if fga > 0 else 0,
            'components': {'FTA': fta, 'FGA': fga}
        }
    
    @staticmethod
    def calculate_efg(df: pd.DataFrame) -> Dict[str, float]:
        """Effective Field Goal %: (FGM + 0.5 * 3PM) / FGA"""
        fgm = df['FGM'].sum()
        tpm = df['3PM'].sum()
        fga = df['FGA'].sum()
        return {
            'value': ((fgm + 0.5 * tpm) / fga * 100) if fga > 0 else 0,
            'components': {'FGM': fgm, '3PM': tpm, 'FGA': fga}
        }
    
    @staticmethod
    def calculate_ts(df: pd.DataFrame) -> Dict[str, float]:
        """True Shooting %: PTS / (2 * (FGA + 0.44 * FTA))"""
        pts = df['PTS'].sum()
        fga = df['FGA'].sum()
        fta = df['FTA'].sum()
        tsa = 2 * (fga + 0.44 * fta)
        return {
            'value': (pts / tsa * 100) if tsa > 0 else 0,
            'components': {'PTS': pts, 'FGA': fga, 'FTA': fta}
        }
    
    @staticmethod
    def calculate_3p_rate(df: pd.DataFrame) -> Dict[str, float]:
        """3-Point Rate: 3PA / FGA"""
        tpa = df['3PA'].sum()
        fga = df['FGA'].sum()
        return {
            'value': (tpa / fga * 100) if fga > 0 else 0,
            'components': {'3PA': tpa, 'FGA': fga}
        }
    
    @staticmethod
    def calculate_3p_pct(df: pd.DataFrame) -> Dict[str, float]:
        """3-Point %: 3PM / 3PA"""
        tpm = df['3PM'].sum()
        tpa = df['3PA'].sum()
        return {
            'value': (tpm / tpa * 100) if tpa > 0 else 0,
            'components': {'3PM': tpm, '3PA': tpa}
        }
    
    @staticmethod
    def calculate_pace(df: pd.DataFrame, n_games: int) -> Dict[str, float]:
        """Pace: Estimated possessions per game"""
        # Pace estimate: FGA + 0.44 * FTA - ORB + TOV
        fga = df['FGA'].sum()
        fta = df['FTA'].sum()
        orb = df.get('ORB', pd.Series([0])).sum() if 'ORB' in df.columns else 0
        tov = df.get('TOV', pd.Series([0])).sum() if 'TOV' in df.columns else 0
        
        total_poss = fga + 0.44 * fta - orb + tov
        pace = (total_poss / n_games) if n_games > 0 else 0
        
        return {
            'value': pace,
            'components': {'FGA': fga, 'FTA': fta, 'ORB': orb, 'TOV': tov, 'games': n_games}
        }
    
    @staticmethod
    def calculate_ppg(df: pd.DataFrame, n_games: int) -> Dict[str, float]:
        """Points per game"""
        pts = df['PTS'].sum()
        return {
            'value': (pts / n_games) if n_games > 0 else 0,
            'components': {'PTS': pts, 'games': n_games}
        }
    
    @staticmethod
    def calculate_tov_rate(df: pd.DataFrame) -> Dict[str, float]:
        """Turnover Rate: TOV / (FGA + 0.44 * FTA + TOV)"""
        tov = df.get('TOV', pd.Series([0])).sum() if 'TOV' in df.columns else 0
        fga = df['FGA'].sum()
        fta = df['FTA'].sum()
        
        possessions = fga + 0.44 * fta + tov
        return {
            'value': (tov / possessions * 100) if possessions > 0 else 0,
            'components': {'TOV': tov, 'FGA': fga, 'FTA': fta}
        }
    
    @staticmethod
    def calculate_orb_rate(df: pd.DataFrame) -> Dict[str, float]:
        """Offensive Rebound Rate: ORB / (ORB + DRB)
        
        Note: This matches the formula in generate_team_stats.py (ORr).
        Used in the model's orb_edge feature (home vs away ORr differential).
        """
        orb = df.get('ORB', pd.Series([0])).sum() if 'ORB' in df.columns else 0
        drb = df.get('DRB', pd.Series([0])).sum() if 'DRB' in df.columns else 0
        
        total_reb = orb + drb
        return {
            'value': (orb / total_reb * 100) if total_reb > 0 else 0,
            'components': {'ORB': orb, 'DRB': drb, 'Total': total_reb}
        }
    
    @staticmethod
    def calculate_drb_rate(df: pd.DataFrame) -> Dict[str, float]:
        """Defensive Rebound Rate: DRB / (ORB + DRB)
        
        Note: This matches the formula in generate_team_stats.py (DRr).
        Used in the model's orb_edge feature (away vs home DRr differential).
        """
        orb = df.get('ORB', pd.Series([0])).sum() if 'ORB' in df.columns else 0
        drb = df.get('DRB', pd.Series([0])).sum() if 'DRB' in df.columns else 0
        
        total_reb = orb + drb
        return {
            'value': (drb / total_reb * 100) if total_reb > 0 else 0,
            'components': {'ORB': orb, 'DRB': drb, 'Total': total_reb}
        }
    
    @staticmethod
    def calculate_ast_rate(df: pd.DataFrame) -> Dict[str, float]:
        """Assist Rate: AST / FGM
        
        Note: This matches the formula in generate_team_stats.py (ASTr).
        Used as a model feature (home_astr, away_astr).
        """
        ast = df.get('AST', pd.Series([0])).sum() if 'AST' in df.columns else 0
        fgm = df.get('FGM', pd.Series([0])).sum() if 'FGM' in df.columns else 0
        
        return {
            'value': (ast / fgm * 100) if fgm > 0 else 0,
            'components': {'AST': ast, 'FGM': fgm}
        }
    
    @staticmethod
    def calculate_oeff(df: pd.DataFrame, n_games: int) -> Dict[str, float]:
        """Offensive Efficiency: Points per 100 possessions (league average)
        
        Note: Uses pre-calculated OEFF from data files if available.
        Falls back to manual calculation if needed.
        This is the core metric for the model's 'edge' feature (OEFF vs DEFF matchup).
        """
        if 'OEFF' in df.columns:
            # Use pre-calculated value from data files
            oeff = df['OEFF'].mean()
            return {
                'value': oeff,
                'components': {'source': 'pre-calculated', 'n_games': n_games}
            }
        else:
            # Fallback: calculate from components
            pts = df['PTS'].sum()
            fga = df['FGA'].sum()
            fta = df['FTA'].sum()
            orb = df.get('ORB', pd.Series([0])).sum() if 'ORB' in df.columns else 0
            tov = df.get('TOV', pd.Series([0])).sum() if 'TOV' in df.columns else 0
            
            possessions = fga + 0.44 * fta - orb + tov
            oeff = (pts / possessions * 100) if possessions > 0 else 0
            
            return {
                'value': oeff,
                'components': {'PTS': pts, 'Possessions': possessions, 'n_games': n_games}
            }
    
    @staticmethod
    def calculate_deff(df: pd.DataFrame, n_games: int) -> Dict[str, float]:
        """Defensive Efficiency: Points allowed per 100 possessions (league average)
        
        Note: Uses pre-calculated DEFF from data files if available.
        This is the core metric for the model's 'edge' feature (OEFF vs DEFF matchup).
        
        The league average DEFF should equal league average OEFF (zero-sum game).
        """
        if 'DEFF' in df.columns:
            # Use pre-calculated value from data files
            deff = df['DEFF'].mean()
            return {
                'value': deff,
                'components': {'source': 'pre-calculated', 'n_games': n_games}
            }
        else:
            # Fallback: DEFF should equal OEFF at league level (zero-sum)
            if 'OEFF' in df.columns:
                oeff = df['OEFF'].mean()
                return {
                    'value': oeff,
                    'components': {'source': 'derived from OEFF', 'n_games': n_games}
                }
            else:
                # Can't calculate without opponent data
                return {
                    'value': None,
                    'components': {'source': 'unavailable', 'n_games': n_games}
                }
    


METRICS_CONFIG = {
    # ========================================================================
    # CORE MODEL FEATURES (used in Champion model)
    # ========================================================================
    
    'OEFF': {
        'name': 'Offensive Efficiency',
        'calculator': MetricCalculator.calculate_oeff,
        'unit': 'pts/100',
        'description': 'Points per 100 possessions [MODEL: edge feature]',
        'needs_games': True,
        'model_feature': True,
        'priority': 'CRITICAL'
    },
    'DEFF': {
        'name': 'Defensive Efficiency',
        'calculator': MetricCalculator.calculate_deff,
        'unit': 'pts/100',
        'description': 'Points allowed per 100 possessions [MODEL: edge feature]',
        'needs_games': True,
        'model_feature': True,
        'priority': 'CRITICAL'
    },
    'ORr': {
        'name': 'Offensive Rebound Rate',
        'calculator': MetricCalculator.calculate_orb_rate,
        'unit': '%',
        'description': 'ORB / (ORB + DRB) [MODEL: orb_edge feature]',
        'needs_games': False,
        'model_feature': True,
        'priority': 'HIGH'
    },
    'DRr': {
        'name': 'Defensive Rebound Rate',
        'calculator': MetricCalculator.calculate_drb_rate,
        'unit': '%',
        'description': 'DRB / (ORB + DRB) [MODEL: orb_edge feature]',
        'needs_games': False,
        'model_feature': True,
        'priority': 'HIGH'
    },
    'TOr': {
        'name': 'Turnover Rate',
        'calculator': MetricCalculator.calculate_tov_rate,
        'unit': '%',
        'description': 'TOV / Possessions [MODEL: tov_edge, home_tor, away_tor]',
        'needs_games': False,
        'model_feature': True,
        'priority': 'HIGH'
    },
    '3PAr': {
        'name': '3-Point Attempt Rate',
        'calculator': MetricCalculator.calculate_3p_rate,
        'unit': '%',
        'description': '3PA / FGA [MODEL: home_tpar, away_tpar]',
        'needs_games': False,
        'model_feature': True,
        'priority': 'HIGH'
    },
    'ASTr': {
        'name': 'Assist Rate',
        'calculator': MetricCalculator.calculate_ast_rate,
        'unit': '%',
        'description': 'AST / FGM [MODEL: home_astr, away_astr]',
        'needs_games': False,
        'model_feature': True,
        'priority': 'HIGH'
    },
    'Pace': {
        'name': 'Pace',
        'calculator': MetricCalculator.calculate_pace,
        'unit': 'poss/gm',
        'description': 'Possessions per game [MODEL: pace_mean, pace_diff]',
        'needs_games': True,
        'model_feature': True,
        'priority': 'HIGH'
    },
    
    # ========================================================================
    # EXCLUDED FROM CHAMPION MODEL (but monitored for context)
    # ========================================================================
    
    'FTR': {
        'name': 'Free Throw Rate',
        'calculator': MetricCalculator.calculate_ftr,
        'unit': '%',
        'description': 'FTA / FGA [EXCLUDED: regime-specific]',
        'needs_games': False,
        'model_feature': False,
        'priority': 'MEDIUM',
        'note': 'Excluded from Champion model due to regime-specific behavior'
    },
    
    # ========================================================================
    # CONTEXT METRICS (not direct model features)
    # ========================================================================
    
    'eFG': {
        'name': 'Effective Field Goal %',
        'calculator': MetricCalculator.calculate_efg,
        'unit': '%',
        'description': '(FGM + 0.5 * 3PM) / FGA [Context: shooting efficiency]',
        'needs_games': False,
        'model_feature': False,
        'priority': 'LOW'
    },
    'TS': {
        'name': 'True Shooting %',
        'calculator': MetricCalculator.calculate_ts,
        'unit': '%',
        'description': 'PTS / (2 * (FGA + 0.44 * FTA)) [Context: overall efficiency]',
        'needs_games': False,
        'model_feature': False,
        'priority': 'LOW'
    },
    '3P_Pct': {
        'name': '3-Point Percentage',
        'calculator': MetricCalculator.calculate_3p_pct,
        'unit': '%',
        'description': '3PM / 3PA [Context: 3P accuracy, not rate]',
        'needs_games': False,
        'model_feature': False,
        'priority': 'LOW'
    },
    'PPG': {
        'name': 'Points Per Game',
        'calculator': MetricCalculator.calculate_ppg,
        'unit': 'pts',
        'description': 'Total points per game [Context: scoring level]',
        'needs_games': True,
        'model_feature': False,
        'priority': 'LOW'
    },
}


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names across different file formats."""
    # Create a mapping of common variations
    column_map = {}
    
    for col in df.columns:
        col_upper = str(col).upper().strip()
        
        # Remove any leading/trailing spaces
        col_clean = col_upper.replace(' ', '')
        
        # Map common variations
        if col_clean in ['FTA', 'FREETHROWATTEMPTS']:
            column_map[col] = 'FTA'
        elif col_clean in ['FTM', 'FREETHROWNMADE', 'FT']:
            # FT in boxscore files means Free Throws Made
            column_map[col] = 'FTM'
        elif col_clean in ['FGA', 'FIELDGOALATTEMPTS']:
            column_map[col] = 'FGA'
        elif col_clean in ['FGM', 'FIELDGOALSMADE']:
            column_map[col] = 'FGM'
        elif col_clean in ['FG']:
            # FG in boxscore files means Field Goals Made
            column_map[col] = 'FGM'
        elif col_clean in ['3PA', 'THREEPOINTATTEMPTS', '3PTA']:
            column_map[col] = '3PA'
        elif col_clean in ['3PM', 'THREEPOINTMADE', '3PTM', '3P']:
            # 3P in boxscore files means 3-pointers Made
            column_map[col] = '3PM'
        elif col_clean in ['PTS', 'POINTS']:
            column_map[col] = 'PTS'
        elif col_clean in ['TOV', 'TURNOVERS', 'TO']:
            column_map[col] = 'TOV'
        elif col_clean in ['ORB', 'OFFENSIVEREBOUNDS', 'OREB', 'OR']:
            column_map[col] = 'ORB'
        elif col_clean in ['DRB', 'DEFENSIVEREBOUNDS', 'DREB', 'DR']:
            column_map[col] = 'DRB'
        elif col_clean in ['AST', 'ASSISTS', 'A']:
            column_map[col] = 'AST'
        elif col_clean in ['OEFF', 'OFFENSIVEEFFICIENCY']:
            column_map[col] = 'OEFF'
        elif col_clean in ['DEFF', 'DEFENSIVEEFFICIENCY']:
            column_map[col] = 'DEFF'
        elif col_clean in ['PACE']:
            column_map[col] = 'PACE'
    
    if column_map:
        df = df.rename(columns=column_map)
    
    return df


def calculate_all_metrics(df: pd.DataFrame, n_games: int) -> Dict[str, Any]:
    """Calculate all metrics for a given dataframe."""
    df = standardize_column_names(df)
    
    metrics = {}
    
    for metric_id, config in METRICS_CONFIG.items():
        try:
            if config['needs_games']:
                result = config['calculator'](df, n_games)
            else:
                result = config['calculator'](df)
            
            metrics[metric_id] = result['value']
        except KeyError as e:
            # Missing required column
            metrics[metric_id] = None
        except Exception as e:
            print(f"  Warning: Error calculating {metric_id}: {e}")
            metrics[metric_id] = None
    
    return metrics


def find_most_recent_boxscore(current_dir: Path) -> Tuple[Path, datetime]:
    """Find the most recent boxscore file in the current directory."""
    xlsx_files = list(current_dir.glob("*.xlsx"))
    
    if not xlsx_files:
        raise FileNotFoundError(f"No .xlsx files found in {current_dir}")
    
    files_with_dates = []
    for file in xlsx_files:
        try:
            date_str = file.stem.split('-nba-season')[0]
            date_obj = datetime.strptime(date_str, "%m-%d-%Y")
            files_with_dates.append((file, date_obj))
        except ValueError:
            continue
    
    if not files_with_dates:
        raise ValueError("No files with valid date format found")
    
    files_with_dates.sort(key=lambda x: x[1], reverse=True)
    return files_with_dates[0]


def load_all_current_files(current_dir: Path) -> pd.DataFrame:
    """Load all current season files and calculate metrics progression.
    
    Uses the most recent file to extract full season progression, since current
    season files are cumulative (contain all games from season start to file date).
    """
    xlsx_files = list(current_dir.glob("*.xlsx"))
    
    if not xlsx_files:
        raise FileNotFoundError(f"No .xlsx files found in {current_dir}")
    
    files_with_dates = []
    for file in xlsx_files:
        try:
            date_str = file.stem.split('-nba-season')[0]
            date_obj = datetime.strptime(date_str, "%m-%d-%Y")
            files_with_dates.append((file, date_obj))
        except ValueError:
            continue
    
    files_with_dates.sort(key=lambda x: x[1], reverse=True)
    
    # Use the most recent file since it contains all games from season start
    most_recent_file = files_with_dates[0][0]
    
    try:
        df = pd.read_excel(most_recent_file)
        df = standardize_column_names(df)
        
        # Find date column
        date_col = None
        for col in df.columns:
            col_lower = str(col).lower()
            if 'date' in col_lower and 'dataset' not in col_lower:
                date_col = col
                break
        
        if date_col is None:
            raise ValueError("No date column found in current season file")
        
        # Parse dates
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        
        # Sort by date
        df = df.sort_values(date_col)
        
        # Get season start date
        season_start = df[date_col].min()
        
        # Get unique dates
        unique_dates = df[date_col].unique()
        unique_dates = sorted(unique_dates)
        
        all_data = []
        
        # Calculate cumulative metrics for each date
        for date in unique_dates:
            # Get all games up to this date
            df_to_date = df[df[date_col] <= date]
            n_games = len(df_to_date) // 2
            
            if n_games == 0:
                continue
            
            # Calculate metrics
            metrics = calculate_all_metrics(df_to_date, n_games)
            metrics['date'] = date
            metrics['days_since_start'] = (date - season_start).days
            metrics['n_games'] = n_games
            
            all_data.append(metrics)
            
    except Exception as e:
        print(f"  Error processing most recent file: {e}")
        raise
    
    if not all_data:
        raise ValueError("No valid data found in current season files")
    
    return pd.DataFrame(all_data)


def load_historical_metrics(historical_dir: Path) -> pd.DataFrame:
    """Load historical metrics from all seasons (season-level summaries)."""
    xlsx_files = list(historical_dir.glob("*.xlsx"))
    
    all_data = []
    
    for file in xlsx_files:
        season = file.stem.split('_')[0]
        
        try:
            df = pd.read_excel(file)
            df = standardize_column_names(df)
            
            n_games = len(df) // 2
            
            metrics = calculate_all_metrics(df, n_games)
            metrics['season'] = season
            metrics['n_games'] = n_games
            
            all_data.append(metrics)
            
        except Exception as e:
            print(f"  Warning: Error processing {file.name}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No valid historical data found")
    
    return pd.DataFrame(all_data)


def load_historical_progression(historical_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load game-by-game progression for each historical season."""
    xlsx_files = list(historical_dir.glob("*.xlsx"))
    
    historical_progressions = {}
    
    for file in xlsx_files:
        season = file.stem.split('_')[0]
        
        try:
            df = pd.read_excel(file)
            df = standardize_column_names(df)
            
            # Find the date column
            date_col = None
            for col in df.columns:
                col_lower = str(col).lower()
                if 'date' in col_lower and 'dataset' not in col_lower:
                    date_col = col
                    break
            
            if date_col is None:
                print(f"  Warning: No date column found in {file.name}")
                continue
            
            # Parse dates
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])
            
            # Sort by date
            df = df.sort_values(date_col)
            
            # Get season start date
            season_start = df[date_col].min()
            
            # Get unique dates to calculate progression
            unique_dates = df[date_col].unique()
            unique_dates = sorted(unique_dates)
            
            progression = []
            
            # Calculate cumulative metrics for each date
            for date in unique_dates:
                # Get all games up to this date
                df_to_date = df[df[date_col] <= date]
                n_games = len(df_to_date) // 2
                
                if n_games == 0:
                    continue
                
                # Calculate metrics
                metrics = calculate_all_metrics(df_to_date, n_games)
                metrics['date'] = date
                metrics['days_since_start'] = (date - season_start).days
                metrics['n_games'] = n_games
                metrics['season'] = season
                
                progression.append(metrics)
            
            if progression:
                historical_progressions[season] = pd.DataFrame(progression)
                
        except Exception as e:
            print(f"  Warning: Error processing {file.name}: {e}")
            continue
    
    return historical_progressions


def plot_metrics_progression(progression_df: pd.DataFrame, 
                            historical_df: pd.DataFrame,
                            historical_progressions: Dict[str, pd.DataFrame],
                            output_dir: Path = None):
    """Create visual charts for all metrics showing current and historical season progressions."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        if output_dir:
            output_dir.mkdir(exist_ok=True)
        
        # Create a multi-panel figure
        n_metrics = len([m for m in METRICS_CONFIG.keys() 
                        if progression_df[m].notna().any()])
        n_cols = 2
        n_rows = (n_metrics + 1) // 2
        
        fig = plt.figure(figsize=(16, 4 * n_rows))
        gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)
        
        # Define distinct colors for historical seasons (easier to differentiate)
        historical_colors = [
            '#ef4444',  # red
            '#f59e0b',  # orange
            '#10b981',  # green
            '#8b5cf6',  # purple
            '#ec4899',  # pink
        ]
        
        plot_idx = 0
        
        for metric_id, config in METRICS_CONFIG.items():
            if not progression_df[metric_id].notna().any():
                continue
            
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            ax = fig.add_subplot(gs[row, col])
            
            # Get current season data with days_since_start
            valid_data = progression_df[['days_since_start', metric_id, 'n_games']].dropna()
            
            # Get max days for current season (to limit x-axis)
            current_season_max_days = valid_data['days_since_start'].max() if len(valid_data) > 0 else 0
            
            # Calculate historical baseline
            historical_values = historical_df[metric_id].dropna()
            if len(historical_values) > 0:
                baseline = historical_values.mean()
                baseline_std = historical_values.std()
            else:
                baseline = None
                baseline_std = None
            
            # Plot historical season progressions
            max_days = current_season_max_days
            if historical_progressions:
                sorted_seasons = sorted(historical_progressions.keys())
                
                for idx, season in enumerate(sorted_seasons):
                    hist_prog = historical_progressions[season]
                    
                    # Skip if metric not available
                    if metric_id not in hist_prog.columns or hist_prog[metric_id].isna().all():
                        continue
                    
                    if 'days_since_start' not in hist_prog.columns:
                        continue
                    
                    hist_data = hist_prog[['days_since_start', metric_id]].dropna()
                    
                    if len(hist_data) > 0:
                        color_idx = idx % len(historical_colors)
                        ax.plot(hist_data['days_since_start'], hist_data[metric_id], 
                               '-', linewidth=2, alpha=0.8,
                               color=historical_colors[color_idx],
                               label=f'{season}')
            
            # Plot current season progression (prominent)
            if len(valid_data) > 0:
                ax.plot(valid_data['days_since_start'], valid_data[metric_id], 
                       'o-', linewidth=3, markersize=6, 
                       color='#3b82f6', label='2025-26', zorder=10)
            
            # Formatting
            ax.set_xlabel('Days Since Season Start', fontsize=10)
            ax.set_ylabel(f"{config['name']} ({config['unit']})", fontsize=10)
            ax.set_title(config['name'], fontsize=12, fontweight='bold')
            
            # Legend with multiple columns to fit all seasons
            ax.legend(loc='best', fontsize=7, ncol=2, framealpha=0.9)
            ax.grid(True, alpha=0.3)
            
            # Set x-axis to match current season's progression
            if current_season_max_days > 0:
                ax.set_xlim(0, current_season_max_days)
            
            # Set y-axis to have reasonable limits
            all_values = []
            if len(valid_data) > 0:
                all_values.extend(valid_data[metric_id].tolist())
            for season_prog in historical_progressions.values():
                if metric_id in season_prog.columns:
                    all_values.extend(season_prog[metric_id].dropna().tolist())
            
            if all_values:
                y_min = min(all_values)
                y_max = max(all_values)
                y_range = y_max - y_min
                if y_range > 0:
                    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
            
            plot_idx += 1
        
        plt.suptitle('NBA Metrics: 2025-26 Season vs. Historical Season Progressions', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        if output_dir:
            output_path = output_dir / 'metrics_progression.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  Saved chart to {output_path}")
        else:
            plt.show()
        
        plt.close()
        
        return True
        
    except ImportError:
        print("  Warning: matplotlib not available, skipping chart generation")
        return False


def calculate_team_distribution_stats(boxscore_file: Path, metric_id: str, 
                                     league_avg: float) -> Dict[str, float]:
    """Calculate distribution of team-level values for a given metric.
    
    This shows how teams are distributed around the league average, which is
    critical for understanding feature variance that the model sees.
    """
    try:
        df = pd.read_excel(boxscore_file)
        df = standardize_column_names(df)
        
        # Get metric config
        config = METRICS_CONFIG.get(metric_id)
        if not config:
            return None
        
        # Calculate metric for each team
        team_values = []
        for team in df['TEAM'].unique():
            team_df = df[df['TEAM'] == team]
            
            if len(team_df) == 0:
                continue
            
            # Calculate metric
            n_games = len(team_df)
            if config['needs_games']:
                result = config['calculator'](team_df, n_games)
            else:
                result = config['calculator'](team_df)
            
            if result and result['value'] is not None:
                team_values.append(result['value'])
        
        if len(team_values) < 5:  # Need at least 5 teams
            return None
        
        # Normalize values relative to league average
        team_values = np.array(team_values)
        normalized_values = team_values - league_avg
        
        return {
            'mean': float(normalized_values.mean()),  # Should be ~0
            'std': float(normalized_values.std()),
            'min': float(normalized_values.min()),
            'max': float(normalized_values.max()),
            'p25': float(np.percentile(normalized_values, 25)),
            'p50': float(np.percentile(normalized_values, 50)),
            'p75': float(np.percentile(normalized_values, 75)),
            'range': float(normalized_values.max() - normalized_values.min()),
            'n_teams': len(team_values)
        }
        
    except Exception as e:
        print(f"  Warning: Error calculating team distribution for {metric_id}: {e}")
        return None


def analyze_distribution_drift(current_dist: Dict[str, float], 
                               historical_dists: List[Dict[str, float]]) -> Dict[str, Any]:
    """Compare current distribution to historical distributions."""
    
    if not historical_dists or not current_dist:
        return None
    
    # Calculate historical baseline
    hist_stds = [d['std'] for d in historical_dists if d is not None]
    hist_ranges = [d['range'] for d in historical_dists if d is not None]
    
    if not hist_stds or not hist_ranges:
        return None
    
    baseline_std = np.mean(hist_stds)
    baseline_range = np.mean(hist_ranges)
    
    # Calculate drift
    std_drift = ((current_dist['std'] - baseline_std) / baseline_std) * 100
    range_drift = ((current_dist['range'] - baseline_range) / baseline_range) * 100
    
    # Determine severity
    if abs(std_drift) > 30 or abs(range_drift) > 30:
        severity = '🚨 CRITICAL'
    elif abs(std_drift) > 20 or abs(range_drift) > 20:
        severity = '⚠️  ALERT'
    elif abs(std_drift) > 10 or abs(range_drift) > 10:
        severity = '⚡ WATCH'
    else:
        severity = '✓ NORMAL'
    
    return {
        'severity': severity,
        'current_std': current_dist['std'],
        'baseline_std': baseline_std,
        'std_drift_pct': std_drift,
        'current_range': current_dist['range'],
        'baseline_range': baseline_range,
        'range_drift_pct': range_drift,
    }


def analyze_intra_season_trends(progression_df: pd.DataFrame, 
                               metric_id: str,
                               early_games_threshold: int = 20,
                               recent_games_window: int = 10) -> Dict[str, Any]:
    """Detailed analysis of how a metric has changed within the current season."""
    values = progression_df[metric_id].dropna()
    games = progression_df['n_games'].dropna()
    dates = progression_df['date'].dropna()
    
    if len(values) == 0:
        return None
    
    # Align all series to same length
    min_len = min(len(values), len(games), len(dates))
    values = values.iloc[:min_len]
    games = games.iloc[:min_len]
    dates = dates.iloc[:min_len]
    
    analysis = {}
    
    # Early season baseline (first snapshot with >= threshold games)
    early_mask = games >= early_games_threshold
    if early_mask.any():
        first_qualifying_idx = early_mask.idxmax()
        analysis['early_value'] = values.loc[first_qualifying_idx]
        analysis['early_games'] = games.loc[first_qualifying_idx]
        analysis['early_date'] = dates.loc[first_qualifying_idx]
    else:
        # Use first available snapshot
        analysis['early_value'] = values.iloc[0]
        analysis['early_games'] = games.iloc[0]
        analysis['early_date'] = dates.iloc[0]
    
    # Current value
    analysis['current_value'] = values.iloc[-1]
    analysis['current_games'] = games.iloc[-1]
    analysis['current_date'] = dates.iloc[-1]
    
    # Season-long change
    analysis['season_change'] = analysis['current_value'] - analysis['early_value']
    analysis['season_change_pct'] = (analysis['season_change'] / analysis['early_value'] * 100) if analysis['early_value'] != 0 else 0
    
    # Recent trend (last N snapshots)
    recent_window = min(recent_games_window, len(values))
    recent_values = values.iloc[-recent_window:]
    
    if len(recent_values) >= 2:
        # Linear regression on recent values to determine trend
        x = np.arange(len(recent_values))
        y = recent_values.values
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            analysis['recent_slope'] = slope
            analysis['recent_trend'] = "Increasing" if slope > 0.01 else "Decreasing" if slope < -0.01 else "Stable"
        else:
            analysis['recent_slope'] = 0
            analysis['recent_trend'] = "Stable"
        
        # Recent volatility
        analysis['recent_std'] = recent_values.std()
        analysis['recent_min'] = recent_values.min()
        analysis['recent_max'] = recent_values.max()
    else:
        analysis['recent_slope'] = 0
        analysis['recent_trend'] = "Insufficient Data"
        analysis['recent_std'] = 0
        analysis['recent_min'] = analysis['current_value']
        analysis['recent_max'] = analysis['current_value']
    
    # Calculate moving average (if enough data points)
    if len(values) >= 3:
        analysis['moving_avg_3'] = values.iloc[-3:].mean()
    else:
        analysis['moving_avg_3'] = analysis['current_value']
    
    if len(values) >= 5:
        analysis['moving_avg_5'] = values.iloc[-5:].mean()
    else:
        analysis['moving_avg_5'] = analysis['current_value']
    
    # Detect regime change point within season
    # Look for significant deviation from early season value
    if len(values) >= 5:
        early_avg = values.iloc[:min(3, len(values))].mean()
        deviations = abs(values - early_avg)
        if deviations.max() > 0:
            significant_deviation_idx = deviations.idxmax()
            if deviations.loc[significant_deviation_idx] > early_avg * 0.05:  # 5% threshold
                analysis['inflection_point'] = dates.loc[significant_deviation_idx]
                analysis['inflection_games'] = games.loc[significant_deviation_idx]
                analysis['inflection_value'] = values.loc[significant_deviation_idx]
            else:
                analysis['inflection_point'] = None
        else:
            analysis['inflection_point'] = None
    else:
        analysis['inflection_point'] = None
    
    # Acceleration/deceleration (second derivative)
    if len(values) >= 3:
        # Compare first half vs second half slopes
        mid_point = len(values) // 2
        first_half = values.iloc[:mid_point]
        second_half = values.iloc[mid_point:]
        
        if len(first_half) >= 2:
            x1 = np.arange(len(first_half))
            slope1 = np.polyfit(x1, first_half.values, 1)[0] if len(x1) > 1 else 0
        else:
            slope1 = 0
        
        if len(second_half) >= 2:
            x2 = np.arange(len(second_half))
            slope2 = np.polyfit(x2, second_half.values, 1)[0] if len(x2) > 1 else 0
        else:
            slope2 = 0
        
        acceleration = slope2 - slope1
        if abs(acceleration) > 0.01:
            if acceleration > 0:
                analysis['acceleration'] = "Accelerating upward"
            else:
                analysis['acceleration'] = "Decelerating / reversing"
        else:
            analysis['acceleration'] = "Steady"
    else:
        analysis['acceleration'] = "Insufficient Data"
    
    return analysis


def analyze_metric_changes(progression_df: pd.DataFrame, 
                          historical_df: pd.DataFrame,
                          alert_threshold: float = 1.5) -> pd.DataFrame:
    """Analyze all metrics for significant changes (both vs. history and within season)."""
    results = []
    
    for metric_id, config in METRICS_CONFIG.items():
        # Get current value (most recent)
        current_values = progression_df[metric_id].dropna()
        if len(current_values) == 0:
            continue
        
        current_value = current_values.iloc[-1]
        
        # Get historical baseline
        historical_values = historical_df[metric_id].dropna()
        if len(historical_values) == 0:
            continue
        
        baseline = historical_values.mean()
        baseline_std = historical_values.std()
        
        # Calculate difference and z-score vs. historical baseline
        difference = current_value - baseline
        z_score = difference / baseline_std if baseline_std > 0 else 0
        
        # Intra-season trend analysis
        intra_season = analyze_intra_season_trends(progression_df, metric_id)
        
        # Determine overall trend status
        if intra_season:
            trend = intra_season['recent_trend']
            season_change = intra_season['season_change']
            acceleration = intra_season['acceleration']
        else:
            trend = "Insufficient Data"
            season_change = 0
            acceleration = "Unknown"
        
        # Alert status (based on historical comparison)
        if abs(z_score) >= 2.0:
            status = "🚨 CRITICAL"
        elif abs(z_score) >= alert_threshold:
            status = "⚠️  ALERT"
        elif abs(z_score) >= 1.0:
            status = "⚡ WATCH"
        else:
            status = "✓ NORMAL"
        
        results.append({
            'Metric': config['name'],
            'Current': current_value,
            'Baseline': baseline,
            'Std Dev': baseline_std,
            'Difference': difference,
            'Z-Score': z_score,
            'Trend': trend,
            'Season_Change': season_change,
            'Acceleration': acceleration,
            'Status': status,
            'Unit': config['unit'],
            'Model_Feature': config.get('model_feature', False),
            'Priority': config.get('priority', 'LOW'),
            'IntraSeason': intra_season  # Store full analysis for detailed view
        })
    
    df_results = pd.DataFrame(results)
    
    # Sort by: 1) Model features first, 2) Absolute Z-score descending
    df_results['abs_z'] = df_results['Z-Score'].abs()
    df_results = df_results.sort_values(
        ['Model_Feature', 'abs_z'], 
        ascending=[False, False]
    ).drop('abs_z', axis=1)
    
    return df_results


def print_header():
    """Print formatted header."""
    print()
    print("=" * 80)
    print("NBA METRICS MONITORING & REGIME CHANGE DETECTION")
    print("=" * 80)
    print()


def print_section(title: str):
    """Print section header."""
    print()
    print("-" * 80)
    print(title)
    print("-" * 80)
    print()


def print_metrics_table(df: pd.DataFrame, verbose: bool = False):
    """Print formatted metrics comparison table."""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " METRIC ANALYSIS SUMMARY".center(78) + "║")
    print("╠" + "═" * 78 + "╣")
    print("║  🎯 = Direct model feature (Champion v2)".ljust(79) + "║")
    print("║     = Context metric (not in model)".ljust(79) + "║")
    print("╠" + "═" * 78 + "╣")
    
    for _, row in df.iterrows():
        print("║" + " " * 78 + "║")
        
        # Add indicator for model features
        metric_name = row['Metric']
        if row.get('Model_Feature', False):
            metric_display = f"🎯 {metric_name}"
        else:
            metric_display = f"   {metric_name}"
        
        print("║  " + metric_display.ljust(78) + "║")
        print("║    " + f"Status: {row['Status']}".ljust(74) + "║")
        
        # Historical comparison
        print("║    " + f"Current: {row['Current']:.2f}{row['Unit']}  |  " 
              f"Baseline: {row['Baseline']:.2f}{row['Unit']}  |  "
              f"Diff: {row['Difference']:+.2f}{row['Unit']}".ljust(74) + "║")
        print("║    " + f"Z-Score: {row['Z-Score']:+.2f} σ".ljust(74) + "║")
        
        # Intra-season changes
        intra = row.get('IntraSeason')
        if intra and isinstance(intra, dict):
            print("║    " + "─" * 74 + "║")
            print("║    " + f"Season Progress: {intra['season_change']:+.2f}{row['Unit']} "
                  f"({intra['season_change_pct']:+.1f}%)".ljust(74) + "║")
            print("║    " + f"Recent Trend: {row['Trend']}  |  "
                  f"{row['Acceleration']}".ljust(74) + "║")
            
            if verbose:
                print("║    " + f"Early Season: {intra['early_value']:.2f}{row['Unit']} "
                      f"({intra['early_games']:.0f} games)".ljust(74) + "║")
                print("║    " + f"Moving Avg (3): {intra['moving_avg_3']:.2f}{row['Unit']}  |  "
                      f"Moving Avg (5): {intra['moving_avg_5']:.2f}{row['Unit']}".ljust(74) + "║")
                print("║    " + f"Recent Range: {intra['recent_min']:.2f} - {intra['recent_max']:.2f}{row['Unit']}".ljust(74) + "║")
                
                if intra['inflection_point']:
                    print("║    " + f"⚡ Inflection: {intra['inflection_point'].strftime('%m/%d')} "
                          f"({intra['inflection_games']:.0f} games)".ljust(74) + "║")
        else:
            print("║    " + f"Recent Trend: {row['Trend']}".ljust(74) + "║")
        
        if verbose and 'Std Dev' in row:
            print("║    " + f"Historical Std Dev: {row['Std Dev']:.2f}{row['Unit']}".ljust(74) + "║")
        
        print("║" + "─" * 78 + "║")
    
    print("╚" + "═" * 78 + "╝")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Monitor NBA metrics and detect regime changes",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--current-dir",
        type=str,
        default="data/team_boxscores/current",
        help="Directory containing current season boxscore files"
    )
    parser.add_argument(
        "--historical-dir",
        type=str,
        default="data/team_boxscores/historical",
        help="Directory containing historical boxscore files"
    )
    parser.add_argument(
        "--alert-threshold",
        type=float,
        default=1.5,
        help="Z-score threshold for alerts (default: 1.5)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed statistics"
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip chart generation"
    )
    
    args = parser.parse_args()
    
    current_dir = Path(args.current_dir)
    historical_dir = Path(args.historical_dir)
    
    print_header()
    
    # Step 1: Load current season progression
    print_section("STEP 1: Loading Current Season Progression")
    
    try:
        print("Loading all current season files...")
        progression_df = load_all_current_files(current_dir)
        
        print(f"Found {len(progression_df)} data snapshots")
        print(f"Date range: {progression_df['date'].min().strftime('%m/%d/%Y')} "
              f"to {progression_df['date'].max().strftime('%m/%d/%Y')}")
        print(f"Games analyzed: {progression_df['n_games'].iloc[-1]}")
        print()
        
        # Show available metrics
        available_metrics = [m for m in METRICS_CONFIG.keys() 
                           if progression_df[m].notna().any()]
        print(f"Available metrics: {', '.join(available_metrics)}")
        
    except Exception as e:
        print(f"Error loading current season data: {e}")
        return
    
    # Step 2: Load historical baseline
    print_section("STEP 2: Loading Historical Baseline")
    
    try:
        print("Loading historical season files...")
        historical_df = load_historical_metrics(historical_dir)
        
        print(f"Historical seasons loaded: {len(historical_df)}")
        print()
        
        if args.verbose:
            print("Historical Data by Season:")
            for _, row in historical_df.iterrows():
                print(f"  {row['season']}: {row['n_games']} games")
        
        # Load historical progressions for charts
        print("Loading historical season progressions...")
        historical_progressions = load_historical_progression(historical_dir)
        print(f"Historical progressions loaded: {len(historical_progressions)} seasons")
        print()
        
    except Exception as e:
        print(f"Error loading historical data: {e}")
        return
    
    # Step 3: Analyze metrics
    print_section("STEP 3: Metrics Analysis & Regime Change Detection")
    
    analysis_df = analyze_metric_changes(progression_df, historical_df, 
                                        args.alert_threshold)
    
    print_metrics_table(analysis_df, verbose=args.verbose)
    
    # Step 3.5: Feature Distribution Analysis
    print_section("STEP 3.5: Feature Distribution Analysis (Team-Level Variance)")
    
    print("Analyzing how feature distributions have changed...")
    print("(This shows if the model is seeing different input ranges)")
    print()
    
    # Get most recent current season file
    current_files = list(current_dir.glob("*.xlsx"))
    if current_files:
        current_files_dated = []
        for file in current_files:
            try:
                date_str = file.stem.split('-nba-season')[0]
                date_obj = datetime.strptime(date_str, "%m-%d-%Y")
                current_files_dated.append((file, date_obj))
            except:
                continue
        
        if current_files_dated:
            current_files_dated.sort(key=lambda x: x[1], reverse=True)
            most_recent_current = current_files_dated[0][0]
            
            # Get historical files
            historical_files = list(historical_dir.glob("*.xlsx"))
            
            # Analyze model features only
            model_features = [m for m, c in METRICS_CONFIG.items() if c.get('model_feature', False)]
            
            distribution_results = []
            
            for metric_id in model_features:
                config = METRICS_CONFIG[metric_id]
                
                # Get current season distribution
                current_league_avg = progression_df[metric_id].iloc[-1] if metric_id in progression_df.columns else None
                
                if current_league_avg is None or pd.isna(current_league_avg):
                    continue
                
                current_dist = calculate_team_distribution_stats(
                    most_recent_current, metric_id, current_league_avg
                )
                
                # Get historical distributions
                historical_dists = []
                for hist_file in historical_files[:5]:  # Last 5 seasons
                    hist_season = hist_file.stem.split('_')[0]
                    hist_data = historical_df[historical_df['season'] == hist_season]
                    
                    if len(hist_data) > 0 and metric_id in hist_data.columns:
                        hist_avg = hist_data[metric_id].iloc[0]
                        if not pd.isna(hist_avg):
                            hist_dist = calculate_team_distribution_stats(
                                hist_file, metric_id, hist_avg
                            )
                            if hist_dist:
                                historical_dists.append(hist_dist)
                
                # Analyze drift
                if current_dist and historical_dists:
                    drift = analyze_distribution_drift(current_dist, historical_dists)
                    
                    if drift:
                        distribution_results.append({
                            'metric': config['name'],
                            'metric_id': metric_id,
                            'unit': config['unit'],
                            'current_dist': current_dist,
                            'drift': drift
                        })
            
            # Display results
            if distribution_results:
                print("Feature Variance Analysis (Normalized Values Across Teams):")
                print()
                
                for result in distribution_results:
                    print(f"  🎯 {result['metric']}:")
                    print(f"     Current Std Dev:     ±{result['current_dist']['std']:.2f}{result['unit']}")
                    print(f"     Historical Std Dev:  ±{result['drift']['baseline_std']:.2f}{result['unit']}")
                    print(f"     Drift:               {result['drift']['std_drift_pct']:+.1f}%")
                    print(f"     Current Range:       {result['current_dist']['range']:.2f}{result['unit']}")
                    print(f"     Historical Range:    {result['drift']['baseline_range']:.2f}{result['unit']}")
                    print(f"     Status:              {result['drift']['severity']}")
                    
                    # Interpretation
                    if abs(result['drift']['std_drift_pct']) > 20:
                        if result['drift']['std_drift_pct'] < 0:
                            print(f"     ⚠️  Teams are MORE CLUSTERED (model sees narrower input range)")
                        else:
                            print(f"     ⚠️  Teams are MORE SPREAD OUT (model sees wider input range)")
                    
                    print()
                
                # Summary
                critical_variance_drift = sum(1 for r in distribution_results 
                                             if '🚨' in r['drift']['severity'])
                alert_variance_drift = sum(1 for r in distribution_results 
                                          if '⚠️' in r['drift']['severity'])
                
                print("Distribution Drift Summary:")
                print(f"  🚨 Critical variance changes: {critical_variance_drift}")
                print(f"  ⚠️  Alert-level variance changes: {alert_variance_drift}")
                
                if critical_variance_drift > 0 or alert_variance_drift > 0:
                    print()
                    print("  ⚠️  MODEL IMPACT: Feature distributions have changed significantly!")
                    print("     The model is seeing different input ranges than during training.")
                    print("     This can affect predictions even with correct normalization.")
                else:
                    print()
                    print("  ✓ Feature distributions remain stable")
                
                print()
    
    
    # Count alerts
    critical = len(analysis_df[analysis_df['Status'].str.contains('CRITICAL')])
    alerts = len(analysis_df[analysis_df['Status'].str.contains('ALERT')])
    watch = len(analysis_df[analysis_df['Status'].str.contains('WATCH')])
    
    print("Alert Summary:")
    print(f"  🚨 Critical Regime Changes: {critical}")
    print(f"  ⚠️  Significant Deviations: {alerts}")
    print(f"  ⚡ Moderate Deviations: {watch}")
    print()
    
    # Model feature coverage summary
    model_features = analysis_df[analysis_df['Model_Feature'] == True]
    model_critical = len(model_features[model_features['Status'].str.contains('CRITICAL')])
    model_alerts = len(model_features[model_features['Status'].str.contains('ALERT')])
    model_watch = len(model_features[model_features['Status'].str.contains('WATCH')])
    
    print("Model Feature Coverage:")
    print(f"  Total model features tracked: {len(model_features)}")
    print(f"  🚨 Critical changes in model features: {model_critical}")
    print(f"  ⚠️  Alerts in model features: {model_alerts}")
    print(f"  ⚡ Watch in model features: {model_watch}")
    if model_critical + model_alerts > 0:
        print(f"  ⚠️  ACTION REQUIRED: Model features showing regime changes!")
    else:
        print(f"  ✓ Model features within expected ranges")
    print()
    
    # Step 4: Generate visualizations
    if not args.no_charts:
        print_section("STEP 4: Generating Visualizations")
        
        print("Creating metrics progression charts...")
        chart_dir = Path("analysis")
        plot_metrics_progression(progression_df, historical_df, historical_progressions, chart_dir)
        print()
    
    # Step 5: Season progression milestones
    print_section("STEP 5: Season Progression Milestones")
    
    # Show metrics at different stages of the season
    total_games = progression_df['n_games'].iloc[-1]
    milestones = {
        'Early (≤25 games)': progression_df[progression_df['n_games'] <= 25].iloc[-1] if len(progression_df[progression_df['n_games'] <= 25]) > 0 else None,
        'Mid (≤82 games)': progression_df[progression_df['n_games'] <= 82].iloc[-1] if len(progression_df[progression_df['n_games'] <= 82]) > 0 else None,
        'Recent (≤164 games)': progression_df[progression_df['n_games'] <= 164].iloc[-1] if len(progression_df[progression_df['n_games'] <= 164]) > 0 else None,
        'Current': progression_df.iloc[-1]
    }
    
    # Filter out None milestones
    milestones = {k: v for k, v in milestones.items() if v is not None}
    
    if len(milestones) > 1:
        print(f"Season progression (Total: {total_games:.0f} games analyzed):")
        print()
        
        # Show key metrics across milestones (prioritize model features)
        key_metrics = ['OEFF', 'DEFF', '3PAr', 'TOr', 'Pace', 'ORr', 'DRr']
        available_metrics = [m for m in key_metrics if m in progression_df.columns 
                            and progression_df[m].notna().any()]
        
        if available_metrics:
            for metric_id in available_metrics:
                config = METRICS_CONFIG[metric_id]
                print(f"📈 {config['name']}:")
                
                for stage_name, milestone in milestones.items():
                    value = milestone[metric_id]
                    games = milestone['n_games']
                    date = milestone['date'].strftime('%m/%d')
                    
                    if pd.notna(value):
                        print(f"   {stage_name:20} ({date}, {games:3.0f} games): "
                              f"{value:6.2f}{config['unit']}")
                
                # Show overall change
                if len(milestones) >= 2:
                    first_value = list(milestones.values())[0][metric_id]
                    last_value = list(milestones.values())[-1][metric_id]
                    if pd.notna(first_value) and pd.notna(last_value):
                        change = last_value - first_value
                        change_pct = (change / first_value * 100) if first_value != 0 else 0
                        print(f"   {'Change:':20} {change:+6.2f}{config['unit']} "
                              f"({change_pct:+.1f}%)")
                
                print()
        print()
    
    # Step 6: Detailed intra-season analysis
    print_section("STEP 6: Detailed Intra-Season Analysis")
    
    print("How metrics have evolved throughout the 2025-26 season:")
    print()
    
    for _, row in analysis_df.iterrows():
        intra = row.get('IntraSeason')
        if not intra or not isinstance(intra, dict):
            continue
        
        print(f"📊 {row['Metric']}")
        print(f"   Early Season ({intra['early_date'].strftime('%m/%d')}, "
              f"{intra['early_games']:.0f} games): {intra['early_value']:.2f}{row['Unit']}")
        print(f"   Current ({intra['current_date'].strftime('%m/%d')}, "
              f"{intra['current_games']:.0f} games): {intra['current_value']:.2f}{row['Unit']}")
        print(f"   Change: {intra['season_change']:+.2f}{row['Unit']} "
              f"({intra['season_change_pct']:+.1f}%)")
        print(f"   Trajectory: {intra['acceleration']}")
        
        if intra['inflection_point']:
            print(f"   ⚡ Notable shift detected: {intra['inflection_point'].strftime('%m/%d/%Y')} "
                  f"(value: {intra['inflection_value']:.2f}{row['Unit']})")
        
        # Interpretation
        if abs(intra['season_change_pct']) > 5:
            print(f"   ⚠️  Significant intra-season change (>{5}%)")
        elif abs(intra['season_change_pct']) > 2:
            print(f"   ⚡ Moderate intra-season change (>{2}%)")
        else:
            print(f"   ✓ Relatively stable within season")
        
        print()
    
    # Step 7: Key findings and recommendations
    print_section("STEP 7: Key Findings & Recommendations")
    
    # Identify most significant changes vs. history
    top_changes = analysis_df.head(3)
    
    print("Top 3 Regime Changes vs. Historical Baseline:")
    for idx, (_, row) in enumerate(top_changes.iterrows(), 1):
        direction = "above" if row['Difference'] > 0 else "below"
        print(f"  {idx}. {row['Metric']}")
        print(f"     Current: {row['Current']:.2f}{row['Unit']} "
              f"({row['Difference']:+.2f}{row['Unit']} {direction} baseline)")
        print(f"     Z-Score: {row['Z-Score']:+.2f} σ  |  Trend: {row['Trend']}")
        print()
    
    # Identify biggest intra-season changes
    intra_season_changes = []
    for _, row in analysis_df.iterrows():
        intra = row.get('IntraSeason')
        if intra and isinstance(intra, dict):
            intra_season_changes.append({
                'metric': row['Metric'],
                'change_pct': abs(intra['season_change_pct']),
                'change': intra['season_change'],
                'unit': row['Unit']
            })
    
    if intra_season_changes:
        intra_season_changes.sort(key=lambda x: x['change_pct'], reverse=True)
        top_intra = intra_season_changes[:3]
        
        print("Top 3 Intra-Season Changes (Early → Current):")
        for idx, item in enumerate(top_intra, 1):
            print(f"  {idx}. {item['metric']}")
            print(f"     Change: {item['change']:+.2f}{item['unit']} "
                  f"({item['change_pct']:+.1f}% from early season)")
            print()
    
    # Combined assessment
    print("Regime Change Assessment:")
    print()
    
    # Check for metrics with both historical deviation AND intra-season instability
    dual_concerns = []
    for _, row in analysis_df.iterrows():
        if "CRITICAL" in row['Status'] or "ALERT" in row['Status']:
            intra = row.get('IntraSeason')
            if intra and isinstance(intra, dict):
                if abs(intra['season_change_pct']) > 3:
                    dual_concerns.append({
                        'metric': row['Metric'],
                        'z_score': row['Z-Score'],
                        'change_pct': intra['season_change_pct'],
                        'acceleration': intra['acceleration']
                    })
    
    if dual_concerns:
        print("⚠️  Metrics with BOTH historical deviation AND intra-season instability:")
        for concern in dual_concerns:
            print(f"   • {concern['metric']}: {concern['z_score']:+.2f}σ from history, "
                  f"{concern['change_pct']:+.1f}% within season")
            print(f"     Trajectory: {concern['acceleration']}")
        print()
        print("   → These represent the highest-risk areas for model performance")
        print("   → Prioritize monitoring and potential model adjustments")
        print()
    
    # Monitoring recommendations
    print("Monitoring Recommendations:")
    
    if critical > 0:
        print("  🚨 CRITICAL: Multiple metrics showing significant regime changes")
        print("     → Investigate rule changes or league-wide tactical shifts")
        print("     → Consider model retraining with current season data")
        print("     → Update feature importance assumptions")
        
        # Check if changes are stabilizing or accelerating
        accelerating = sum(1 for _, row in analysis_df.iterrows() 
                          if row.get('Acceleration') == "Accelerating upward")
        if accelerating > 0:
            print(f"     ⚠️  {accelerating} metric(s) still accelerating - regime change ongoing")
        
    elif alerts > 0:
        print("  ⚠️  ALERT: Some metrics deviating from historical norms")
        print("     → Monitor closely for continued divergence")
        print("     → Test model performance on current season data")
        print("     → Consider ensemble approaches for robustness")
    else:
        print("  ✓ NORMAL: All metrics within expected ranges")
        print("     → Continue with standard monitoring schedule")
        print("     → Existing models should perform as expected")
    
    print()
    
    print("Next Steps:")
    print("  1. Run this script weekly to track metric trends")
    print("  2. Document any external factors (rule changes, injuries, etc.)")
    print("  3. Compare model predictions vs. actual results for metrics with high alerts")
    print("  4. Update models if regime changes persist for 3+ weeks")
    print("  5. Consider feature engineering based on diverging metrics")
    print("  6. Focus on 🎯 model features showing deviation (CRITICAL for predictions)")
    print("  7. Monitor if league normalization baselines are shifting")
    print()
    
    print("=" * 80)
    print("MODEL ALIGNMENT NOTE")
    print("=" * 80)
    print()
    print("This monitoring script now tracks ALL metrics used in the Champion model:")
    print()
    print("  Core Model Features (🎯):")
    print("    • OEFF, DEFF - Used in 'edge' features (offensive vs defensive matchup)")
    print("    • ORr, DRr - Used in 'orb_edge' features (rebounding battles)")
    print("    • TOr - Used in 'tov_edge' and team turnover features")
    print("    • 3PAr - Used in 'tpar' features (3-point attempt rates)")
    print("    • ASTr - Used in 'astr' features (assist rates)")
    print("    • Pace - Used in 'pace_mean' and 'pace_diff' features")
    print()
    print("  Excluded from Champion Model:")
    print("    • FTR - Excluded due to regime-specific behavior")
    print("      (still monitored for context)")
    print()
    print("  Context Metrics (not direct features):")
    print("    • eFG%, TS%, 3P%, PPG - Provide broader context")
    print()
    print("  All model features use league-relative normalization:")
    print("    normalized_value = raw_team_value - league_avg_on_date")
    print()
    print("  This script tracks the league_avg_on_date baseline to detect")
    print("  shifts that would affect the normalization process.")
    print()
    
    # Save analysis to CSV
    output_file = Path("analysis") / f"metrics_analysis_{datetime.now().strftime('%Y%m%d')}.csv"
    output_file.parent.mkdir(exist_ok=True)
    analysis_df.to_csv(output_file, index=False)
    print(f"Analysis saved to: {output_file}")
    print()
    
    print("=" * 80)
    print(f"Analysis complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()

