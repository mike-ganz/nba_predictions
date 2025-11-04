"""
Analyze correlations between candidate new features (b2b, three_in_four, pace_diff)
and currently used features to assess incremental information value.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict

from data.loaders import GameDataLoader
from features.builder import FeatureBuilder, HOME_FEATURE_KEYS, AWAY_FEATURE_KEYS, SHARED_FEATURE_KEYS
from training.margin_dataset import MarginTrainingDataset


def extract_all_features(records, exclude_features: List[str] = None) -> pd.DataFrame:
    """
    Extract both currently used features and candidate new features.
    """
    if exclude_features is None:
        exclude_features = []
    
    builder = FeatureBuilder()
    rows = []
    
    for record in records:
        features = builder.build(record)
        
        # Currently used features
        row = {}
        
        # Home features
        for k in HOME_FEATURE_KEYS:
            if f'home_{k}' not in exclude_features:
                row[f'home_{k}'] = features.x_home[k]
        
        # Away features
        for k in AWAY_FEATURE_KEYS:
            if f'away_{k}' not in exclude_features:
                row[f'away_{k}'] = features.x_away[k]
        
        # Shared features
        for k in SHARED_FEATURE_KEYS:
            if f'shared_{k}' not in exclude_features:
                row[f'shared_{k}'] = features.x_shared[k]
        
        # Difference features (currently used in Ridge model)
        for k in HOME_FEATURE_KEYS:
            if k in AWAY_FEATURE_KEYS:
                if f'diff_{k}' not in exclude_features:
                    row[f'diff_{k}'] = features.x_home[k] - features.x_away.get(k, 0)
        
        # NEW CANDIDATE FEATURES
        # Extract from raw record data
        home_team = record.teams.H
        away_team = record.teams.A
        
        # B2B flags
        row['NEW_home_b2b'] = float(home_team.b2b) if home_team.b2b is not None else 0.0
        row['NEW_away_b2b'] = float(away_team.b2b) if away_team.b2b is not None else 0.0
        row['NEW_diff_b2b'] = row['NEW_home_b2b'] - row['NEW_away_b2b']
        
        # Three in four flags
        row['NEW_home_three_in_four'] = float(home_team.three_in_four) if home_team.three_in_four is not None else 0.0
        row['NEW_away_three_in_four'] = float(away_team.three_in_four) if away_team.three_in_four is not None else 0.0
        row['NEW_diff_three_in_four'] = row['NEW_home_three_in_four'] - row['NEW_away_three_in_four']
        
        # Pace diff (already computed in matchup but not used)
        home_pace = home_team.pace_norm if home_team.pace_norm is not None else home_team.pace
        away_pace = away_team.pace_norm if away_team.pace_norm is not None else away_team.pace
        row['NEW_pace_diff'] = home_pace - away_pace
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def analyze_correlations(df: pd.DataFrame) -> Dict:
    """
    Compute correlations between new features and existing features.
    """
    # Separate existing and new features
    existing_cols = [c for c in df.columns if not c.startswith('NEW_')]
    new_cols = [c for c in df.columns if c.startswith('NEW_')]
    
    # Compute correlation matrix
    corr_matrix = df.corr()
    
    # Extract correlations between new and existing features
    new_to_existing = corr_matrix.loc[new_cols, existing_cols]
    
    # Also get correlations among new features themselves
    new_to_new = corr_matrix.loc[new_cols, new_cols]
    
    return {
        'full_corr': corr_matrix,
        'new_to_existing': new_to_existing,
        'new_to_new': new_to_new,
        'existing_cols': existing_cols,
        'new_cols': new_cols
    }


def print_correlation_report(corr_data: Dict, df: pd.DataFrame):
    """
    Print a formatted correlation report.
    """
    print("=" * 100)
    print("NEW FEATURE CORRELATION ANALYSIS")
    print("=" * 100)
    
    print(f"\nDataset: {len(df)} games")
    print(f"Existing features: {len(corr_data['existing_cols'])}")
    print(f"New candidate features: {len(corr_data['new_cols'])}")
    
    # Check data availability
    print("\n" + "=" * 100)
    print("DATA AVAILABILITY CHECK")
    print("=" * 100)
    
    new_feature_stats = []
    for col in corr_data['new_cols']:
        non_zero = (df[col] != 0).sum()
        pct_non_zero = 100 * non_zero / len(df)
        mean_val = df[col].mean()
        std_val = df[col].std()
        
        new_feature_stats.append({
            'Feature': col,
            'Non-Zero Count': non_zero,
            'Non-Zero %': f"{pct_non_zero:.1f}%",
            'Mean': f"{mean_val:.3f}",
            'Std': f"{std_val:.3f}"
        })
    
    stats_df = pd.DataFrame(new_feature_stats)
    print("\n" + stats_df.to_string(index=False))
    
    # Correlations among new features
    print("\n" + "=" * 100)
    print("CORRELATIONS AMONG NEW FEATURES")
    print("=" * 100)
    print("\n" + corr_data['new_to_new'].to_string())
    
    # Top correlations with existing features
    print("\n" + "=" * 100)
    print("TOP CORRELATIONS: NEW FEATURES vs EXISTING FEATURES")
    print("=" * 100)
    
    new_to_existing = corr_data['new_to_existing']
    
    for new_feat in corr_data['new_cols']:
        print(f"\n{new_feat}:")
        print("-" * 80)
        
        # Get correlations for this new feature (absolute value for ranking)
        corrs = new_to_existing.loc[new_feat].abs().sort_values(ascending=False)
        
        # Show top 10
        print("\nTop 10 correlations with existing features:")
        for feat, corr in corrs.head(10).items():
            actual_corr = new_to_existing.loc[new_feat, feat]
            print(f"  {feat:40s}: {actual_corr:+.3f} (|corr|={corr:.3f})")
        
        # Check if highly correlated with any existing feature
        max_corr = corrs.iloc[0]
        if max_corr > 0.7:
            print(f"\n  ⚠️  WARNING: High correlation ({max_corr:.3f}) with existing feature!")
            print(f"      This feature may have LIMITED incremental value.")
        elif max_corr > 0.5:
            print(f"\n  ⚡ MODERATE correlation ({max_corr:.3f}) detected.")
            print(f"      May still provide some incremental information.")
        else:
            print(f"\n  ✅ LOW correlation ({max_corr:.3f}) - likely provides NEW information!")
    
    # Summary recommendations
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)
    
    recommendations = []
    
    for new_feat in corr_data['new_cols']:
        corrs = new_to_existing.loc[new_feat].abs()
        max_corr = corrs.max()
        non_zero_pct = 100 * (df[new_feat] != 0).sum() / len(df)
        
        # Handle case where all correlations are NaN (no variance in feature)
        if pd.isna(max_corr):
            status = "❌ SKIP"
            reason = f"No data available ({non_zero_pct:.1f}% non-zero)"
            max_feat_str = "N/A"
            max_corr_str = "N/A"
        else:
            max_feat = corrs.idxmax()
            max_feat_str = str(max_feat)[:35]
            max_corr_str = f"{max_corr:.3f}"
            
            if non_zero_pct < 5:
                status = "❌ SKIP"
                reason = f"Insufficient data ({non_zero_pct:.1f}% non-zero)"
            elif max_corr > 0.7:
                status = "⚠️  MAYBE"
                reason = f"High correlation with {max_feat} ({max_corr:.3f})"
            elif max_corr > 0.5:
                status = "⚡ TEST"
                reason = f"Moderate correlation with {max_feat} ({max_corr:.3f})"
            else:
                status = "✅ ADD"
                reason = f"Low correlation, likely new info (max={max_corr:.3f})"
        
        recommendations.append({
            'Feature': new_feat.replace('NEW_', ''),
            'Status': status,
            'Max |Corr|': max_corr_str,
            'Most Correlated With': max_feat_str,
            'Data %': f"{non_zero_pct:.1f}%",
            'Reason': reason
        })
    
    rec_df = pd.DataFrame(recommendations)
    print("\n" + rec_df.to_string(index=False))
    
    print("\n" + "=" * 100)
    print("NEXT STEPS")
    print("=" * 100)
    print("""
1. For features marked ✅ ADD: Strong candidates, implement immediately
2. For features marked ⚡ TEST: Worth trying, but monitor for multicollinearity
3. For features marked ⚠️  MAYBE: May provide marginal value, test if time permits
4. For features marked ❌ SKIP: Insufficient data or redundant, don't implement

After adding features, re-run permutation importance analysis to validate.
""")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze correlations between candidate and existing features"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/games_2025_2026_current_norm.jsonl",
        help="Path to training data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis/new_feature_correlations",
        help="Output directory for correlation reports"
    )
    parser.add_argument(
        "--exclude-features",
        nargs="*",
        default=[
            "shared_team_weighted_ts_away",
            "away_usage_share_top2",
            "shared_implied_home_winprob"
        ],
        help="Features to exclude (same as model config)"
    )
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.data}...")
    data_path = Path(args.data)
    loader = GameDataLoader(data_path.parent)
    collection = loader.read_games(data_path.name)
    records = collection.games
    print(f"  Loaded {len(records)} games\n")
    
    # Extract features
    print("Extracting features (this may take a minute)...")
    df = extract_all_features(records, exclude_features=args.exclude_features)
    print(f"  Extracted {len(df.columns)} total features\n")
    
    # Analyze correlations
    print("Computing correlations...")
    corr_data = analyze_correlations(df)
    
    # Print report
    print_correlation_report(corr_data, df)
    
    # Save detailed outputs
    print(f"\n\nSaving detailed results to {output_dir}/")
    
    # Save full correlation matrix
    corr_data['full_corr'].to_csv(output_dir / "full_correlation_matrix.csv")
    print(f"  ✓ Saved: full_correlation_matrix.csv")
    
    # Save new-to-existing correlations
    corr_data['new_to_existing'].to_csv(output_dir / "new_to_existing_correlations.csv")
    print(f"  ✓ Saved: new_to_existing_correlations.csv")
    
    # Save feature data
    df.to_csv(output_dir / "feature_data.csv", index=False)
    print(f"  ✓ Saved: feature_data.csv")
    
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()

