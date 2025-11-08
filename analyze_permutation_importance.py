#!/usr/bin/env python
"""
Permutation importance analysis for XGBoost models across different datasets.

This script measures how much each feature contributes to ATS accuracy by
shuffling features and measuring performance degradation. Unlike intrinsic
importance (Gain/Weight), permutation importance is DATASET-SPECIFIC and can
reveal which features are more valuable on different time periods.

Key Question: Are FTR and role indicator features MORE important on 2025-26
(with increased FTR) than on the 2024-25 holdout?

Usage:
    python analyze_permutation_importance.py \
        --model artifacts/margin_experiment2_ftr_plus_role \
        --data1 data/games_predict_2024_2025_with_players_norm.jsonl \
        --data2 data/games_2025_2026_current_norm.jsonl \
        --output analysis/permutation_importance
"""

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from data.loaders import GameDataLoader
from training.margin_dataset import MarginTrainingDataset
from training.margin_dataset_extended import MarginTrainingDatasetExtended
from features.builder import HOME_FEATURE_KEYS, AWAY_FEATURE_KEYS, SHARED_FEATURE_KEYS
from features.builder_extended import (
    HOME_FEATURE_KEYS_EXTENDED,
    AWAY_FEATURE_KEYS_EXTENDED,
    SHARED_FEATURE_KEYS_EXTENDED,
)


def get_feature_names(exclude_features=None, include_diff_features=True, include_fav_underdog=False):
    """Get feature names in order."""
    exclude_features = exclude_features or []
    names = []
    
    if include_fav_underdog:
        home_keys = HOME_FEATURE_KEYS_EXTENDED
        away_keys = AWAY_FEATURE_KEYS_EXTENDED
        shared_keys = SHARED_FEATURE_KEYS_EXTENDED
    else:
        home_keys = HOME_FEATURE_KEYS
        away_keys = AWAY_FEATURE_KEYS
        shared_keys = SHARED_FEATURE_KEYS
    
    # Home features
    for k in home_keys:
        if f'home_{k}' not in exclude_features:
            names.append(f"home_{k}")
    
    # Away features
    for k in away_keys:
        if f'away_{k}' not in exclude_features:
            names.append(f"away_{k}")
    
    # Shared features
    shared_keys_to_use = shared_keys.copy()
    if not include_fav_underdog:
        shared_keys_to_use = [k for k in shared_keys_to_use 
                             if k not in ['is_home_favorite', 'is_home_underdog', 
                                         'is_away_favorite', 'is_away_underdog']]
    
    for k in shared_keys_to_use:
        if f'shared_{k}' not in exclude_features:
            names.append(f"shared_{k}")
    
    # Difference features
    if include_diff_features:
        for k in home_keys:
            if k in away_keys:
                if f'home_{k}' not in exclude_features and f'away_{k}' not in exclude_features:
                    names.append(f"diff_{k}")
    
    return names


def load_dataset(data_path: Path, exclude_features: List[str], 
                include_diff_features: bool, include_fav_underdog: bool):
    """Load and build dataset."""
    loader = GameDataLoader(data_path.parent)
    collection = loader.read_games(data_path.name)
    records = collection.games
    
    # Choose dataset builder based on config
    if include_fav_underdog or 'home_ftr' not in exclude_features:
        dataset = MarginTrainingDatasetExtended(
            records,
            exclude_features=exclude_features,
            include_diff_features=include_diff_features,
            include_fav_underdog_features=include_fav_underdog
        )
    else:
        dataset = MarginTrainingDataset(
            records,
            exclude_features=exclude_features,
            include_diff_features=include_diff_features
        )
    
    batch = dataset.build()
    return batch, len(records)


def compute_ats_accuracy(pred_margins: np.ndarray, actual_margins: np.ndarray, 
                        spreads: np.ndarray) -> float:
    """Compute ATS accuracy."""
    # Home covers if actual_margin > -spread
    actual_home_covers = (actual_margins > -spreads).astype(int)
    
    # Model picks home if pred > -spread
    pred_home_covers = (pred_margins > -spreads).astype(int)
    
    # ATS accuracy
    correct = (pred_home_covers == actual_home_covers).astype(int)
    return correct.mean()


def permutation_importance_single_dataset(
    model,
    X: np.ndarray,
    y_margin: np.ndarray,
    baseline_margin: np.ndarray,
    spreads: np.ndarray,
    feature_names: List[str],
    n_permutations: int = 10,
    seed: int = 42
) -> pd.DataFrame:
    """
    Compute permutation importance for a single dataset.
    
    Returns:
        DataFrame with columns: feature, importance, importance_std, baseline_ats
    """
    rng = np.random.default_rng(seed)
    
    # Baseline performance
    mu_pred = model.predict(X, baseline_margin)
    baseline_ats = compute_ats_accuracy(mu_pred, y_margin, spreads)
    
    print(f"  Baseline ATS: {baseline_ats:.4f} ({baseline_ats*100:.2f}%)")
    
    importances = []
    
    for i, name in enumerate(feature_names):
        deltas = []
        
        for perm_idx in range(n_permutations):
            # Permute this feature
            X_perm = X.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            
            # Re-predict
            mu_pred_perm = model.predict(X_perm, baseline_margin)
            ats_perm = compute_ats_accuracy(mu_pred_perm, y_margin, spreads)
            
            # Importance = degradation in ATS when feature is shuffled
            deltas.append(baseline_ats - ats_perm)
        
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)
        
        importances.append({
            'feature': name,
            'importance': mean_delta,
            'importance_std': std_delta,
            'baseline_ats': baseline_ats
        })
        
        if (i + 1) % 5 == 0 or (i + 1) == len(feature_names):
            print(f"    Processed {i + 1}/{len(feature_names)} features...")
    
    df = pd.DataFrame(importances)
    df = df.sort_values('importance', ascending=False)
    
    return df


def plot_permutation_comparison(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    dataset1_name: str,
    dataset2_name: str,
    output_file: Path,
    top_n: int = 15
):
    """Plot side-by-side comparison of permutation importance."""
    
    # Get union of top features from both datasets
    top_features_1 = set(df1.head(top_n)['feature'])
    top_features_2 = set(df2.head(top_n)['feature'])
    all_top_features = list(top_features_1 | top_features_2)
    
    # Create comparison dataframe
    comparison = pd.DataFrame({'feature': all_top_features})
    
    # Merge importance from both datasets
    comparison = comparison.merge(
        df1[['feature', 'importance']].rename(columns={'importance': f'{dataset1_name}_imp'}),
        on='feature',
        how='left'
    )
    comparison = comparison.merge(
        df2[['feature', 'importance']].rename(columns={'importance': f'{dataset2_name}_imp'}),
        on='feature',
        how='left'
    )
    
    # Fill NaN with 0
    comparison = comparison.fillna(0)
    
    # Sort by sum of importances
    comparison['total_importance'] = comparison[f'{dataset1_name}_imp'] + comparison[f'{dataset2_name}_imp']
    comparison = comparison.sort_values('total_importance', ascending=True)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, max(8, len(comparison) * 0.4)))
    
    y_pos = np.arange(len(comparison))
    width = 0.35
    
    # Convert to percentage points
    comparison[f'{dataset1_name}_imp'] = comparison[f'{dataset1_name}_imp'] * 100
    comparison[f'{dataset2_name}_imp'] = comparison[f'{dataset2_name}_imp'] * 100
    
    ax.barh(y_pos - width/2, comparison[f'{dataset1_name}_imp'], 
            width, label=dataset1_name, color='#3b82f6', alpha=0.8)
    ax.barh(y_pos + width/2, comparison[f'{dataset2_name}_imp'], 
            width, label=dataset2_name, color='#10b981', alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparison['feature'])
    ax.set_xlabel('ATS Degradation when Feature Shuffled (percentage points)', fontsize=12)
    ax.set_title(f'Permutation Importance Comparison\n{dataset1_name} vs {dataset2_name}', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved plot to {output_file}")
    plt.close()
    
    return comparison


def analyze_feature_categories(df1: pd.DataFrame, df2: pd.DataFrame,
                               dataset1_name: str, dataset2_name: str) -> pd.DataFrame:
    """Analyze importance by feature category (FTR, role indicators, etc.)."""
    
    categories = {
        'FTR': ['home_ftr', 'away_ftr'],
        'Role Indicators': ['shared_is_home_favorite', 'shared_is_home_underdog',
                           'shared_is_away_favorite', 'shared_is_away_underdog'],
        'Edge/Rating': ['home_edge', 'away_edge', 'home_orb_edge'],
        'Turnover': ['home_tov_edge', 'away_tov_edge'],
        'Market Implied': ['shared_implied_home_winprob', 'shared_implied_away_winprob'],
        'Shooting': ['shared_team_weighted_ts_home', 'shared_team_weighted_ts_away',
                    'home_tpar', 'away_tpar'],
        'Player Availability': ['home_star_out', 'away_star_out', 
                               'home_minutes_missing_top2', 'away_minutes_missing_top2',
                               'home_usage_share_top2', 'away_usage_share_top2'],
    }
    
    results = []
    
    for category, features in categories.items():
        # Sum importance for features in this category
        imp1 = df1[df1['feature'].isin(features)]['importance'].sum()
        imp2 = df2[df2['feature'].isin(features)]['importance'].sum()
        
        # Count how many features exist
        count1 = df1['feature'].isin(features).sum()
        count2 = df2['feature'].isin(features).sum()
        
        results.append({
            'category': category,
            f'{dataset1_name}_importance': imp1,
            f'{dataset2_name}_importance': imp2,
            'difference': imp2 - imp1,
            'pct_change': ((imp2 - imp1) / imp1 * 100) if imp1 > 0 else 0,
            'num_features': max(count1, count2)
        })
    
    df_categories = pd.DataFrame(results)
    df_categories = df_categories.sort_values('difference', ascending=False)
    
    return df_categories


def main():
    parser = argparse.ArgumentParser(
        description="Analyze permutation importance across datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--data1",
        type=str,
        required=True,
        help="Path to first dataset (e.g., 2024-25 holdout)"
    )
    parser.add_argument(
        "--data2",
        type=str,
        required=True,
        help="Path to second dataset (e.g., 2025-26 current)"
    )
    parser.add_argument(
        "--data1-name",
        type=str,
        default="2024-25",
        help="Display name for first dataset"
    )
    parser.add_argument(
        "--data2-name",
        type=str,
        default="2025-26",
        help="Display name for second dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis/permutation_importance",
        help="Output directory"
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10,
        help="Number of permutations per feature (default: 10)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of top features to display (default: 15)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("PERMUTATION IMPORTANCE ANALYSIS")
    print("=" * 80)
    print()
    print(f"Model: {args.model}")
    print(f"Dataset 1 ({args.data1_name}): {args.data1}")
    print(f"Dataset 2 ({args.data2_name}): {args.data2}")
    print(f"Permutations per feature: {args.n_permutations}")
    print()
    
    # Load model and config
    model_path = Path(args.model)
    model_file = model_path / "margin_model.joblib"
    print(f"Loading model from {model_file}...")
    model = joblib.load(model_file)
    
    config_file = model_path / "config.yaml"
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    model_config = config.get('model', {})
    exclude_features = model_config.get('exclude_features', [])
    include_diff_features = model_config.get('include_diff_features', False)
    include_fav_underdog = model_config.get('include_fav_underdog_features', False)
    
    print(f"  Excluded features: {len(exclude_features)}")
    print(f"  Include fav/underdog: {include_fav_underdog}")
    print()
    
    # Get feature names
    feature_names = get_feature_names(
        exclude_features=exclude_features,
        include_diff_features=include_diff_features,
        include_fav_underdog=include_fav_underdog
    )
    print(f"Total features: {len(feature_names)}")
    print()
    
    # Load Dataset 1
    print("=" * 80)
    print(f"DATASET 1: {args.data1_name}")
    print("=" * 80)
    print(f"Loading data from {args.data1}...")
    batch1, n_games1 = load_dataset(
        Path(args.data1),
        exclude_features,
        include_diff_features,
        include_fav_underdog
    )
    print(f"  Loaded {n_games1} games")
    print()
    
    print("Computing permutation importance...")
    df1 = permutation_importance_single_dataset(
        model,
        batch1.x,
        batch1.y_margin,
        batch1.baseline_margin,
        batch1.market_spread_home,
        feature_names,
        n_permutations=args.n_permutations
    )
    print()
    
    # Load Dataset 2
    print("=" * 80)
    print(f"DATASET 2: {args.data2_name}")
    print("=" * 80)
    print(f"Loading data from {args.data2}...")
    batch2, n_games2 = load_dataset(
        Path(args.data2),
        exclude_features,
        include_diff_features,
        include_fav_underdog
    )
    print(f"  Loaded {n_games2} games")
    print()
    
    print("Computing permutation importance...")
    df2 = permutation_importance_single_dataset(
        model,
        batch2.x,
        batch2.y_margin,
        batch2.baseline_margin,
        batch2.market_spread_home,
        feature_names,
        n_permutations=args.n_permutations
    )
    print()
    
    # Display results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    
    print(f"Top {args.top_n} Features by Permutation Importance")
    print()
    
    print(f"{args.data1_name} ({n_games1} games, baseline ATS: {df1['baseline_ats'].iloc[0]*100:.2f}%):")
    display_df1 = df1.head(args.top_n)[['feature', 'importance', 'importance_std']].copy()
    display_df1['importance_pct'] = display_df1['importance'] * 100
    display_df1['importance_std_pct'] = display_df1['importance_std'] * 100
    print(display_df1[['feature', 'importance_pct', 'importance_std_pct']].to_string(index=False))
    print()
    
    print(f"{args.data2_name} ({n_games2} games, baseline ATS: {df2['baseline_ats'].iloc[0]*100:.2f}%):")
    display_df2 = df2.head(args.top_n)[['feature', 'importance', 'importance_std']].copy()
    display_df2['importance_pct'] = display_df2['importance'] * 100
    display_df2['importance_std_pct'] = display_df2['importance_std'] * 100
    print(display_df2[['feature', 'importance_pct', 'importance_std_pct']].to_string(index=False))
    print()
    
    # Analyze by category
    print("=" * 80)
    print("IMPORTANCE BY FEATURE CATEGORY")
    print("=" * 80)
    print()
    
    category_df = analyze_feature_categories(df1, df2, args.data1_name, args.data2_name)
    
    # Format for display
    display_cat = category_df.copy()
    display_cat[f'{args.data1_name}_pct'] = display_cat[f'{args.data1_name}_importance'] * 100
    display_cat[f'{args.data2_name}_pct'] = display_cat[f'{args.data2_name}_importance'] * 100
    display_cat['diff_pct'] = display_cat['difference'] * 100
    
    print(display_cat[['category', f'{args.data1_name}_pct', f'{args.data2_name}_pct', 
                       'diff_pct', 'pct_change', 'num_features']].to_string(index=False))
    print()
    
    # Highlight key findings
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()
    
    # Find biggest gainers
    merged = df1[['feature', 'importance']].merge(
        df2[['feature', 'importance']],
        on='feature',
        suffixes=(f'_{args.data1_name}', f'_{args.data2_name}')
    )
    merged['difference'] = merged[f'importance_{args.data2_name}'] - merged[f'importance_{args.data1_name}']
    merged = merged.sort_values('difference', ascending=False)
    
    print(f"Features MORE important on {args.data2_name} (top 5):")
    gainers = merged.head(5)
    for _, row in gainers.iterrows():
        diff_pct = row['difference'] * 100
        feat = row['feature']
        print(f"  {feat}: +{diff_pct:.3f} pp")
    print()
    
    print(f"Features LESS important on {args.data2_name} (top 5):")
    losers = merged.tail(5).iloc[::-1]
    for _, row in losers.iterrows():
        diff_pct = row['difference'] * 100
        feat = row['feature']
        print(f"  {feat}: {diff_pct:.3f} pp")
    print()
    
    # Save results
    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    print()
    
    df1.to_csv(output_dir / f"permutation_importance_{args.data1_name}.csv", index=False)
    print(f"Saved: {output_dir / f'permutation_importance_{args.data1_name}.csv'}")
    
    df2.to_csv(output_dir / f"permutation_importance_{args.data2_name}.csv", index=False)
    print(f"Saved: {output_dir / f'permutation_importance_{args.data2_name}.csv'}")
    
    category_df.to_csv(output_dir / "category_comparison.csv", index=False)
    print(f"Saved: {output_dir / 'category_comparison.csv'}")
    
    merged.to_csv(output_dir / "feature_comparison.csv", index=False)
    print(f"Saved: {output_dir / 'feature_comparison.csv'}")
    print()
    
    # Generate plot
    print("Generating comparison plot...")
    plot_file = output_dir / f"permutation_comparison_{args.data1_name}_vs_{args.data2_name}.png"
    comparison = plot_permutation_comparison(
        df1, df2,
        args.data1_name, args.data2_name,
        plot_file,
        top_n=args.top_n
    )
    print()
    
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()

