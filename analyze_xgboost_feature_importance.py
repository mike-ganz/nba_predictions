#!/usr/bin/env python
"""
Feature importance analysis for XGBoost margin prediction models.

Analyzes feature importance for XGBoost models (including extended models with
home_ftr and favorite/underdog role indicators) using multiple metrics:
1. Gain: Average gain across all splits using the feature
2. Weight: Number of times feature is used in splits
3. Cover: Average coverage (samples affected) across splits

Usage:
    python analyze_xgboost_feature_importance.py --model artifacts/margin_experiment2_ftr_plus_role
    
    # Compare multiple models
    python analyze_xgboost_feature_importance.py \
        --model artifacts/margin_experiment2_ftr_plus_role \
        --compare artifacts/margin_champion \
        --output analysis/exp2_vs_champion
"""

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml
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
    """Get feature names in the order they appear in the combined feature vector.
    
    Args:
        exclude_features: List of features to exclude
        include_diff_features: Whether difference features are included
        include_fav_underdog: Whether favorite/underdog indicators are included
    
    Returns:
        List of feature names in order
    """
    exclude_features = exclude_features or []
    names = []
    
    # Determine which feature keys to use
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
    
    # Shared features (filter out fav/underdog if not included)
    shared_keys_to_use = shared_keys.copy()
    if not include_fav_underdog:
        shared_keys_to_use = [k for k in shared_keys_to_use 
                             if k not in ['is_home_favorite', 'is_home_underdog', 
                                         'is_away_favorite', 'is_away_underdog']]
    
    for k in shared_keys_to_use:
        if f'shared_{k}' not in exclude_features:
            names.append(f"shared_{k}")
    
    # Difference features (only for features that exist on both sides)
    if include_diff_features:
        for k in home_keys:
            if k in away_keys:
                # Only add if neither home nor away version is excluded
                if f'home_{k}' not in exclude_features and f'away_{k}' not in exclude_features:
                    names.append(f"diff_{k}")
    
    return names


def load_model_and_config(model_path):
    """Load model and its configuration."""
    model_path = Path(model_path)
    
    # Load model
    model_file = model_path / "margin_model.joblib" if model_path.is_dir() else model_path
    print(f"Loading model from {model_file}...")
    model = joblib.load(model_file)
    
    # Load config
    if model_path.is_dir():
        config_file = model_path / "config.yaml"
    else:
        config_file = model_path.parent / "config.yaml"
    
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f)
    else:
        config = {}
        print("  Warning: config.yaml not found, using defaults")
    
    return model, config


def extract_xgboost_importance(model, feature_names):
    """Extract feature importance from XGBoost model.
    
    Args:
        model: Trained MarginXGBoostModel or XGBoost model
        feature_names: List of feature names
    
    Returns:
        DataFrame with feature importance by different metrics
    """
    # Get the underlying booster
    # Handle both MarginXGBoostModel wrapper and raw XGBoost models
    if hasattr(model, 'model') and model.model is not None:
        # MarginXGBoostModel wrapper
        booster = model.model.get_booster()
    elif hasattr(model, 'get_booster'):
        # Raw XGBoost model
        booster = model.get_booster()
    else:
        raise ValueError(f"Cannot extract booster from model of type {type(model)}")
    
    # Get importance by different metrics
    importance_gain = booster.get_score(importance_type='gain')
    importance_weight = booster.get_score(importance_type='weight')
    importance_cover = booster.get_score(importance_type='cover')
    
    # Create dataframe
    data = []
    for i, name in enumerate(feature_names):
        feature_key = f'f{i}'  # XGBoost uses f0, f1, f2, ...
        
        gain = importance_gain.get(feature_key, 0.0)
        weight = importance_weight.get(feature_key, 0.0)
        cover = importance_cover.get(feature_key, 0.0)
        
        data.append({
            'feature': name,
            'gain': gain,
            'weight': weight,
            'cover': cover,
        })
    
    df = pd.DataFrame(data)
    
    # Normalize scores to 0-100 scale for comparison
    if df['gain'].sum() > 0:
        df['gain_pct'] = 100 * df['gain'] / df['gain'].sum()
    else:
        df['gain_pct'] = 0.0
        
    if df['weight'].sum() > 0:
        df['weight_pct'] = 100 * df['weight'] / df['weight'].sum()
    else:
        df['weight_pct'] = 0.0
        
    if df['cover'].sum() > 0:
        df['cover_pct'] = 100 * df['cover'] / df['cover'].sum()
    else:
        df['cover_pct'] = 0.0
    
    # Sort by gain (most informative metric)
    df = df.sort_values('gain', ascending=False)
    
    return df


def plot_importance(df, top_n=20, output_file=None):
    """Create visualization of feature importance.
    
    Args:
        df: Feature importance dataframe
        top_n: Number of top features to plot
        output_file: Path to save plot (if None, display only)
    """
    # Get top N features by gain
    top_features = df.head(top_n).copy()
    top_features = top_features.sort_values('gain', ascending=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle(f'Top {top_n} Features by Importance', fontsize=16, fontweight='bold')
    
    # Plot 1: Gain
    axes[0].barh(range(len(top_features)), top_features['gain_pct'], color='#3b82f6')
    axes[0].set_yticks(range(len(top_features)))
    axes[0].set_yticklabels(top_features['feature'])
    axes[0].set_xlabel('Importance (%)', fontsize=11)
    axes[0].set_title('Gain (Information Gain)', fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Plot 2: Weight
    axes[1].barh(range(len(top_features)), top_features['weight_pct'], color='#10b981')
    axes[1].set_yticks(range(len(top_features)))
    axes[1].set_yticklabels(top_features['feature'])
    axes[1].set_xlabel('Importance (%)', fontsize=11)
    axes[1].set_title('Weight (# of Splits)', fontsize=12, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    # Plot 3: Cover
    axes[2].barh(range(len(top_features)), top_features['cover_pct'], color='#f59e0b')
    axes[2].set_yticks(range(len(top_features)))
    axes[2].set_yticklabels(top_features['feature'])
    axes[2].set_xlabel('Importance (%)', fontsize=11)
    axes[2].set_title('Cover (Sample Coverage)', fontsize=12, fontweight='bold')
    axes[2].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved plot to {output_file}")
    else:
        plt.show()
    
    plt.close()


def compare_models(model1_df, model2_df, model1_name, model2_name, top_n=15, output_file=None):
    """Compare feature importance between two models.
    
    Args:
        model1_df: Feature importance dataframe for model 1
        model2_df: Feature importance dataframe for model 2
        model1_name: Name of model 1
        model2_name: Name of model 2
        top_n: Number of top features to compare
        output_file: Path to save plot
    """
    # Get union of top features from both models
    top_features_1 = set(model1_df.head(top_n)['feature'])
    top_features_2 = set(model2_df.head(top_n)['feature'])
    all_top_features = list(top_features_1 | top_features_2)
    
    # Create comparison dataframe
    comparison = pd.DataFrame({'feature': all_top_features})
    
    # Merge importance from both models
    comparison = comparison.merge(
        model1_df[['feature', 'gain_pct']].rename(columns={'gain_pct': f'{model1_name}_gain'}),
        on='feature',
        how='left'
    )
    comparison = comparison.merge(
        model2_df[['feature', 'gain_pct']].rename(columns={'gain_pct': f'{model2_name}_gain'}),
        on='feature',
        how='left'
    )
    
    # Fill NaN with 0 (feature not in that model)
    comparison = comparison.fillna(0)
    
    # Sort by sum of importances
    comparison['total_importance'] = comparison[f'{model1_name}_gain'] + comparison[f'{model2_name}_gain']
    comparison = comparison.sort_values('total_importance', ascending=True)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, max(8, len(comparison) * 0.4)))
    
    y_pos = np.arange(len(comparison))
    width = 0.35
    
    ax.barh(y_pos - width/2, comparison[f'{model1_name}_gain'], 
            width, label=model1_name, color='#3b82f6', alpha=0.8)
    ax.barh(y_pos + width/2, comparison[f'{model2_name}_gain'], 
            width, label=model2_name, color='#10b981', alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparison['feature'])
    ax.set_xlabel('Feature Importance (% Gain)', fontsize=12)
    ax.set_title(f'Feature Importance Comparison: {model1_name} vs {model2_name}', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved comparison plot to {output_file}")
    else:
        plt.show()
    
    plt.close()
    
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Analyze XGBoost feature importance",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model directory or .joblib file"
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Path to second model for comparison"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis/feature_importance",
        help="Output directory for results"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top features to display/plot"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("XGBOOST FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    print()
    
    # Load primary model
    model, config = load_model_and_config(args.model)
    model_name = Path(args.model).stem if Path(args.model).is_dir() else Path(args.model).parent.stem
    
    # Extract config parameters
    model_config = config.get('model', {})
    exclude_features = model_config.get('exclude_features', [])
    include_diff_features = model_config.get('include_diff_features', False)
    include_fav_underdog = model_config.get('include_fav_underdog_features', False)
    
    print(f"Model: {model_name}")
    print(f"  Excluded features: {len(exclude_features)}")
    if exclude_features:
        print(f"    {exclude_features}")
    print(f"  Include diff features: {include_diff_features}")
    print(f"  Include fav/underdog features: {include_fav_underdog}")
    print()
    
    # Get feature names
    feature_names = get_feature_names(
        exclude_features=exclude_features,
        include_diff_features=include_diff_features,
        include_fav_underdog=include_fav_underdog
    )
    print(f"Total features in model: {len(feature_names)}")
    print()
    
    # Extract importance
    print("Extracting feature importance...")
    importance_df = extract_xgboost_importance(model, feature_names)
    
    # Display results
    print()
    print("=" * 80)
    print(f"TOP {args.top_n} FEATURES BY GAIN")
    print("=" * 80)
    print()
    
    top_features = importance_df.head(args.top_n)
    
    # Format for display
    display_df = top_features[['feature', 'gain_pct', 'weight', 'cover']].copy()
    display_df.columns = ['Feature', 'Gain %', 'Weight', 'Cover']
    print(display_df.to_string(index=False))
    
    # Check for FTR and role features
    print()
    print("=" * 80)
    print("EXPERIMENTAL FEATURES ANALYSIS")
    print("=" * 80)
    print()
    
    ftr_features = importance_df[importance_df['feature'].str.contains('ftr', case=False)]
    if len(ftr_features) > 0:
        print("Free Throw Rate (FTR) Features:")
        print(ftr_features[['feature', 'gain_pct', 'weight']].to_string(index=False))
        print()
    else:
        print("No FTR features in this model")
        print()
    
    role_features = importance_df[importance_df['feature'].str.contains('favorite|underdog', case=False)]
    if len(role_features) > 0:
        print("Favorite/Underdog Role Features:")
        print(role_features[['feature', 'gain_pct', 'weight']].to_string(index=False))
        print()
    else:
        print("No favorite/underdog role features in this model")
        print()
    
    # Save results
    csv_file = output_dir / f"{model_name}_feature_importance.csv"
    importance_df.to_csv(csv_file, index=False)
    print(f"Saved full results to: {csv_file}")
    print()
    
    # Generate plot
    if not args.no_plot:
        plot_file = output_dir / f"{model_name}_feature_importance.png"
        print("Generating importance plot...")
        plot_importance(importance_df, top_n=args.top_n, output_file=plot_file)
        print()
    
    # Compare with second model if provided
    if args.compare:
        print("=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        print()
        
        # Load comparison model
        model2, config2 = load_model_and_config(args.compare)
        model2_name = Path(args.compare).stem if Path(args.compare).is_dir() else Path(args.compare).parent.stem
        
        # Extract config parameters for model 2
        model2_config = config2.get('model', {})
        exclude_features2 = model2_config.get('exclude_features', [])
        include_diff_features2 = model2_config.get('include_diff_features', False)
        include_fav_underdog2 = model2_config.get('include_fav_underdog_features', False)
        
        print(f"Comparison Model: {model2_name}")
        print(f"  Excluded features: {len(exclude_features2)}")
        print(f"  Include diff features: {include_diff_features2}")
        print(f"  Include fav/underdog features: {include_fav_underdog2}")
        print()
        
        # Get feature names for model 2
        feature_names2 = get_feature_names(
            exclude_features=exclude_features2,
            include_diff_features=include_diff_features2,
            include_fav_underdog=include_fav_underdog2
        )
        
        # Extract importance for model 2
        print("Extracting feature importance for comparison model...")
        importance_df2 = extract_xgboost_importance(model2, feature_names2)
        
        # Save model 2 results
        csv_file2 = output_dir / f"{model2_name}_feature_importance.csv"
        importance_df2.to_csv(csv_file2, index=False)
        print(f"Saved comparison model results to: {csv_file2}")
        print()
        
        # Generate comparison plot
        if not args.no_plot:
            comparison_plot = output_dir / f"{model_name}_vs_{model2_name}_comparison.png"
            print("Generating comparison plot...")
            comparison_df = compare_models(
                importance_df, importance_df2,
                model_name, model2_name,
                top_n=args.top_n,
                output_file=comparison_plot
            )
            
            # Save comparison data
            comparison_csv = output_dir / f"{model_name}_vs_{model2_name}_comparison.csv"
            comparison_df.to_csv(comparison_csv, index=False)
            print(f"Saved comparison data to: {comparison_csv}")
            print()
    
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()

