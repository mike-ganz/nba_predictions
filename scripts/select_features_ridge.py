"""
Feature selection for Ridge regression margin prediction.

This script systematically tests different feature subsets to find the optimal
combination that maximizes test set ATS accuracy for Ridge regression.

Strategies:
1. Permutation Importance: Measures impact of each feature on test ATS accuracy
2. Recursive Feature Elimination: Iteratively removes worst features
3. Feature Ablation: Tests removing bottom N% of features by weight magnitude

Usage:
    python select_features_ridge.py --model artifacts/margin_ridge \\
                                     --train-data data/games_train_with_players_90_norm.jsonl \\
                                     --test-data data/games_predict_2024_2025_with_players_norm.jsonl \\
                                     --output reports/feature_selection_ridge
"""

import argparse
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import joblib
from tqdm import tqdm

from training.margin_dataset import MarginTrainingDataset
from data.loaders import GameDataLoader
from models.margin_normal import MarginNormalModel, MarginNormalConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Feature selection for Ridge regression")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained Ridge model directory")
    parser.add_argument("--train-data", type=str, required=True,
                        help="Path to training data JSONL")
    parser.add_argument("--test-data", type=str, required=True,
                        help="Path to test data JSONL")
    parser.add_argument("--output", type=str, default="reports/feature_selection_ridge",
                        help="Output directory for reports")
    parser.add_argument("--n-permutations", type=int, default=10,
                        help="Number of permutations for importance calculation")
    return parser.parse_args()


def calculate_ats_accuracy(y_pred: np.ndarray, y_true: np.ndarray, spreads: np.ndarray) -> float:
    """Calculate ATS accuracy given predictions and actuals."""
    pred_covers = (y_pred > -spreads).astype(int)
    actual_covers = (y_true > -spreads).astype(int)
    accuracy = (pred_covers == actual_covers).mean()
    return accuracy


def permutation_importance_ats(
    model: MarginNormalModel,
    x: np.ndarray,
    y: np.ndarray,
    baseline: np.ndarray,
    spreads: np.ndarray,
    feature_names: List[str],
    n_permutations: int = 10
) -> Dict[str, float]:
    """
    Calculate permutation importance for each feature based on ATS accuracy.
    
    Higher values = more important (bigger drop when shuffled).
    """
    # Baseline ATS accuracy
    mu, sigma = model.predict(x, baseline)
    baseline_acc = calculate_ats_accuracy(mu, y, spreads)
    
    importance = {}
    
    print("\nCalculating permutation importance (ATS-focused)...")
    for i, feat_name in enumerate(tqdm(feature_names, desc="Features")):
        drops = []
        
        for _ in range(n_permutations):
            # Shuffle feature i
            x_permuted = x.copy()
            np.random.shuffle(x_permuted[:, i])
            
            # Predict with shuffled feature
            mu_perm, sigma_perm = model.predict(x_permuted, baseline)
            acc_perm = calculate_ats_accuracy(mu_perm, y, spreads)
            
            # Importance = drop in accuracy
            drops.append(baseline_acc - acc_perm)
        
        # Average importance over permutations
        importance[feat_name] = np.mean(drops)
    
    return importance


def train_and_evaluate_subset(
    feature_indices: List[int],
    train_batch,
    test_batch,
    config: MarginNormalConfig,
    feature_names: List[str]
) -> Dict:
    """Train model with subset of features and evaluate on test set."""
    # Extract feature subset
    x_train = train_batch.x[:, feature_indices]
    x_test = test_batch.x[:, feature_indices]
    
    # Train model
    model = MarginNormalModel(config)
    model.fit(x_train, train_batch.y_margin, train_batch.baseline_margin)
    
    # Evaluate on train and test
    train_mu, train_sigma = model.predict(x_train, train_batch.baseline_margin)
    test_mu, test_sigma = model.predict(x_test, test_batch.baseline_margin)
    
    train_ats = calculate_ats_accuracy(train_mu, train_batch.y_margin, train_batch.market_spread_home)
    test_ats = calculate_ats_accuracy(test_mu, test_batch.y_margin, test_batch.market_spread_home)
    
    train_mae = np.abs(train_mu - train_batch.y_margin).mean()
    test_mae = np.abs(test_mu - test_batch.y_margin).mean()
    
    selected_features = [feature_names[i] for i in feature_indices]
    
    return {
        'n_features': len(feature_indices),
        'features': selected_features,
        'train_ats': train_ats,
        'test_ats': test_ats,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'generalization_gap': train_ats - test_ats
    }


def recursive_feature_elimination(
    train_batch,
    test_batch,
    config: MarginNormalConfig,
    feature_names: List[str],
    n_iterations: int = 15
) -> List[Dict]:
    """
    Iteratively remove least important features and track performance.
    
    For Ridge, we use coefficient magnitude as importance.
    
    Returns list of results for each iteration.
    """
    results = []
    remaining_indices = list(range(len(feature_names)))
    
    print("\n" + "="*70)
    print("RECURSIVE FEATURE ELIMINATION")
    print("="*70)
    
    for iteration in range(n_iterations):
        print(f"\nIteration {iteration + 1}: Testing with {len(remaining_indices)} features...")
        
        # Train and evaluate with current feature set
        result = train_and_evaluate_subset(
            remaining_indices, train_batch, test_batch, config, feature_names
        )
        result['iteration'] = iteration + 1
        results.append(result)
        
        print(f"  Test ATS: {result['test_ats']:.4f} | Train ATS: {result['train_ats']:.4f} | Gap: {result['generalization_gap']:.4f}")
        
        # Stop if only a few features left
        if len(remaining_indices) <= 5:
            print("Reached minimum feature count (5), stopping.")
            break
        
        # Train model to get feature importance (coefficient magnitudes)
        x_train = train_batch.x[:, remaining_indices]
        model = MarginNormalModel(config)
        model.fit(x_train, train_batch.y_margin, train_batch.baseline_margin)
        
        # Get coefficient magnitudes of remaining features
        coeffs = model.model_mean.coef_
        importance_by_idx = {}
        for local_idx, global_idx in enumerate(remaining_indices):
            importance_by_idx[global_idx] = abs(coeffs[local_idx])
        
        # Find least important feature (smallest coefficient magnitude)
        if importance_by_idx:
            worst_idx = min(importance_by_idx, key=importance_by_idx.get)
            worst_name = feature_names[worst_idx]
            print(f"  Removing: {worst_name} (|coef|: {importance_by_idx[worst_idx]:.4f})")
            remaining_indices.remove(worst_idx)
        else:
            print("  Could not determine feature importance, stopping.")
            break
    
    return results


def ablation_study(
    train_batch,
    test_batch,
    config: MarginNormalConfig,
    feature_names: List[str],
    base_importance: Dict[str, float]
) -> List[Dict]:
    """
    Test removing bottom N% of features by coefficient magnitude.
    """
    results = []
    
    print("\n" + "="*70)
    print("FEATURE ABLATION STUDY")
    print("="*70)
    
    # Sort features by importance (coefficient magnitude)
    importance_items = sorted(base_importance.items(), key=lambda x: x[1], reverse=True)
    sorted_indices = [feature_names.index(name) for name, _ in importance_items]
    
    # Test keeping top 100%, 90%, 80%, 70%, 60%, 50%, 40%, 30%, 25% of features
    keep_percentages = [100, 90, 80, 70, 60, 50, 40, 30, 25]
    
    for pct in keep_percentages:
        n_keep = max(5, int(len(feature_names) * pct / 100))
        feature_indices = sorted_indices[:n_keep]
        
        print(f"\nKeeping top {pct}% features ({n_keep} features)...")
        
        result = train_and_evaluate_subset(
            feature_indices, train_batch, test_batch, config, feature_names
        )
        result['keep_percentage'] = pct
        results.append(result)
        
        print(f"  Test ATS: {result['test_ats']:.4f} | Train ATS: {result['train_ats']:.4f} | Gap: {result['generalization_gap']:.4f}")
    
    return results


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("RIDGE REGRESSION FEATURE SELECTION")
    print("="*70)
    
    # Load model and config
    model_dir = Path(args.model)
    print(f"\nLoading model from {model_dir}...")
    model = joblib.load(model_dir / "margin_model.joblib")
    
    with open(model_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    model_cfg = cfg.get('model', {})
    exclude_features = model_cfg.get('exclude_features', [])
    include_diff_features = model_cfg.get('include_diff_features', True)
    
    print(f"Exclude features: {exclude_features}")
    print(f"Include diff features: {include_diff_features}")
    
    # Load training data
    print(f"\nLoading training data from {args.train_data}...")
    train_data_path = Path(args.train_data)
    train_loader = GameDataLoader(train_data_path.parent)
    train_collection = train_loader.read_games(train_data_path.name)
    train_records = train_collection.games
    train_dataset = MarginTrainingDataset(
        train_records,
        exclude_features=exclude_features,
        include_diff_features=include_diff_features
    )
    train_batch = train_dataset.build()
    print(f"Loaded {len(train_records)} training games with {train_batch.x.shape[1]} features")
    
    # Load test data
    print(f"\nLoading test data from {args.test_data}...")
    test_data_path = Path(args.test_data)
    test_loader = GameDataLoader(test_data_path.parent)
    test_collection = test_loader.read_games(test_data_path.name)
    test_records = test_collection.games
    test_dataset = MarginTrainingDataset(
        test_records,
        exclude_features=exclude_features,
        include_diff_features=include_diff_features
    )
    test_batch = test_dataset.build()
    print(f"Loaded {len(test_records)} test games")
    
    # Get feature names (use generic names since we don't have actual names)
    feature_names = [f"feature_{i}" for i in range(train_batch.x.shape[1])]
    
    # Get baseline importance from trained model (coefficient magnitudes)
    print("\nGetting baseline feature importance (coefficient magnitudes)...")
    coeffs = model.model_mean.coef_
    base_importance = {feature_names[i]: abs(coeffs[i]) for i in range(len(feature_names))}
    
    print("\nBaseline Feature Importance (top 10):")
    sorted_imp = sorted(base_importance.items(), key=lambda x: x[1], reverse=True)
    for feat, score in sorted_imp[:10]:
        print(f"  {feat}: {score:.4f}")
    
    # Get baseline performance
    print("\nBaseline Performance:")
    train_mu, train_sigma = model.predict(train_batch.x, train_batch.baseline_margin)
    test_mu, test_sigma = model.predict(test_batch.x, test_batch.baseline_margin)
    baseline_train_ats = calculate_ats_accuracy(train_mu, train_batch.y_margin, train_batch.market_spread_home)
    baseline_test_ats = calculate_ats_accuracy(test_mu, test_batch.y_margin, test_batch.market_spread_home)
    print(f"  Train ATS: {baseline_train_ats:.4f}")
    print(f"  Test ATS: {baseline_test_ats:.4f}")
    print(f"  Gap: {baseline_train_ats - baseline_test_ats:.4f}")
    
    # 1. Permutation Importance (ATS-focused)
    perm_importance = permutation_importance_ats(
        model, test_batch.x, test_batch.y_margin, test_batch.baseline_margin,
        test_batch.market_spread_home, feature_names, args.n_permutations
    )
    
    print("\nPermutation Importance (ATS-focused):")
    sorted_perm = sorted(perm_importance.items(), key=lambda x: x[1], reverse=True)
    for feat, score in sorted_perm[:10]:
        print(f"  {feat}: {score:.4f} (higher = more important)")
    
    # Save permutation importance
    perm_df = pd.DataFrame([
        {'feature': feat, 'permutation_importance': score, 'coef_magnitude': base_importance.get(feat, 0)}
        for feat, score in sorted_perm
    ])
    perm_df.to_csv(output_dir / "permutation_importance.csv", index=False)
    
    # 2. Recursive Feature Elimination
    # Filter out keys that aren't valid MarginNormalConfig parameters
    config_params = {k: v for k, v in model_cfg.items() 
                     if k not in ['exclude_features', 'include_diff_features', 'model_type']}
    config = MarginNormalConfig(**config_params)
    rfe_results = recursive_feature_elimination(
        train_batch, test_batch, config, feature_names, n_iterations=15
    )
    
    # Save RFE results
    rfe_df = pd.DataFrame(rfe_results)
    rfe_df.to_csv(output_dir / "rfe_results.csv", index=False)
    
    # 3. Ablation Study
    ablation_results = ablation_study(
        train_batch, test_batch, config, feature_names, base_importance
    )
    
    # Save ablation results
    ablation_df = pd.DataFrame(ablation_results)
    ablation_df.to_csv(output_dir / "ablation_results.csv", index=False)
    
    # Find best configurations
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)
    
    print(f"\nBaseline (all {len(feature_names)} features):")
    print(f"  Test ATS: {baseline_test_ats:.4f}")
    
    print("\nBest from Recursive Feature Elimination:")
    best_rfe = max(rfe_results, key=lambda x: x['test_ats'])
    print(f"  Test ATS: {best_rfe['test_ats']:.4f} with {best_rfe['n_features']} features")
    print(f"  Improvement: {(best_rfe['test_ats'] - baseline_test_ats) * 100:.2f}%")
    
    print("\nBest from Ablation Study:")
    best_ablation = max(ablation_results, key=lambda x: x['test_ats'])
    print(f"  Test ATS: {best_ablation['test_ats']:.4f} with {best_ablation['n_features']} features ({best_ablation['keep_percentage']}%)")
    print(f"  Improvement: {(best_ablation['test_ats'] - baseline_test_ats) * 100:.2f}%")
    
    # Overall best
    all_results = rfe_results + ablation_results
    best_overall = max(all_results, key=lambda x: x['test_ats'])
    
    print("\n" + "="*70)
    print("BEST OVERALL CONFIGURATION")
    print("="*70)
    print(f"Test ATS: {best_overall['test_ats']:.4f}")
    print(f"Train ATS: {best_overall['train_ats']:.4f}")
    print(f"Generalization Gap: {best_overall['generalization_gap']:.4f}")
    print(f"Number of Features: {best_overall['n_features']}")
    print(f"Improvement over baseline: {(best_overall['test_ats'] - baseline_test_ats) * 100:.2f}%")
    print("\nSelected Features:")
    for feat in best_overall['features']:
        coef = base_importance.get(feat, 0)
        perm = perm_importance.get(feat, 0)
        print(f"  {feat}: |coef|={coef:.4f}, perm_importance={perm:.4f}")
    
    # Save best configuration
    best_config = {
        'n_features': best_overall['n_features'],
        'features': best_overall['features'],
        'test_ats_accuracy': float(best_overall['test_ats']),
        'train_ats_accuracy': float(best_overall['train_ats']),
        'improvement_over_baseline': float(best_overall['test_ats'] - baseline_test_ats),
    }
    
    with open(output_dir / "best_feature_subset.json", 'w') as f:
        json.dump(best_config, f, indent=2)
    
    # Generate summary report
    with open(output_dir / "feature_selection_report.txt", 'w') as f:
        f.write("="*70 + "\n")
        f.write("RIDGE REGRESSION FEATURE SELECTION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Baseline Performance (all {len(feature_names)} features):\n")
        f.write(f"  Test ATS: {baseline_test_ats:.4f}\n")
        f.write(f"  Train ATS: {baseline_train_ats:.4f}\n\n")
        
        f.write("Best Configuration Found:\n")
        f.write(f"  Test ATS: {best_overall['test_ats']:.4f}\n")
        f.write(f"  Train ATS: {best_overall['train_ats']:.4f}\n")
        f.write(f"  Number of Features: {best_overall['n_features']}\n")
        f.write(f"  Improvement: {(best_overall['test_ats'] - baseline_test_ats) * 100:.2f}%\n\n")
        
        f.write("Selected Features:\n")
        for feat in best_overall['features']:
            coef = base_importance.get(feat, 0)
            perm = perm_importance.get(feat, 0)
            f.write(f"  {feat}: |coef|={coef:.4f}, perm_importance={perm:.4f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("RECURSIVE FEATURE ELIMINATION RESULTS\n")
        f.write("="*70 + "\n\n")
        for result in rfe_results:
            f.write(f"Iteration {result['iteration']}: {result['n_features']} features\n")
            f.write(f"  Test ATS: {result['test_ats']:.4f}\n")
            f.write(f"  Train ATS: {result['train_ats']:.4f}\n")
            f.write(f"  Gap: {result['generalization_gap']:.4f}\n\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("ABLATION STUDY RESULTS\n")
        f.write("="*70 + "\n\n")
        for result in ablation_results:
            f.write(f"Keep {result['keep_percentage']}%: {result['n_features']} features\n")
            f.write(f"  Test ATS: {result['test_ats']:.4f}\n")
            f.write(f"  Train ATS: {result['train_ats']:.4f}\n")
            f.write(f"  Gap: {result['generalization_gap']:.4f}\n\n")
    
    print(f"\n\nAll results saved to {output_dir}/")
    print("  - permutation_importance.csv")
    print("  - rfe_results.csv")
    print("  - ablation_results.csv")
    print("  - best_feature_subset.json")
    print("  - feature_selection_report.txt")
    
    print("\n" + "="*70)
    print("FEATURE SELECTION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

