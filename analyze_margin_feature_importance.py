"""
Feature importance analysis for the direct margin prediction model.

Analyzes:
1. Ridge coefficient magnitudes for mu (mean) and sigma (variance) models
2. Permutation importance on ATS accuracy
3. Ablation study: what happens if we drop low-importance features?
"""

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error

from data.loaders import GameDataLoader
from training.margin_dataset import MarginTrainingDataset
from features.builder import HOME_FEATURE_KEYS, AWAY_FEATURE_KEYS, SHARED_FEATURE_KEYS


def get_feature_names():
    """Get feature names in the order they appear in the combined feature vector."""
    # Combined format: [home_features, away_features, shared_features, home-away diffs]
    names = []
    
    # Home features
    names.extend([f"home_{k}" for k in HOME_FEATURE_KEYS])
    
    # Away features
    names.extend([f"away_{k}" for k in AWAY_FEATURE_KEYS])
    
    # Shared features
    names.extend([f"shared_{k}" for k in SHARED_FEATURE_KEYS])
    
    # Difference features (only for features that exist on both sides)
    for k in HOME_FEATURE_KEYS:
        if k in AWAY_FEATURE_KEYS:
            names.append(f"diff_{k}")
    
    return names


def analyze_coefficients(model, feature_names):
    """Analyze Ridge model coefficients."""
    # Get coefficients from both models
    mu_coef = model.model_mean.coef_
    sigma_coef = model.model_variance.coef_
    
    # Create dataframe
    df = pd.DataFrame({
        'feature': feature_names,
        'mu_coef': mu_coef,
        'mu_abs_coef': np.abs(mu_coef),
        'sigma_coef': sigma_coef,
        'sigma_abs_coef': np.abs(sigma_coef),
        'combined_importance': np.abs(mu_coef) + np.abs(sigma_coef)
    })
    
    # Sort by combined importance
    df = df.sort_values('combined_importance', ascending=False)
    
    return df


def permutation_importance(model, X, y_margin, y_sigma, spreads, feature_names, n_permutations=5, seed=42):
    """
    Compute permutation importance by shuffling each feature and measuring ATS degradation.
    """
    rng = np.random.default_rng(seed)
    
    # Baseline performance
    mu_pred, sigma_pred = model.predict(X, None)
    baseline_ats = compute_ats_accuracy(mu_pred, y_margin, spreads)
    
    importances = []
    
    print("\nRunning permutation importance (this may take a minute)...")
    for i, name in enumerate(feature_names):
        deltas = []
        for _ in range(n_permutations):
            # Permute this feature
            X_perm = X.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            
            # Re-predict
            mu_pred_perm, sigma_pred_perm = model.predict(X_perm, None)
            ats_perm = compute_ats_accuracy(mu_pred_perm, y_margin, spreads)
            
            # Importance = how much ATS degrades when feature is shuffled
            deltas.append(baseline_ats - ats_perm)
        
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)
        
        importances.append({
            'feature': name,
            'importance': mean_delta,
            'importance_std': std_delta
        })
        
        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/{len(feature_names)} features...")
    
    df = pd.DataFrame(importances)
    df = df.sort_values('importance', ascending=False)
    df['baseline_ats'] = baseline_ats
    
    return df


def compute_ats_accuracy(mu_pred, actual_margin, spreads):
    """Compute ATS accuracy given predictions and actuals."""
    # Home covers if actual_margin > -spread
    actual_home_covers = (actual_margin > -spreads).astype(int)
    
    # Model picks home if mu > -spread
    pred_home_covers = (mu_pred > -spreads).astype(int)
    
    # ATS accuracy
    ats_correct = (pred_home_covers == actual_home_covers).astype(int)
    return ats_correct.mean()


def ablation_study(model, X, y_margin, y_sigma, spreads, feature_names, coef_df, top_n_list=[5, 10, 15, 20]):
    """
    Test performance using only top N features by importance.
    """
    print("\nRunning ablation study (retraining with feature subsets)...")
    
    results = []
    
    # Baseline (all features)
    mu_pred, sigma_pred = model.predict(X, None)
    baseline_ats = compute_ats_accuracy(mu_pred, y_margin, spreads)
    baseline_mae = mean_absolute_error(y_margin, mu_pred)
    
    results.append({
        'n_features': len(feature_names),
        'features': 'all',
        'ats_accuracy': baseline_ats,
        'mae': baseline_mae
    })
    
    # Test using only top N features
    for top_n in top_n_list:
        if top_n >= len(feature_names):
            continue
            
        # Get indices of top N features
        top_features = coef_df.head(top_n)
        top_indices = [feature_names.index(f) for f in top_features['feature']]
        
        # Train new model with only these features
        X_subset = X[:, top_indices]
        
        # Retrain
        model_subset_mu = RidgeCV(alphas=model.config.alphas_mean, cv=5)
        model_subset_sigma = RidgeCV(alphas=model.config.alphas_variance, cv=5)
        
        model_subset_mu.fit(X_subset, y_margin)
        model_subset_sigma.fit(X_subset, np.abs(y_margin - model_subset_mu.predict(X_subset)))
        
        # Predict
        mu_pred_subset = model_subset_mu.predict(X_subset)
        ats_subset = compute_ats_accuracy(mu_pred_subset, y_margin, spreads)
        mae_subset = mean_absolute_error(y_margin, mu_pred_subset)
        
        results.append({
            'n_features': top_n,
            'features': f"top_{top_n}",
            'ats_accuracy': ats_subset,
            'mae': mae_subset
        })
        
        print(f"  Top {top_n} features: ATS={ats_subset:.4f}, MAE={mae_subset:.2f}")
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Analyze feature importance for margin model")
    parser.add_argument("--model", type=str, default="artifacts/margin_normalized/margin_model.joblib",
                       help="Path to trained model")
    parser.add_argument("--data", type=str, default="data/games_train_with_players_90_norm.jsonl",
                       help="Training data for analysis")
    parser.add_argument("--output", type=str, default="analysis/margin_feature_importance",
                       help="Output directory")
    parser.add_argument("--skip-permutation", action="store_true",
                       help="Skip permutation importance (slow)")
    parser.add_argument("--skip-ablation", action="store_true",
                       help="Skip ablation study (slow)")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("MARGIN MODEL FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    
    # Load model
    print(f"\nLoading model from {args.model}...")
    model = joblib.load(args.model)
    
    # Load data
    print(f"Loading data from {args.data}...")
    data_path = Path(args.data)
    loader = GameDataLoader(data_path.parent)
    collection = loader.read_games(data_path.name)
    records = collection.games
    print(f"  Loaded {len(records)} games")
    
    # Load config to get excluded features
    import yaml
    config_path = Path(args.model).parent / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        exclude_features = cfg.get('model', {}).get('exclude_features', [])
    else:
        exclude_features = []
    
    if exclude_features:
        print(f"Excluding {len(exclude_features)} features: {exclude_features}")
    
    # Build dataset with same exclusions as model
    print("Building dataset...")
    dataset = MarginTrainingDataset(records, exclude_features=exclude_features)
    batch = dataset.build()
    
    X = batch.x
    y_margin = batch.y_margin
    spreads = batch.market_spread_home
    
    # Get actual sigma (for ablation study)
    mu_pred, sigma_pred = model.predict(X, batch.baseline_margin)
    y_sigma = np.abs(y_margin - mu_pred)
    
    # Get feature names (excluding the ones that were excluded)
    all_feature_names = get_feature_names()
    feature_names = [name for name in all_feature_names if name not in exclude_features]
    print(f"Total features: {len(feature_names)}")
    
    # 1. Coefficient analysis
    print("\n" + "=" * 80)
    print("1. COEFFICIENT-BASED IMPORTANCE")
    print("=" * 80)
    
    coef_df = analyze_coefficients(model, feature_names)
    
    print("\nTop 20 features by combined importance (|mu_coef| + |sigma_coef|):")
    print(coef_df.head(20).to_string(index=False))
    
    print("\nBottom 10 features (potentially adding noise):")
    print(coef_df.tail(10).to_string(index=False))
    
    coef_df.to_csv(output_dir / "coefficient_importance.csv", index=False)
    print(f"\nSaved: {output_dir / 'coefficient_importance.csv'}")
    
    # 2. Permutation importance
    if not args.skip_permutation:
        print("\n" + "=" * 80)
        print("2. PERMUTATION IMPORTANCE")
        print("=" * 80)
        
        perm_df = permutation_importance(model, X, y_margin, y_sigma, spreads, feature_names)
        
        print("\nTop 15 features by permutation importance:")
        print(perm_df.head(15).to_string(index=False))
        
        print("\nFeatures with NEGATIVE importance (hurt performance when included):")
        negative = perm_df[perm_df['importance'] < -0.001]
        if len(negative) > 0:
            print(negative.to_string(index=False))
        else:
            print("  None found")
        
        perm_df.to_csv(output_dir / "permutation_importance.csv", index=False)
        print(f"\nSaved: {output_dir / 'permutation_importance.csv'}")
    
    # 3. Ablation study
    if not args.skip_ablation:
        print("\n" + "=" * 80)
        print("3. ABLATION STUDY")
        print("=" * 80)
        
        ablation_df = ablation_study(model, X, y_margin, y_sigma, spreads, 
                                     feature_names, coef_df, 
                                     top_n_list=[5, 10, 15, 20, 25])
        
        print("\nPerformance with different feature subsets:")
        print(ablation_df.to_string(index=False))
        
        ablation_df.to_csv(output_dir / "ablation_study.csv", index=False)
        print(f"\nSaved: {output_dir / 'ablation_study.csv'}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()

