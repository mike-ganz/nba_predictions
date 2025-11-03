"""Comprehensive feature analysis for spread coverage prediction model.

Analyzes:
1. Feature correlations (redundancy)
2. Permutation importance (which features hurt performance)
3. VIF (multicollinearity)
4. Feature ablation (drop one at a time)
5. Coefficient stability

Usage:
    python analyze_feature_redundancy.py
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.metrics import log_loss, accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

from data.loaders import GameDataLoader
from training.margin_dataset import MarginTrainingDataset
from models.margin_normal import MarginNormalModel, MarginNormalConfig


def load_data():
    """Load training data."""
    print("Loading training data...")
    data_path = Path("data/games_train_with_players_90_norm.jsonl")
    loader = GameDataLoader(data_path.parent)
    collection = loader.read_games(data_path.name)
    
    # Load config for exclude_features
    with open("configs/margin_default.yaml") as f:
        cfg = yaml.safe_load(f)
    exclude_features = cfg.get('model', {}).get('exclude_features', [])
    
    dataset = MarginTrainingDataset(collection.games, exclude_features=exclude_features)
    batch = dataset.build()
    
    # Filter to valid games
    valid_mask = ~np.isnan(batch.y_home_covers)
    X = batch.x[valid_mask]
    y = batch.y_home_covers[valid_mask].astype(int)
    
    print(f"Loaded {len(y)} games with {X.shape[1]} features")
    return X, y, exclude_features


def get_feature_names():
    """Get feature names in order."""
    from features.builder import HOME_FEATURE_KEYS, AWAY_FEATURE_KEYS, SHARED_FEATURE_KEYS
    
    feature_names = []
    
    # Home features
    for k in HOME_FEATURE_KEYS:
        feature_names.append(f"home_{k}")
    
    # Away features
    for k in AWAY_FEATURE_KEYS:
        feature_names.append(f"away_{k}")
    
    # Shared features
    for k in SHARED_FEATURE_KEYS:
        feature_names.append(f"shared_{k}")
    
    # Difference features
    for k in HOME_FEATURE_KEYS:
        if k in AWAY_FEATURE_KEYS:
            feature_names.append(f"diff_{k}")
    
    return feature_names


def analyze_correlations(X, feature_names, output_dir):
    """Analyze feature correlations to find redundancy."""
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)
    
    df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df.corr()
    
    # Find highly correlated pairs (excluding diagonal)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:  # Threshold for high correlation
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    if high_corr_pairs:
        print(f"\nFound {len(high_corr_pairs)} highly correlated pairs (|r| > 0.8):")
        for pair in sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True):
            print(f"  {pair['feature1']:<30} <-> {pair['feature2']:<30} r={pair['correlation']:+.3f}")
    else:
        print("\nNo highly correlated pairs found (|r| > 0.8)")
    
    # Save correlation matrix
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, vmin=-1, vmax=1, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nCorrelation matrix saved to {output_dir / 'correlation_matrix.png'}")
    
    return high_corr_pairs


def analyze_vif(X, feature_names):
    """Calculate Variance Inflation Factor for multicollinearity."""
    print("\n" + "="*70)
    print("MULTICOLLINEARITY ANALYSIS (VIF)")
    print("="*70)
    print("VIF > 10 indicates high multicollinearity")
    print("VIF > 5 is concerning\n")
    
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # Standardize features first
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    vif_data = []
    for i in range(X_scaled.shape[1]):
        try:
            vif = variance_inflation_factor(X_scaled, i)
            vif_data.append({
                'feature': feature_names[i],
                'VIF': vif
            })
        except:
            vif_data.append({
                'feature': feature_names[i],
                'VIF': np.nan
            })
    
    vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
    
    # Show high VIF features
    high_vif = vif_df[vif_df['VIF'] > 5]
    if len(high_vif) > 0:
        print("Features with high VIF (>5):")
        for _, row in high_vif.iterrows():
            status = "[SEVERE]" if row['VIF'] > 10 else "[MODERATE]"
            print(f"  {status} {row['feature']:<30} VIF={row['VIF']:.2f}")
    else:
        print("[OK] No features with high VIF (all < 5)")
    
    return vif_df


def train_baseline_model(X, y):
    """Train baseline model and get performance."""
    print("\n" + "="*70)
    print("BASELINE MODEL PERFORMANCE")
    print("="*70)
    
    with open("configs/margin_default.yaml") as f:
        cfg = yaml.safe_load(f)
    
    model_cfg = {k: v for k, v in cfg.get('model', {}).items() if k != 'exclude_features'}
    model_config = MarginNormalConfig(**model_cfg)
    model = MarginNormalModel(model_config)
    
    # Cross-validation
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(model.model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Train on full data for subsequent analysis
    model.fit(X, y)
    y_pred_proba = model.predict(X)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    accuracy = accuracy_score(y, y_pred)
    logloss = log_loss(y, y_pred_proba)
    
    print(f"Training Accuracy: {accuracy:.4f}")
    print(f"Training Log Loss: {logloss:.4f}")
    
    return model, accuracy, logloss


def analyze_permutation_importance(model, X, y, feature_names):
    """Permutation importance to find harmful features."""
    print("\n" + "="*70)
    print("PERMUTATION IMPORTANCE ANALYSIS")
    print("="*70)
    print("Negative importance = feature hurts performance!\n")
    
    # Calculate permutation importance
    result = permutation_importance(
        model.model, 
        model.scaler.transform(X) if model.scaler else X,
        y,
        n_repeats=10,
        random_state=42,
        scoring='accuracy',
        n_jobs=-1
    )
    
    perm_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=True)
    
    # Show harmful features (negative importance)
    harmful = perm_imp_df[perm_imp_df['importance_mean'] < 0]
    if len(harmful) > 0:
        print("[HARMFUL] FEATURES (negative importance):")
        for _, row in harmful.iterrows():
            print(f"  {row['feature']:<30} {row['importance_mean']:+.6f} +/- {row['importance_std']:.6f}")
    else:
        print("[OK] No features with negative importance")
    
    # Show top helpful features
    print("\n[HELPFUL] TOP 10 HELPFUL FEATURES:")
    top_helpful = perm_imp_df.tail(10)
    for _, row in top_helpful.iterrows():
        print(f"  {row['feature']:<30} {row['importance_mean']:+.6f} +/- {row['importance_std']:.6f}")
    
    return perm_imp_df


def analyze_feature_ablation(X, y, feature_names, baseline_accuracy):
    """Drop each feature one at a time and measure impact."""
    print("\n" + "="*70)
    print("FEATURE ABLATION ANALYSIS")
    print("="*70)
    print("Dropping each feature individually to see impact\n")
    
    with open("configs/margin_default.yaml") as f:
        cfg = yaml.safe_load(f)
    
    ablation_results = []
    
    for i, feat_name in enumerate(feature_names):
        # Create dataset without this feature
        X_ablated = np.delete(X, i, axis=1)
        
        # Train model
        model_cfg = {k: v for k, v in cfg.get('model', {}).items() if k != 'exclude_features'}
        model_config = MarginNormalConfig(**model_cfg)
        model = MarginNormalModel(model_config)
        
        # Quick CV (3-fold for speed)
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(model.model, X_ablated, y, cv=3, scoring='accuracy', n_jobs=-1)
        
        accuracy_without = cv_scores.mean()
        impact = accuracy_without - baseline_accuracy
        
        ablation_results.append({
            'feature': feat_name,
            'accuracy_without': accuracy_without,
            'impact': impact
        })
        
        status = "[BAD]" if impact > 0.001 else "[WARN]" if impact > 0 else "[OK]"
        print(f"  {status} Without {feat_name:<30} Acc={accuracy_without:.4f} (Delta={impact:+.4f})")
    
    ablation_df = pd.DataFrame(ablation_results).sort_values('impact', ascending=False)
    
    # Summary
    print("\n" + "="*70)
    print("ABLATION SUMMARY")
    print("="*70)
    harmful_ablation = ablation_df[ablation_df['impact'] > 0.001]
    if len(harmful_ablation) > 0:
        print(f"\n[HARMFUL] FEATURES (accuracy improves >0.1% when dropped):")
        for _, row in harmful_ablation.iterrows():
            print(f"  {row['feature']:<30} Delta={row['impact']:+.4f}")
    else:
        print("\n[OK] No features improve accuracy when dropped")
    
    return ablation_df


def analyze_coefficients(model, feature_names):
    """Analyze model coefficients."""
    print("\n" + "="*70)
    print("COEFFICIENT ANALYSIS")
    print("="*70)
    
    coeffs = model.get_coefficients()
    
    coeff_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coeffs,
        'abs_coefficient': np.abs(coeffs)
    }).sort_values('abs_coefficient', ascending=False)
    
    print("\nTop 15 features by coefficient magnitude:")
    for _, row in coeff_df.head(15).iterrows():
        direction = "+" if row['coefficient'] > 0 else "-"
        print(f"  {row['feature']:<30} {row['coefficient']:+.6f} {direction}")
    
    # Near-zero coefficients (regularized away)
    near_zero = coeff_df[coeff_df['abs_coefficient'] < 0.001]
    print(f"\nFeatures with near-zero coefficients (<0.001): {len(near_zero)}")
    if len(near_zero) > 0:
        print("These features are mostly ignored by the model:")
        for _, row in near_zero.iterrows():
            print(f"  {row['feature']:<30} {row['coefficient']:+.6f}")
    
    return coeff_df


def generate_recommendations(high_corr_pairs, vif_df, perm_imp_df, ablation_df, coeff_df):
    """Generate actionable recommendations."""
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    recommendations = []
    
    # 1. Harmful features from permutation importance
    harmful_perm = perm_imp_df[perm_imp_df['importance_mean'] < -0.0001]['feature'].tolist()
    
    # 2. Harmful features from ablation
    harmful_ablation = ablation_df[ablation_df['impact'] > 0.001]['feature'].tolist()
    
    # 3. High VIF features
    high_vif_features = vif_df[vif_df['VIF'] > 10]['feature'].tolist() if 'VIF' in vif_df.columns else []
    
    # 4. Near-zero coefficients
    zero_coeff = coeff_df[coeff_df['abs_coefficient'] < 0.001]['feature'].tolist()
    
    # Combine recommendations
    definitely_remove = set(harmful_perm) & set(harmful_ablation)
    probably_remove = (set(harmful_perm) | set(harmful_ablation)) - definitely_remove
    consider_remove = set(high_vif_features) | set(zero_coeff)
    
    print("\n[DEFINITELY REMOVE] (harmful in both tests):")
    if definitely_remove:
        for feat in definitely_remove:
            print(f"  - {feat}")
            recommendations.append(feat)
    else:
        print("  None identified")
    
    print("\n[PROBABLY REMOVE] (harmful in one test):")
    if probably_remove:
        for feat in probably_remove:
            print(f"  - {feat}")
            recommendations.append(feat)
    else:
        print("  None identified")
    
    print("\n[CONSIDER REMOVING] (high VIF or zero coefficient):")
    consider_list = list(consider_remove - definitely_remove - probably_remove)[:10]
    if consider_list:
        for feat in consider_list:
            reason = []
            if feat in high_vif_features:
                vif_val = vif_df[vif_df['feature'] == feat]['VIF'].values[0]
                reason.append(f"VIF={vif_val:.1f}")
            if feat in zero_coeff:
                reason.append("near-zero coeff")
            print(f"  - {feat:<30} ({', '.join(reason)})")
    else:
        print("  None identified")
    
    # Redundant pairs
    if high_corr_pairs:
        print("\n[REDUNDANT PAIRS] (keep one from each pair):")
        for pair in high_corr_pairs[:5]:  # Top 5
            print(f"  - {pair['feature1']} <-> {pair['feature2']} (r={pair['correlation']:.3f})")
    
    return recommendations


def main():
    """Run comprehensive feature analysis."""
    output_dir = Path("analysis/feature_redundancy")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X, y, current_excludes = load_data()
    feature_names = get_feature_names()
    
    # Remove already excluded features from analysis
    if current_excludes:
        print(f"\nCurrently excluded features: {current_excludes}")
        keep_indices = [i for i, name in enumerate(feature_names) if name not in current_excludes]
        feature_names = [feature_names[i] for i in keep_indices]
    
    # Run analyses
    high_corr_pairs = analyze_correlations(X, feature_names, output_dir)
    vif_df = analyze_vif(X, feature_names)
    model, baseline_acc, baseline_loss = train_baseline_model(X, y)
    perm_imp_df = analyze_permutation_importance(model, X, y, feature_names)
    ablation_df = analyze_feature_ablation(X, y, feature_names, baseline_acc)
    coeff_df = analyze_coefficients(model, feature_names)
    
    # Save results
    vif_df.to_csv(output_dir / 'vif_analysis.csv', index=False)
    perm_imp_df.to_csv(output_dir / 'permutation_importance.csv', index=False)
    ablation_df.to_csv(output_dir / 'ablation_analysis.csv', index=False)
    coeff_df.to_csv(output_dir / 'coefficients.csv', index=False)
    
    print(f"\n\nDetailed results saved to {output_dir}/")
    
    # Generate recommendations
    recommendations = generate_recommendations(
        high_corr_pairs, vif_df, perm_imp_df, ablation_df, coeff_df
    )
    
    # Save recommendations
    with open(output_dir / 'recommendations.txt', 'w') as f:
        f.write("FEATURES TO EXCLUDE\n")
        f.write("="*70 + "\n\n")
        f.write("Add these to configs/margin_default.yaml under exclude_features:\n\n")
        for feat in recommendations:
            f.write(f"  - {feat}\n")
    
    print(f"\nRecommendations saved to {output_dir / 'recommendations.txt'}")
    print("\n[COMPLETE] Analysis complete!")


if __name__ == "__main__":
    main()

