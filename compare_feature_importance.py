"""
Compare feature importance between old and new Champion models.
Focus on injury features to validate reconstruction methodology.
"""
import joblib
import yaml
from pathlib import Path
import pandas as pd


def load_model_and_config(model_dir: str):
    """Load model and its configuration."""
    model_path = Path(model_dir)
    
    # Load model
    model = joblib.load(model_path / "margin_model.joblib")
    
    # Load config
    with open(model_path / "config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    return model, config


def get_feature_names(config):
    """
    Get feature names based on config.
    Based on MarginTrainingDataset logic.
    """
    # Home features
    HOME_FEATURE_KEYS = [
        "edge", "orb_edge", "tov_edge", "tpar", "ftr",
        "minutes_missing_top2", "star_out", "usage_share_top2"
    ]
    
    # Away features (same as home)
    AWAY_FEATURE_KEYS = HOME_FEATURE_KEYS.copy()
    
    # Shared features
    SHARED_FEATURE_KEYS = [
        "pace_mean", "pace_diff", 
        "implied_home_winprob", "implied_away_winprob",
        "team_weighted_ts_home", "team_weighted_ts_away"
    ]
    
    # Build feature list
    exclude_features = config.get('model', {}).get('exclude_features', [])
    
    all_features = []
    
    # Add home features
    for feat in HOME_FEATURE_KEYS:
        full_name = f"home_{feat}"
        if full_name not in exclude_features:
            all_features.append(full_name)
    
    # Add away features
    for feat in AWAY_FEATURE_KEYS:
        full_name = f"away_{feat}"
        if full_name not in exclude_features:
            all_features.append(full_name)
    
    # Add shared features
    for feat in SHARED_FEATURE_KEYS:
        full_name = f"shared_{feat}"
        if full_name not in exclude_features:
            all_features.append(full_name)
    
    return all_features


def analyze_model_importance(model_dir: str, model_name: str):
    """Analyze feature importance for a model."""
    print("="*80)
    print(f"MODEL: {model_name}")
    print(f"Path: {model_dir}")
    print("="*80)
    print()
    
    model, config = load_model_and_config(model_dir)
    feature_names = get_feature_names(config)
    
    # Get feature importance (gain)
    # The model is wrapped in MarginXGBoostModel, access the underlying XGBRegressor
    if hasattr(model, 'model'):
        xgb_model = model.model
    else:
        xgb_model = model
    
    importance = xgb_model.get_booster().get_score(importance_type='gain')
    
    # Map to feature names
    importance_dict = {}
    for i, feat_name in enumerate(feature_names):
        # XGBoost uses f0, f1, f2, ... format
        xgb_feat = f"f{i}"
        if xgb_feat in importance:
            importance_dict[feat_name] = importance[xgb_feat]
        else:
            importance_dict[feat_name] = 0.0
    
    # Sort by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    print("Top 12 Features by Importance (Gain):")
    print("-"*80)
    for i, (feat, imp) in enumerate(sorted_features[:12], 1):
        print(f"{i:2d}. {feat:30s} {imp:10.2f}")
    
    print()
    
    return importance_dict, feature_names


def compare_injury_features(old_importance, new_importance):
    """Compare injury-related features between models."""
    print("="*80)
    print("INJURY FEATURE COMPARISON")
    print("="*80)
    print()
    
    injury_features = [
        'home_minutes_missing_top2',
        'home_star_out',
        'away_minutes_missing_top2',
        'away_star_out',
        'home_usage_share_top2',
        'away_usage_share_top2'
    ]
    
    print("Feature                         Old Model    New Model    Change")
    print("-"*80)
    
    total_old = sum(old_importance.values())
    total_new = sum(new_importance.values())
    
    injury_old_total = 0
    injury_new_total = 0
    
    for feat in injury_features:
        old_val = old_importance.get(feat, 0.0)
        new_val = new_importance.get(feat, 0.0)
        change = new_val - old_val
        
        # Calculate percentage of total importance
        old_pct = (old_val / total_old * 100) if total_old > 0 else 0
        new_pct = (new_val / total_new * 100) if total_new > 0 else 0
        
        if 'minutes_missing' in feat or 'star_out' in feat:
            injury_old_total += old_val
            injury_new_total += new_val
        
        change_indicator = "+" if change > 0 else ""
        print(f"{feat:30s} {old_val:8.2f} ({old_pct:4.1f}%)  {new_val:8.2f} ({new_pct:4.1f}%)  {change_indicator}{change:8.2f}")
    
    print()
    print("Summary:")
    print("-"*80)
    
    # Core injury features (minutes_missing_top2 and star_out)
    core_injury_old = sum(old_importance.get(f, 0) for f in injury_features if 'minutes_missing' in f or 'star_out' in f)
    core_injury_new = sum(new_importance.get(f, 0) for f in injury_features if 'minutes_missing' in f or 'star_out' in f)
    
    core_old_pct = (core_injury_old / total_old * 100) if total_old > 0 else 0
    core_new_pct = (core_injury_new / total_new * 100) if total_new > 0 else 0
    
    print(f"Core injury features (minutes_missing + star_out):")
    print(f"  Old model: {core_injury_old:8.2f} ({core_old_pct:4.1f}% of total importance)")
    print(f"  New model: {core_injury_new:8.2f} ({core_new_pct:4.1f}% of total importance)")
    print(f"  Change: {core_injury_new - core_injury_old:+8.2f} ({core_new_pct - core_old_pct:+4.1f} pp)")
    
    print()
    
    # Assessment
    print("ASSESSMENT:")
    print("-"*80)
    
    if core_new_pct >= 15:
        print("[EXCELLENT] Injury features are highly important (15%+ of total)")
        print("The reconstruction is capturing meaningful signal.")
    elif core_new_pct >= 10:
        print("[GOOD] Injury features have meaningful importance (10-15% of total)")
        print("The reconstruction is working as intended.")
    elif core_new_pct >= 5:
        print("[MODERATE] Injury features have some importance (5-10% of total)")
        print("They're contributing but not dominant.")
    else:
        print("[POOR] Injury features have low importance (<5% of total)")
        print("Either injuries don't matter much, or reconstruction is too noisy.")
    
    print()
    
    if core_new_pct > core_old_pct * 1.5:
        print("[SUCCESS] New model uses injury features significantly more than old model.")
        print("This validates that unified injury handling provides meaningful signal.")
    elif core_new_pct > core_old_pct:
        print("[IMPROVED] New model uses injury features more than old model.")
    else:
        print("[CONCERN] New model doesn't show increased injury feature importance.")


def main():
    print()
    print("="*80)
    print("FEATURE IMPORTANCE ANALYSIS: OLD vs NEW CHAMPION MODEL")
    print("="*80)
    print()
    print("Objective: Validate that injury features are meaningful in the new model")
    print()
    
    # Analyze old model
    old_dir = "artifacts/champion_corrected_rest_days_BEFORE_INJURY_FIX"
    old_importance, old_features = analyze_model_importance(old_dir, "Old Champion (Legacy Injury Handling)")
    
    print()
    
    # Analyze new model
    new_dir = "artifacts/champion_corrected_rest_days"
    new_importance, new_features = analyze_model_importance(new_dir, "New Champion (Unified Injury Handling)")
    
    print()
    
    # Compare injury features
    compare_injury_features(old_importance, new_importance)
    
    print()
    print("="*80)
    print("CONCLUSION")
    print("="*80)
    print()
    print("If injury features show meaningful importance in the new model,")
    print("this validates that the injury reconstruction methodology is capturing")
    print("real signal rather than just noise.")
    print()


if __name__ == '__main__':
    main()

