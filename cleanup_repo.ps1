# Cleanup Script for NBA Predictions Repository
# Removes old bivariate model code, experiments, and temporary files

$ErrorActionPreference = "Stop"

Write-Host "=" -ForegroundColor Red -NoNewline
Write-Host ("=" * 79) -ForegroundColor Red
Write-Host "NBA PREDICTIONS REPOSITORY CLEANUP" -ForegroundColor Red
Write-Host ("=" * 80) -ForegroundColor Red
Write-Host ""
Write-Host "This script will delete old model code, experiments, and temporary files." -ForegroundColor Yellow
Write-Host "The current margin model, data, and strategy analysis will be preserved." -ForegroundColor Green
Write-Host ""

# Define all items to delete
$itemsToDelete = @(
    # Category 1: Old Bivariate Poisson Model
    "train.py",
    "predict.py",
    "evaluate.py",
    "calibration",
    "configs\default.yaml",
    
    # Category 2: Old Model Artifacts
    "artifacts\run_2022_2024",
    "artifacts\run_baseline",
    "artifacts\run_modern",
    "artifacts\run_small",
    "artifacts\run_with_players",
    "artifacts\margin_test",
    
    # Category 3: Old Reports
    "reports\2425_with_players",
    "reports\baseline_2425",
    "reports\run_2022_2024",
    "reports\run_2425_val",
    "reports\run_modern_predictions_2024_25.json",
    "reports\run_modern_val",
    "reports\run_modern_val_fast",
    "reports\val_fixed_calibration",
    "reports\val_no_ats_calib",
    "reports\val_with_players",
    "reports\feature_importance",
    "reports\calibrated_2425",
    
    # Category 4: Calibration Experiment Scripts
    "scripts\calibrate_predictions.py",
    "scripts\compare_calibration.py",
    "scripts\walk_forward_meta_validation.py",
    "scripts\test_advanced_corrections.py",
    "scripts\test_sigma_calibration.py",
    "scripts\diagnose_mu_bias.py",
    "scripts\investigate_home_away_bias.py",
    "run_calibration.ps1",
    
    # Category 5: Temporary Diagnostic Scripts
    "scripts\check_nan_values.py",
    "scripts\compare_bivariate_vs_margin.py",
    "scripts\test_margin_approach.ps1",
    "scripts\test_normalized_features.py",
    "analyze_results.py",
    "compare_baseline_vs_current.py",
    "quick_calibration_check.py",
    "quick_test.py",
    
    # Category 6: Old Data Generation Scripts
    "train_and_evaluate.ps1",
    "split_existing.py",
    
    # Category 7: Play-by-Play Simulation (Legacy)
    "generate_training_data_OPTIMIZED.py",
    "generate_training_data.py",
    "player_stats_from_pbp.py",
    "build_pca_cache_optimized.py",
    "build_player_cache_optimized.py",
    "pca_optimized.py",
    "pca.py",
    "transform_player_stats_optimized.py",
    "transform_player_stats.py",
    "game_context_builder.py",
    "game_contexts",
    "main.py",
    
    # Category 8: Old Analysis Scripts
    "analysis\debug_foul_counting.py",
    "analysis\player_stats.py",
    "analysis\tmp_prediction_summary.py",
    "analysis\validate_compact_jsonl.py",
    "analysis\validate_gemini_output.py",
    "analysis\validate_player_fouls.py",
    "analysis\validate_team_fouls.py",
    "analysis\validate_tuple_distributions.py",
    "analysis\walk_forward_meta_results.csv",
    "diagnostics",
    
    # Category 9: Old Config & Data
    "config",
    "data\games_train_baseline_90.jsonl",
    "data\games_train_baseline.jsonl",
    "data\games_val_baseline.jsonl",
    "data\games_predict_2024_2025_baseline.jsonl",
    "data\games_train_with_players.jsonl",
    "predictions\per_game_predictions.csv",
    "predictions\baseline_2425_predictions.csv",
    "predictions\val_with_players_predictions.csv",
    
    # Category 10: Miscellaneous Old Files
    "convert_data.py",
    "generate_2023_2024_season.py",
    "generate_lineup.py",
    "get_team_city.py",
    "qa_game_data.py",
    "season_game_validator.py",
    "transform_boxscore_to_games.py",
    "DATA_SOURCES_SUMMARY.md",
    "TRAINING_COMMANDS.md",
    "nba_predictions.np_sizes",
    "notes",
    "sample_[10-22-2024]-[06-22-2025]-combined-stats.csv",
    "sample_2024-2025_NBA_Box_Score_Team-Stats.xlsx",
    "sample_NBA-2024-2025-Player-BoxScore-Dataset.xlsx",
    
    # Category 11: Game Simulation Code
    "game"
)

# Count items
$totalItems = $itemsToDelete.Count
Write-Host "Items to delete: $totalItems" -ForegroundColor Cyan
Write-Host ""

# Show what will be deleted
Write-Host "Categories:" -ForegroundColor Cyan
Write-Host "  [1] Old Bivariate Poisson Model code"
Write-Host "  [2] Old Model Artifacts (6 runs)"
Write-Host "  [3] Old Reports (11 report directories)"
Write-Host "  [4] Calibration Experiment Scripts (didn't help)"
Write-Host "  [5] Temporary Diagnostic Scripts"
Write-Host "  [6] Old Data Generation Scripts"
Write-Host "  [7] Play-by-Play Simulation (legacy)"
Write-Host "  [8] Old Analysis Scripts"
Write-Host "  [9] Old Config & Data"
Write-Host "  [10] Miscellaneous Old Files"
Write-Host "  [11] Game Simulation Code"
Write-Host ""

Write-Host "Files and directories that will be KEPT:" -ForegroundColor Green
Write-Host "  - train_margin.py, predict_margin.py, evaluate_margin.py"
Write-Host "  - models/margin_*.py"
Write-Host "  - training/margin_dataset.py"
Write-Host "  - configs/margin_default.yaml"
Write-Host "  - league_normalizer.py"
Write-Host "  - All current data processing scripts"
Write-Host "  - scripts/normalize_all_data.py"
Write-Host "  - scripts/analyze_filtering_strategies.py"
Write-Host "  - scripts/validate_away_favorites_strategy.py"
Write-Host "  - scripts/audit_features_comprehensive.py"
Write-Host "  - scripts/qa_normalized_data.py"
Write-Host "  - train_and_evaluate_normalized.ps1"
Write-Host "  - artifacts/margin_normalized/ (current model)"
Write-Host "  - reports/normalized_2425/ (current results)"
Write-Host "  - All *_norm.jsonl data files"
Write-Host "  - README.md, requirements.txt, etc."
Write-Host ""

# Confirmation prompt
$confirmation = Read-Host "Type 'DELETE' to proceed with cleanup (or anything else to cancel)"

if ($confirmation -ne "DELETE") {
    Write-Host ""
    Write-Host "Cleanup cancelled." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "=" -ForegroundColor Red -NoNewline
Write-Host ("=" * 79) -ForegroundColor Red
Write-Host "STARTING CLEANUP" -ForegroundColor Red
Write-Host ("=" * 80) -ForegroundColor Red
Write-Host ""

$deletedCount = 0
$notFoundCount = 0
$errorCount = 0

foreach ($item in $itemsToDelete) {
    $fullPath = Join-Path $PSScriptRoot $item
    
    if (Test-Path $fullPath) {
        try {
            if (Test-Path $fullPath -PathType Container) {
                Write-Host "[DIR]  Deleting: $item" -ForegroundColor Yellow
                Remove-Item -Path $fullPath -Recurse -Force
            } else {
                Write-Host "[FILE] Deleting: $item" -ForegroundColor Gray
                Remove-Item -Path $fullPath -Force
            }
            $deletedCount++
        } catch {
            Write-Host "[ERROR] Failed to delete: $item" -ForegroundColor Red
            Write-Host "        Error: $($_.Exception.Message)" -ForegroundColor Red
            $errorCount++
        }
    } else {
        Write-Host "[SKIP] Not found: $item" -ForegroundColor DarkGray
        $notFoundCount++
    }
}

Write-Host ""
Write-Host "=" -ForegroundColor Green -NoNewline
Write-Host ("=" * 79) -ForegroundColor Green
Write-Host "CLEANUP COMPLETE" -ForegroundColor Green
Write-Host ("=" * 80) -ForegroundColor Green
Write-Host ""
Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "  Deleted: $deletedCount items" -ForegroundColor Green
Write-Host "  Not found: $notFoundCount items" -ForegroundColor DarkGray
Write-Host "  Errors: $errorCount items" -ForegroundColor $(if ($errorCount -gt 0) { "Red" } else { "Green" })
Write-Host ""

if ($errorCount -eq 0) {
    Write-Host "[+] Repository cleanup successful!" -ForegroundColor Green
    Write-Host "    The repo now contains only the current margin model and strategy." -ForegroundColor Green
} else {
    Write-Host "[!] Some items could not be deleted. Check errors above." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Review remaining files with: Get-ChildItem -Recurse"
Write-Host "  2. Commit changes: git add -A && git commit -m 'Clean up old experiments'"
Write-Host "  3. Continue with current strategy using train_and_evaluate_normalized.ps1"
Write-Host ""

