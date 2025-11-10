# Cleanup Script: Remove Artifacts from Unified Injury Handling Implementation
# Created: November 9, 2025
# Purpose: Remove temporary files, backups, and old datasets created during implementation

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "  CLEANUP: Unified Injury Handling Implementation Artifacts" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""

$FilesToDelete = @(
    # Backup training data files
    "data/games_train_with_players_90.jsonl.backup",
    "data/games_train_with_players_90_norm.jsonl.backup",
    "data/games_train_with_players_90_BEFORE_INJURY_FIX.jsonl",
    "data/games_train_with_players_90_norm_BEFORE_INJURY_FIX.jsonl",
    
    # Temporary current season data (regenerated each time)
    "data/games_2025_2026_current_new.jsonl",
    "data/games_2025_2026_current_new_norm.jsonl",
    
    # Temporary prediction files
    "predictions/current_season_new_predictions.csv",
    
    # Analysis/comparison scripts (now documented in summary)
    "check_lal_atl_players.py",
    "verify_actual_model_features.py",
    "verify_actual_model_features_v2.py",
    "verify_actual_model_features_v3.py",
    "verify_moneyline_impact.py",
    "compare_game_features.py",
    "identify_features.py",
    "check_rest_days_change.py",
    "analyze_fix_results.py",
    
    # Analysis markdown files (redundant with implementation summary)
    "ANALYSIS_FINDINGS_2025-11-08.md",
    "analysis_2025-11-08_prediction_comparison.md",
    "FINAL_ANALYSIS_2025-11-08.md",
    "FEATURE_IMPORTANCE_ANALYSIS.md",
    
    # Feature comparison outputs
    "feature_comparison_output.txt",
    "actual_features_comparison.txt",
    
    # Temporary configs
    "configs/champion_with_injuries.yaml"
)

$DirectoriesToDelete = @(
    # Old model backup (keep _OLD for reference, delete _BEFORE_INJURY_FIX as it's documented)
    "artifacts/champion_corrected_rest_days_BEFORE_INJURY_FIX"
)

Write-Host "Files to be deleted:" -ForegroundColor Yellow
$DeletedCount = 0
$NotFoundCount = 0

foreach ($File in $FilesToDelete) {
    if (Test-Path $File) {
        Write-Host "  [DELETE] $File" -ForegroundColor Red
        Remove-Item $File -Force
        $DeletedCount++
    } else {
        Write-Host "  [SKIP]   $File (not found)" -ForegroundColor Gray
        $NotFoundCount++
    }
}

Write-Host ""
Write-Host "Directories to be deleted:" -ForegroundColor Yellow

foreach ($Dir in $DirectoriesToDelete) {
    if (Test-Path $Dir) {
        Write-Host "  [DELETE] $Dir" -ForegroundColor Red
        Remove-Item $Dir -Recurse -Force
        $DeletedCount++
    } else {
        Write-Host "  [SKIP]   $Dir (not found)" -ForegroundColor Gray
        $NotFoundCount++
    }
}

Write-Host ""
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "  CLEANUP COMPLETE" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""
Write-Host "Summary:" -ForegroundColor Yellow
Write-Host "  Files/Directories deleted: $DeletedCount" -ForegroundColor White
Write-Host "  Files/Directories not found: $NotFoundCount" -ForegroundColor Gray
Write-Host ""
Write-Host "Note: The following files are KEPT for reference:" -ForegroundColor Cyan
Write-Host "  - UNIFIED_INJURY_IMPLEMENTATION_SUMMARY.md (documentation)" -ForegroundColor White
Write-Host "  - artifacts/champion_corrected_rest_days/ (production model)" -ForegroundColor White
Write-Host "  - data/games_train_with_players_90_norm.jsonl (production training data)" -ForegroundColor White
Write-Host ""

