# Cleanup Script for NBA Predictions Experimental Files
# 
# This script removes files created during the hyperparameter tuning and
# feature experimentation workstream (November 2024).
#
# PRESERVED FILES (NOT deleted):
#   • artifacts/champion_individually_tuned/ - Final production model
#   • artifacts/margin_xgboost_optimized_with2425/ - Original champion (reference)
#   • configs/champion_individually_tuned.yaml - Final config
#   • configs/margin_xgboost_optimized.yaml - Original config
#   • daily_betting_recommendations_champion.py - New production script
#   • daily_betting_recommendations.py - Original production script
#
# Run with: .\cleanup_experimental_files.ps1

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "EXPERIMENTAL FILES CLEANUP" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This script will DELETE the following experimental files and directories:" -ForegroundColor Yellow
Write-Host ""

# ============================================================================
# CATEGORY 1: One-Time Tuning & Analysis Scripts
# ============================================================================
$tuningScripts = @(
    "tune_exp2_hyperparameters.py",
    "tune_exp2_phase2_ats.py",
    "tune_model_specific.py",
    "verify_feature_redundancy.py",
    "investigate_redundancy_mystery.py",
    "create_optimal_configs.py"
)

Write-Host "TUNING & ANALYSIS SCRIPTS:" -ForegroundColor Yellow
foreach ($file in $tuningScripts) {
    if (Test-Path $file) {
        Write-Host "  - $file" -ForegroundColor White
    }
}
Write-Host ""

# ============================================================================
# CATEGORY 2: PowerShell Experiment Orchestration Scripts
# ============================================================================
$psScripts = @(
    "compare_ftr_experiments.ps1",
    "compare_with_without_tov_redundancy.ps1",
    "test_colsample_hypothesis.ps1",
    "retrain_and_evaluate_optimal.ps1"
)

Write-Host "POWERSHELL EXPERIMENT SCRIPTS:" -ForegroundColor Yellow
foreach ($file in $psScripts) {
    if (Test-Path $file) {
        Write-Host "  - $file" -ForegroundColor White
    }
}
Write-Host ""

# ============================================================================
# CATEGORY 3: Experimental Model Configs (keeping only champion_individually_tuned.yaml)
# ============================================================================
$expConfigs = @(
    "configs\margin_xgboost_experiment1_home_ftr.yaml",
    "configs\margin_xgboost_experiment2_home_ftr_plus_role.yaml",
    "configs\margin_xgboost_experiment3_role_only.yaml",
    "configs\champion_optimal.yaml",
    "configs\exp1_optimal.yaml",
    "configs\exp2_optimal.yaml",
    "configs\exp3_optimal.yaml",
    "configs\exp1_individually_tuned.yaml",
    "configs\exp2_individually_tuned.yaml",
    "configs\exp3_individually_tuned.yaml"
)

Write-Host "EXPERIMENTAL CONFIG FILES:" -ForegroundColor Yellow
foreach ($file in $expConfigs) {
    if (Test-Path $file) {
        Write-Host "  - $file" -ForegroundColor White
    }
}
Write-Host ""

# ============================================================================
# CATEGORY 4: Experimental Model Artifacts (keeping only champion_individually_tuned)
# ============================================================================
$expArtifacts = @(
    "artifacts\margin_experiment1_home_ftr",
    "artifacts\margin_experiment2_ftr_plus_role",
    "artifacts\margin_experiment3_role_only",
    "artifacts\margin_experiment1_ats_loss",
    "artifacts\margin_experiment2_ats_loss",
    "artifacts\margin_experiment3_ats_loss",
    "artifacts\margin_champion_ats_loss",
    "artifacts\champion_optimal",
    "artifacts\exp1_optimal",
    "artifacts\exp2_optimal",
    "artifacts\exp3_optimal",
    "artifacts\exp1_individually_tuned",
    "artifacts\exp2_individually_tuned",
    "artifacts\exp3_individually_tuned",
    "artifacts\champion_no_redundancy",
    "artifacts\exp1_no_redundancy",
    "artifacts\exp2_no_redundancy",
    "artifacts\exp3_no_redundancy",
    "artifacts\champion_colsample1",
    "artifacts\exp2_colsample1",
    "artifacts\exp2_phase1_tuned",
    "artifacts\exp2_phase2_ats_tuned"
)

Write-Host "EXPERIMENTAL MODEL ARTIFACTS:" -ForegroundColor Yellow
foreach ($dir in $expArtifacts) {
    if (Test-Path $dir) {
        Write-Host "  - $dir\" -ForegroundColor White
    }
}
Write-Host ""

# ============================================================================
# CATEGORY 5: Experiment Report Directories
# ============================================================================
$reportDirs = @(
    "reports\ftr_experiments",
    "reports\ats_experiments",
    "reports\redundancy_comparison",
    "reports\colsample_test",
    "reports\optimal_comparison",
    "reports\rigorous_comparison",
    "reports\champion_comparison"
)

Write-Host "EXPERIMENT REPORT DIRECTORIES:" -ForegroundColor Yellow
foreach ($dir in $reportDirs) {
    if (Test-Path $dir) {
        Write-Host "  - $dir\" -ForegroundColor White
    }
}
Write-Host ""

# ============================================================================
# CATEGORY 6: Experimental Prediction Files
# ============================================================================
$predPatterns = @(
    "predictions\*_exp1_*.csv",
    "predictions\*_exp2_*.csv",
    "predictions\*_exp3_*.csv",
    "predictions\*_champion_no_redundancy_*.csv",
    "predictions\*_champion_colsample1_*.csv",
    "predictions\*_champion_optimal_*.csv",
    "predictions\*_exp*_optimal_*.csv",
    "predictions\*_exp*_individually_tuned_*.csv",
    "predictions\*_rigorous_*.csv",
    "predictions\original_champion_*.csv",
    "predictions\new_champion_*.csv",
    "predictions\email_preview_exp2_*.html",
    "predictions\email_backup_exp2_*.html"
)

Write-Host "EXPERIMENTAL PREDICTION FILES:" -ForegroundColor Yellow
$predCount = 0
foreach ($pattern in $predPatterns) {
    $files = Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue
    if ($files) {
        $predCount += $files.Count
    }
}
if ($predCount -gt 0) {
    Write-Host "  - $predCount prediction/email files from experiments" -ForegroundColor White
}
Write-Host ""

# ============================================================================
# CATEGORY 7: Deprecated Production Scripts
# ============================================================================
$deprecatedScripts = @(
    "daily_betting_recommendations_exp2.py"
)

Write-Host "DEPRECATED PRODUCTION SCRIPTS:" -ForegroundColor Yellow
foreach ($file in $deprecatedScripts) {
    if (Test-Path $file) {
        Write-Host "  - $file" -ForegroundColor White
    }
}
Write-Host ""

# ============================================================================
# Count total items
# ============================================================================
$totalItems = 0
$totalItems += ($tuningScripts | Where-Object { Test-Path $_ }).Count
$totalItems += ($psScripts | Where-Object { Test-Path $_ }).Count
$totalItems += ($expConfigs | Where-Object { Test-Path $_ }).Count
$totalItems += ($expArtifacts | Where-Object { Test-Path $_ }).Count
$totalItems += ($reportDirs | Where-Object { Test-Path $_ }).Count
$totalItems += $predCount
$totalItems += ($deprecatedScripts | Where-Object { Test-Path $_ }).Count

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "SUMMARY" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Total items to delete: $totalItems" -ForegroundColor Yellow
Write-Host ""
Write-Host "  PRESERVED (NOT deleted):" -ForegroundColor Green
Write-Host "    + artifacts/champion_individually_tuned/" -ForegroundColor Green
Write-Host "    + artifacts/margin_xgboost_optimized_with2425/" -ForegroundColor Green
Write-Host "    + configs/champion_individually_tuned.yaml" -ForegroundColor Green
Write-Host "    + configs/margin_xgboost_optimized.yaml" -ForegroundColor Green
Write-Host "    + daily_betting_recommendations_champion.py" -ForegroundColor Green
Write-Host "    + daily_betting_recommendations.py" -ForegroundColor Green
Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Prompt for confirmation
$confirmation = Read-Host "Do you want to DELETE these files? (yes/no)"

if ($confirmation -ne "yes") {
    Write-Host ""
    Write-Host "Cleanup cancelled." -ForegroundColor Yellow
    Write-Host ""
    exit 0
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "DELETING FILES..." -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

$deletedCount = 0

# Delete tuning scripts
foreach ($file in $tuningScripts) {
    if (Test-Path $file) {
        Remove-Item $file -Force
        Write-Host "  + Deleted: $file" -ForegroundColor Gray
        $deletedCount++
    }
}

# Delete PowerShell scripts
foreach ($file in $psScripts) {
    if (Test-Path $file) {
        Remove-Item $file -Force
        Write-Host "  + Deleted: $file" -ForegroundColor Gray
        $deletedCount++
    }
}

# Delete experimental configs
foreach ($file in $expConfigs) {
    if (Test-Path $file) {
        Remove-Item $file -Force
        Write-Host "  + Deleted: $file" -ForegroundColor Gray
        $deletedCount++
    }
}

# Delete experimental artifacts
foreach ($dir in $expArtifacts) {
    if (Test-Path $dir) {
        Remove-Item $dir -Recurse -Force
        Write-Host "  + Deleted: $dir\" -ForegroundColor Gray
        $deletedCount++
    }
}

# Delete report directories
foreach ($dir in $reportDirs) {
    if (Test-Path $dir) {
        Remove-Item $dir -Recurse -Force
        Write-Host "  + Deleted: $dir\" -ForegroundColor Gray
        $deletedCount++
    }
}

# Delete prediction files
foreach ($pattern in $predPatterns) {
    $files = Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue
    foreach ($file in $files) {
        Remove-Item $file -Force
        Write-Host "  + Deleted: $($file.Name)" -ForegroundColor Gray
        $deletedCount++
    }
}

# Delete deprecated scripts
foreach ($file in $deprecatedScripts) {
    if (Test-Path $file) {
        Remove-Item $file -Force
        Write-Host "  + Deleted: $file" -ForegroundColor Gray
        $deletedCount++
    }
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "CLEANUP COMPLETE" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Deleted $deletedCount items" -ForegroundColor Green
Write-Host ""
Write-Host "  Your production files are safe:" -ForegroundColor Green
Write-Host "    - artifacts/champion_individually_tuned/" -ForegroundColor White
Write-Host "    - configs/champion_individually_tuned.yaml" -ForegroundColor White
Write-Host "    - daily_betting_recommendations_champion.py" -ForegroundColor White
Write-Host ""

