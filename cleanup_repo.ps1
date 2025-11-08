# Cleanup Script - Remove Temporary Investigation Files and Old Model Artifacts
# Created: 2025-11-08
# Purpose: Clean up investigation files and consolidate model artifacts after rest days fix

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Repository Cleanup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$rootPath = $PSScriptRoot
Set-Location $rootPath

# Track what we're deleting
$deletedFiles = @()
$deletedDirs = @()
$errors = @()

function Remove-ItemSafe {
    param(
        [string]$Path,
        [string]$Type = "File"
    )
    
    $fullPath = Join-Path $rootPath $Path
    
    if (Test-Path $fullPath) {
        try {
            Remove-Item -Path $fullPath -Recurse -Force -ErrorAction Stop
            if ($Type -eq "Directory") {
                $script:deletedDirs += $Path
            } else {
                $script:deletedFiles += $Path
            }
            Write-Host "[OK] Deleted: $Path" -ForegroundColor Green
        } catch {
            $script:errors += "Failed to delete $Path : $_"
            Write-Host "[ERROR] Failed to delete: $Path" -ForegroundColor Red
        }
    } else {
        Write-Host "[SKIP] Not found: $Path" -ForegroundColor Yellow
    }
}

# Display what will be deleted
Write-Host "This script will delete the following:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Investigation/Debugging Documentation (8 files)" -ForegroundColor White
Write-Host "2. Temporary Verification Scripts (3 files)" -ForegroundColor White
Write-Host "3. Old Backup Directories (3 directories)" -ForegroundColor White
Write-Host "4. Redundant Model Artifacts (~22 model directories)" -ForegroundColor White
Write-Host "5. Redundant Config Files (~21 config files)" -ForegroundColor White
Write-Host ""
Write-Host "Total: ~57 items" -ForegroundColor Cyan
Write-Host ""

$confirmation = Read-Host "Do you want to proceed? (yes/no)"
if ($confirmation -ne "yes") {
    Write-Host "Cleanup cancelled." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "Starting cleanup..." -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# 1. Investigation/Debugging Documentation
# ============================================================================
Write-Host "Removing investigation documentation..." -ForegroundColor Cyan

Remove-ItemSafe "BUG_REPORT_rest_days_calculation.md"
Remove-ItemSafe "FINDINGS_rest_days_discrepancy.md"
Remove-ItemSafe "data_comparison_2025-11-07.md"
Remove-ItemSafe "README_FILES_CREATED.md"
Remove-ItemSafe "REST_DAYS_FIX_README.md"
Remove-ItemSafe "VISUAL_SUMMARY.txt"
Remove-ItemSafe "QUICK_MODE_RESULTS.md"
Remove-ItemSafe "FIX_SUMMARY_20251108_141224.txt"

# ============================================================================
# 2. Temporary Verification Scripts
# ============================================================================
Write-Host ""
Write-Host "Removing temporary verification scripts..." -ForegroundColor Cyan

Remove-ItemSafe "check_nov7_rest_days.py"
Remove-ItemSafe "compare_old_new_rest_days.py"
Remove-ItemSafe "verify_rest_days_fix.py"

# ============================================================================
# 3. Old Backup Directories
# ============================================================================
Write-Host ""
Write-Host "Removing old backup directories..." -ForegroundColor Cyan

Remove-ItemSafe "backups\pre_rest_days_fix_20251108_141046" "Directory"
Remove-ItemSafe "backups\pre_rest_days_fix_20251108_141143" "Directory"
Remove-ItemSafe "backups\pre_rest_days_fix_20251108_141224" "Directory"

# ============================================================================
# 4. Redundant Model Artifacts
# ============================================================================
Write-Host ""
Write-Host "Removing redundant model artifacts..." -ForegroundColor Cyan

# Old baselines
Remove-ItemSafe "artifacts\margin_ridge_baseline" "Directory"
Remove-ItemSafe "artifacts\margin_ridge_no_diff" "Directory"
Remove-ItemSafe "artifacts\margin_xgboost" "Directory"
Remove-ItemSafe "artifacts\margin_xgboost_baseline_no_pace_diff" "Directory"
Remove-ItemSafe "artifacts\margin_xgboost_no_diff" "Directory"
Remove-ItemSafe "artifacts\margin_xgboost_tuned" "Directory"
Remove-ItemSafe "artifacts\margin_xgboost_ats" "Directory"

# Experiment variants
Remove-ItemSafe "artifacts\margin_exp1_optimal" "Directory"
Remove-ItemSafe "artifacts\margin_exp2_optimal" "Directory"
Remove-ItemSafe "artifacts\margin_exp2_ats_tuned" "Directory"
Remove-ItemSafe "artifacts\margin_exp2_colsample1_with_redundancy" "Directory"
Remove-ItemSafe "artifacts\margin_exp2_colsample1_without_redundancy" "Directory"
Remove-ItemSafe "artifacts\margin_exp2_retuned" "Directory"
Remove-ItemSafe "artifacts\margin_exp3_optimal" "Directory"
Remove-ItemSafe "artifacts\margin_experiment1_home_ftr_ats" "Directory"
Remove-ItemSafe "artifacts\margin_experiment1_home_ftr_no_redundancy" "Directory"
Remove-ItemSafe "artifacts\margin_experiment1_home_ftr_v2" "Directory"
Remove-ItemSafe "artifacts\margin_experiment2_ftr_plus_role_ats" "Directory"
Remove-ItemSafe "artifacts\margin_experiment2_ftr_plus_role_no_redundancy" "Directory"
Remove-ItemSafe "artifacts\margin_experiment2_ftr_plus_role_v2" "Directory"
Remove-ItemSafe "artifacts\margin_experiment3_role_only_ats" "Directory"
Remove-ItemSafe "artifacts\margin_experiment3_role_only_no_redundancy" "Directory"

# Superseded champion versions
Remove-ItemSafe "artifacts\margin_champion_ats" "Directory"
Remove-ItemSafe "artifacts\margin_champion_optimal" "Directory"
Remove-ItemSafe "artifacts\margin_champion_v2" "Directory"

# ============================================================================
# 5. Redundant Config Files
# ============================================================================
Write-Host ""
Write-Host "Removing redundant config files..." -ForegroundColor Cyan

Remove-ItemSafe "configs\margin_default.yaml"
Remove-ItemSafe "configs\margin_ridge_no_diff.yaml"
Remove-ItemSafe "configs\margin_xgboost.yaml"
Remove-ItemSafe "configs\margin_xgboost_ats.yaml"
Remove-ItemSafe "configs\margin_xgboost_ats_objective.yaml"
Remove-ItemSafe "configs\margin_xgboost_champion_ats.yaml"
Remove-ItemSafe "configs\margin_xgboost_champion_optimal_hyperparams.yaml"
Remove-ItemSafe "configs\margin_xgboost_exp1_optimal_hyperparams.yaml"
Remove-ItemSafe "configs\margin_xgboost_exp2_ats_tuned.yaml"
Remove-ItemSafe "configs\margin_xgboost_exp2_colsample1_with_redundancy.yaml"
Remove-ItemSafe "configs\margin_xgboost_exp2_colsample1_without_redundancy.yaml"
Remove-ItemSafe "configs\margin_xgboost_exp2_optimal_hyperparams.yaml"
Remove-ItemSafe "configs\margin_xgboost_exp2_retuned.yaml"
Remove-ItemSafe "configs\margin_xgboost_exp3_optimal_hyperparams.yaml"
Remove-ItemSafe "configs\margin_xgboost_experiment1_home_ftr_ats.yaml"
Remove-ItemSafe "configs\margin_xgboost_experiment1_home_ftr_no_redundancy.yaml"
Remove-ItemSafe "configs\margin_xgboost_experiment2_home_ftr_plus_role_ats.yaml"
Remove-ItemSafe "configs\margin_xgboost_experiment2_home_ftr_plus_role_no_redundancy.yaml"
Remove-ItemSafe "configs\margin_xgboost_experiment3_role_only_ats.yaml"
Remove-ItemSafe "configs\margin_xgboost_experiment3_role_only_no_redundancy.yaml"
Remove-ItemSafe "configs\margin_xgboost_tuned.yaml"

# ============================================================================
# Summary
# ============================================================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Cleanup Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Summary:" -ForegroundColor White
Write-Host "  Files deleted:       $($deletedFiles.Count)" -ForegroundColor Green
Write-Host "  Directories deleted: $($deletedDirs.Count)" -ForegroundColor Green
Write-Host "  Errors:              $($errors.Count)" -ForegroundColor $(if ($errors.Count -gt 0) { "Red" } else { "Green" })
Write-Host ""

if ($errors.Count -gt 0) {
    Write-Host "Errors encountered:" -ForegroundColor Red
    foreach ($error in $errors) {
        Write-Host "  - $error" -ForegroundColor Red
    }
    Write-Host ""
}

Write-Host "Kept active models:" -ForegroundColor Cyan
Write-Host "  - artifacts\champion_individually_tuned\" -ForegroundColor Green
Write-Host "  - artifacts\challenger_wide_search\" -ForegroundColor Green
Write-Host "  - artifacts\challenger_timeseries_cv\" -ForegroundColor Green
Write-Host "  - artifacts\margin_xgboost_optimized\" -ForegroundColor Green
Write-Host "  - artifacts\margin_xgboost_optimized_with2425\" -ForegroundColor Green
Write-Host "  - artifacts\margin_xgboost_optimized_with2425_no_redundancy\" -ForegroundColor Green
Write-Host ""

Write-Host "Kept configs:" -ForegroundColor Cyan
Write-Host "  - configs\champion_individually_tuned.yaml" -ForegroundColor Green
Write-Host "  - configs\challenger_wide_search.yaml" -ForegroundColor Green
Write-Host "  - configs\champion_timeseries_cv.yaml" -ForegroundColor Green
Write-Host "  - configs\margin_xgboost_optimized.yaml" -ForegroundColor Green
Write-Host "  - configs\margin_xgboost_optimized_no_redundancy.yaml" -ForegroundColor Green
Write-Host ""

Write-Host "You can now run .\fix_rest_days_and_retrain.ps1 with a clean repo!" -ForegroundColor Yellow
Write-Host ""

