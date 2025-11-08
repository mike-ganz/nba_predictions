# Test if retuned hyperparameters can match performance without redundancy
#
# Goal: See if colsample_bytree=0.94 can achieve 60%+ ATS on 2025-26
# without needing the away_tov_edge redundant feature

$ErrorActionPreference = "Stop"
$startTime = Get-Date

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "TESTING RETUNED HYPERPARAMETERS" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  Goal: Match 60.50% ATS (2025-26) WITHOUT redundant features" -ForegroundColor White
Write-Host ""
Write-Host "  Key Changes:" -ForegroundColor Yellow
Write-Host "    colsample_bytree: 0.70 → 0.94 (+35%)" -ForegroundColor Yellow
Write-Host "    subsample:        0.60 → 0.99 (+65%)" -ForegroundColor Yellow
Write-Host "    n_estimators:     100 → 476 (+376%)" -ForegroundColor Yellow
Write-Host ""

# Create output directory
$outputDir = "reports\retuned_hyperparameters_test"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "STEP 1: Train Model with Retuned Hyperparameters" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

$artifact = "artifacts\margin_exp2_retuned"
python train_margin_extended.py `
    --data data/games_train_with_players_90_norm.jsonl `
    --config configs/margin_xgboost_exp2_retuned.yaml `
    --output $artifact

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ✗ Training failed!" -ForegroundColor Red
    exit 1
}

Write-Host "  ✓ Training complete" -ForegroundColor Green
Write-Host ""

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "STEP 2: Test on 2024-25 Holdout" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

$pred2425 = "predictions\exp2_retuned_2425.csv"
python predict_margin.py `
    --data data/games_predict_2024_2025_with_players_norm.jsonl `
    --model $artifact `
    --output $pred2425 | Out-Null

python evaluate_margin.py `
    --predictions $pred2425 `
    --output "$outputDir\holdout_2425" | Out-Null

Write-Host "  ✓ Evaluation complete" -ForegroundColor Green
Write-Host ""

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "STEP 3: Test on 2025-26 Current Season" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

$pred2526 = "predictions\exp2_retuned_2526.csv"
python predict_margin.py `
    --data data/games_2025_2026_current_norm.jsonl `
    --model $artifact `
    --output $pred2526 | Out-Null

python evaluate_margin.py `
    --predictions $pred2526 `
    --output "$outputDir\current_2526" | Out-Null

Write-Host "  ✓ Evaluation complete" -ForegroundColor Green
Write-Host ""

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "RESULTS" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Parse results
$summary2425 = Get-Content "$outputDir\holdout_2425\evaluation_summary.txt" -Raw
$summary2526 = Get-Content "$outputDir\current_2526\evaluation_summary.txt" -Raw

$ats2425 = [double][regex]::Match($summary2425, "ATS Accuracy:\s+([\d\.]+)%").Groups[1].Value
$mae2425 = [double][regex]::Match($summary2425, "Mean Absolute Error:\s+([\d\.]+)").Groups[1].Value
$rmse2425 = [double][regex]::Match($summary2425, "Root Mean Squared Error:\s+([\d\.]+)").Groups[1].Value

$ats2526 = [double][regex]::Match($summary2526, "ATS Accuracy:\s+([\d\.]+)%").Groups[1].Value
$mae2526 = [double][regex]::Match($summary2526, "Mean Absolute Error:\s+([\d\.]+)").Groups[1].Value
$rmse2526 = [double][regex]::Match($summary2526, "Root Mean Squared Error:\s+([\d\.]+)").Groups[1].Value

Write-Host "2024-25 Holdout Results:" -ForegroundColor White
Write-Host "  ATS Accuracy: $($ats2425.ToString('F2'))%" -ForegroundColor White
Write-Host "  MAE:          $($mae2425.ToString('F2')) points" -ForegroundColor White
Write-Host "  RMSE:         $($rmse2425.ToString('F2')) points" -ForegroundColor White
Write-Host ""

Write-Host "2025-26 Current Season Results:" -ForegroundColor White
Write-Host "  ATS Accuracy: $($ats2526.ToString('F2'))%" -ForegroundColor White
Write-Host "  MAE:          $($mae2526.ToString('F2')) points" -ForegroundColor White
Write-Host "  RMSE:         $($rmse2526.ToString('F2')) points" -ForegroundColor White
Write-Host ""

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "COMPARISON TO PREVIOUS RESULTS" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "2024-25 Holdout:" -ForegroundColor Yellow
Write-Host "  Exp2 (colsample=0.7, with redundancy):    49.66%" -ForegroundColor White
Write-Host "  Exp2 (colsample=1.0, no redundancy):      50.27%" -ForegroundColor White
Write-Host "  Exp2 (colsample=0.94, no redundancy):     $($ats2425.ToString('F2'))%" -ForegroundColor $(if ($ats2425 -ge 50.0) { "Green" } else { "Red" })
Write-Host ""

Write-Host "2025-26 Current Season:" -ForegroundColor Yellow
Write-Host "  Exp2 (colsample=0.7, with redundancy):    60.50%  ← BEST SO FAR" -ForegroundColor White
Write-Host "  Exp2 (colsample=1.0, no redundancy):      56.30%" -ForegroundColor White
Write-Host "  Exp2 (colsample=0.94, no redundancy):     $($ats2526.ToString('F2'))%" -ForegroundColor $(if ($ats2526 -ge 60.0) { "Green" } elseif ($ats2526 -ge 58.0) { "Yellow" } else { "Red" })
Write-Host ""

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "VERDICT" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

if ($ats2526 -ge 60.5) {
    Write-Host "🎉 SUCCESS! MATCHED OR EXCEEDED TARGET!" -ForegroundColor Green
    Write-Host ""
    Write-Host "The retuned hyperparameters achieve $($ats2526.ToString('F2'))% ATS on 2025-26" -ForegroundColor Green
    Write-Host "WITHOUT needing redundant features!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Key insight: colsample_bytree=0.94 provides enough feature availability" -ForegroundColor Green
    Write-Host "that redundant features are no longer needed." -ForegroundColor Green
    Write-Host ""
    Write-Host "✅ DEPLOY THIS MODEL" -ForegroundColor Green
} elseif ($ats2526 -ge 58.0) {
    Write-Host "⚠️  PARTIAL SUCCESS - Close but not quite there" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "The retuned hyperparameters achieve $($ats2526.ToString('F2'))% ATS on 2025-26" -ForegroundColor Yellow
    Write-Host "Gap to target: $((60.5 - $ats2526).ToString('F2')) percentage points" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor White
    Write-Host "  1. Keep redundant feature (gets 60.50%)" -ForegroundColor White
    Write-Host "  2. Deploy this cleaner model (gets $($ats2526.ToString('F2'))%)" -ForegroundColor White
    Write-Host "  3. Try Phase 2 tuning optimizing ATS directly" -ForegroundColor White
} else {
    Write-Host "❌ DID NOT MEET TARGET" -ForegroundColor Red
    Write-Host ""
    Write-Host "The retuned hyperparameters achieve $($ats2526.ToString('F2'))% ATS on 2025-26" -ForegroundColor Red
    Write-Host "Gap to target: $((60.5 - $ats2526).ToString('F2')) percentage points" -ForegroundColor Red
    Write-Host ""
    Write-Host "Recommendation: Keep redundant feature for now, or run Phase 2 tuning" -ForegroundColor Red
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "TEST COMPLETE" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

$elapsed = (Get-Date) - $startTime
Write-Host "  Total time: $([math]::Round($elapsed.TotalMinutes, 1)) minutes" -ForegroundColor White
Write-Host "  Results saved to: $outputDir" -ForegroundColor White
Write-Host ""

