# Test script for direct margin prediction approach
# This runs the entire pipeline: train -> predict -> evaluate

Set-Location -Path $PSScriptRoot\..
$ErrorActionPreference = "Stop"

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "   TESTING DIRECT MARGIN PREDICTION APPROACH" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

# Create output directory
$outputDir = "artifacts/margin_test"
if (Test-Path $outputDir) {
    Write-Host "`nCleaning existing test artifacts..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $outputDir
}

Write-Host "`n[STEP 1/5] Training margin model on historical data..." -ForegroundColor Yellow
Write-Host "---------------------------------------------------------------------"
python train_margin.py `
    --data data/games_train_with_players_90.jsonl `
    --output artifacts/margin_test `
    --config configs/margin_default.yaml

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: Training failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n[STEP 2/5] Predicting on validation set (2021-2024)..." -ForegroundColor Yellow
Write-Host "---------------------------------------------------------------------"
python predict_margin.py `
    --data data/games_val_with_players.jsonl `
    --model artifacts/margin_test `
    --output artifacts/margin_test/val_predictions.csv

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: Validation prediction failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n[STEP 3/5] Predicting on 2024-2025 season..." -ForegroundColor Yellow
Write-Host "---------------------------------------------------------------------"
python predict_margin.py `
    --data data/games_predict_2024_2025_with_players.jsonl `
    --model artifacts/margin_test `
    --output artifacts/margin_test/2425_predictions.csv

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: 2024-2025 prediction failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n[STEP 4/5] Evaluating validation set performance..." -ForegroundColor Yellow
Write-Host "---------------------------------------------------------------------"
python evaluate_margin.py `
    --predictions artifacts/margin_test/val_predictions.csv `
    --output artifacts/margin_test/val_evaluation

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: Validation evaluation failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n[STEP 5/5] Evaluating 2024-2025 performance..." -ForegroundColor Yellow
Write-Host "---------------------------------------------------------------------"
python evaluate_margin.py `
    --predictions artifacts/margin_test/2425_predictions.csv `
    --output artifacts/margin_test/2425_evaluation

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: 2024-2025 evaluation failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n=====================================================================" -ForegroundColor Green
Write-Host "   TESTING COMPLETE!" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Green

Write-Host "`nResults saved to:"
Write-Host "  Validation:  artifacts/margin_test/val_evaluation/" -ForegroundColor Cyan
Write-Host "  2024-2025:   artifacts/margin_test/2425_evaluation/" -ForegroundColor Cyan

Write-Host "`nNext step: Run comparison script to compare with bivariate model" -ForegroundColor Yellow
Write-Host "  python scripts/compare_bivariate_vs_margin.py" -ForegroundColor Cyan

