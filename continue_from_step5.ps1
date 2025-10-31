# Continue from step 5 (training) since data regeneration completed successfully
# Expected time: 2-3 minutes total

$StartTime = Get-Date
$ErrorActionPreference = "Stop"

Write-Host "=" -NoNewline -ForegroundColor Green
Write-Host ("=" * 99) -ForegroundColor Green
Write-Host "CONTINUING FROM STEP 5 - TRAINING & EVALUATION" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Green
Write-Host ("=" * 99) -ForegroundColor Green
Write-Host "Started at: $($StartTime.ToString('HH:mm:ss'))" -ForegroundColor Gray
Write-Host ""

# Step 5: Train OLD model (21-24)
Write-Host "[Step 5/9] Training OLD model (21-24 data)..." -ForegroundColor Yellow

$Step5Start = Get-Date
python train_margin.py --data data/games_train_with_players_90_norm.jsonl --output artifacts/margin_normalized --config configs/margin_default.yaml

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ FAILED at old model training" -ForegroundColor Red
    exit 1
}

$Step5Duration = (Get-Date) - $Step5Start
Write-Host "✓ Step 5 completed in $([math]::Round($Step5Duration.TotalSeconds, 1)) seconds" -ForegroundColor Green
Write-Host ""

# Step 6: Train NEW model (21-25)
Write-Host "[Step 6/9] Training NEW model (21-25 data)..." -ForegroundColor Yellow

$Step6Start = Get-Date
python train_expanded_model.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ FAILED at new model training" -ForegroundColor Red
    exit 1
}

$Step6Duration = (Get-Date) - $Step6Start
Write-Host "✓ Step 6 completed in $([math]::Round($Step6Duration.TotalSeconds, 1)) seconds" -ForegroundColor Green
Write-Host ""

Write-Host "=" -NoNewline -ForegroundColor Green
Write-Host ("=" * 99) -ForegroundColor Green
Write-Host "TRAINING COMPLETE - NOW GENERATING PREDICTIONS" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Green
Write-Host ("=" * 99) -ForegroundColor Green
Write-Host ""

# Step 7: Predict on 24-25 with both models
Write-Host "[Step 7/9] Generating predictions on 24-25 test set..." -ForegroundColor Yellow

$Step7Start = Get-Date
Write-Host "  -> OLD model..." -ForegroundColor Gray
python predict_margin.py --model artifacts/margin_normalized --data data/games_predict_2024_2025_with_players_norm.jsonl --output predictions/test_2425_OLD_fixed.csv

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ FAILED at old model 24-25 predictions" -ForegroundColor Red
    exit 1
}

Write-Host "  -> NEW model..." -ForegroundColor Gray
python predict_margin.py --model artifacts/margin_normalized_21_25 --data data/games_predict_2024_2025_with_players_norm.jsonl --output predictions/test_2425_NEW_fixed.csv

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ FAILED at new model 24-25 predictions" -ForegroundColor Red
    exit 1
}

$Step7Duration = (Get-Date) - $Step7Start
Write-Host "✓ Step 7 completed in $([math]::Round($Step7Duration.TotalSeconds, 1)) seconds" -ForegroundColor Green
Write-Host ""

# Step 8: Predict on 25-26 with both models
Write-Host "[Step 8/9] Generating predictions on 25-26 test set..." -ForegroundColor Yellow

$Step8Start = Get-Date
Write-Host "  -> OLD model..." -ForegroundColor Gray
python predict_margin.py --model artifacts/margin_normalized --data data/games_2025_2026_current_norm.jsonl --output predictions/test_2526_OLD_fixed.csv

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ FAILED at old model 25-26 predictions" -ForegroundColor Red
    exit 1
}

Write-Host "  -> NEW model..." -ForegroundColor Gray
python predict_margin.py --model artifacts/margin_normalized_21_25 --data data/games_2025_2026_current_norm.jsonl --output predictions/test_2526_NEW_fixed.csv

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ FAILED at new model 25-26 predictions" -ForegroundColor Red
    exit 1
}

$Step8Duration = (Get-Date) - $Step8Start
Write-Host "✓ Step 8 completed in $([math]::Round($Step8Duration.TotalSeconds, 1)) seconds" -ForegroundColor Green
Write-Host ""

# Step 9: Run comprehensive evaluation
Write-Host "[Step 9/9] Running comprehensive evaluation..." -ForegroundColor Yellow

$Step9Start = Get-Date
python evaluate_fixed_models.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ FAILED at evaluation" -ForegroundColor Red
    exit 1
}

$Step9Duration = (Get-Date) - $Step9Start
Write-Host "✓ Step 9 completed in $([math]::Round($Step9Duration.TotalSeconds, 1)) seconds" -ForegroundColor Green
Write-Host ""

# Summary
$TotalDuration = (Get-Date) - $StartTime

Write-Host "=" -NoNewline -ForegroundColor Green
Write-Host ("=" * 99) -ForegroundColor Green
Write-Host "ALL STEPS COMPLETE!" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Green
Write-Host ("=" * 99) -ForegroundColor Green
Write-Host ""
Write-Host "Total time: $([math]::Round($TotalDuration.TotalMinutes, 1)) minutes" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results saved to: predictions/model_comparison_fixed.csv" -ForegroundColor Cyan
Write-Host ""
Write-Host "Finished at: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
Write-Host "="*100 -ForegroundColor Green

