# Complete pipeline to fix data leakage and retrain both models
# This will take a while - expect 15-20 minutes total

$StartTime = Get-Date
$ErrorActionPreference = "Stop"

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 99) -ForegroundColor Cyan
Write-Host "DATA LEAKAGE FIX & COMPLETE RETRAINING PIPELINE" -ForegroundColor Cyan
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 99) -ForegroundColor Cyan
Write-Host "Started at: $($StartTime.ToString('HH:mm:ss'))" -ForegroundColor Gray
Write-Host ""

# Step 1: Regenerate training data (2021-2024)
Write-Host "[Step 1/9] Regenerating training data (2021-2024)..." -ForegroundColor Yellow
Write-Host "Expected: 5-10 minutes for ~3,700 games" -ForegroundColor Gray
Write-Host ""

$Step1Start = Get-Date
python -B .\prepare_data.py `
    --team-boxscores-dir data/team_boxscores/historical `
    --player-boxscores-dir data/player_boxscores/historical `
    --output data/games_train_with_players.jsonl `
    --seasons 2021-2022 2022-2023 2023-2024 `
    --include-players `
    --debug

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ FAILED at training data generation" -ForegroundColor Red
    exit 1
}

$Step1Duration = (Get-Date) - $Step1Start
Write-Host "✓ Step 1 completed in $([math]::Round($Step1Duration.TotalMinutes, 1)) minutes" -ForegroundColor Green
Write-Host ""

# Step 2: Split into 90% train / 10% validation
Write-Host "[Step 2/9] Splitting into train 90% / validation 10%..." -ForegroundColor Yellow

$Step2Start = Get-Date
$splitScript = @'
import json
import random
random.seed(42)
data = [json.loads(line) for line in open('data/games_train_with_players.jsonl', 'r', encoding='utf-8')]
random.shuffle(data)
split = int(len(data) * 0.9)
print(f'Total games: {len(data)}')
print(f'Training: {split} games')
print(f'Validation: {len(data) - split} games')
with open('data/games_train_with_players_90.jsonl', 'w', encoding='utf-8') as f:
    for r in data[:split]:
        f.write(json.dumps(r) + "\n")
with open('data/games_val_with_players.jsonl', 'w', encoding='utf-8') as f:
    for r in data[split:]:
        f.write(json.dumps(r) + "\n")
'@

$splitScript | python -B -

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ FAILED at train/val split" -ForegroundColor Red
    exit 1
}

$Step2Duration = (Get-Date) - $Step2Start
Write-Host "✓ Step 2 completed in $([math]::Round($Step2Duration.TotalSeconds, 1)) seconds" -ForegroundColor Green
Write-Host ""

# Step 3: Generate 2024-2025 test data
Write-Host "[Step 3/9] Regenerating 24-25 test data..." -ForegroundColor Yellow
Write-Host "Expected: 2-3 minutes for ~1,300 games" -ForegroundColor Gray
Write-Host ""

$Step3Start = Get-Date
python -B .\prepare_data.py `
    --team-boxscores-dir data/team_boxscores/historical `
    --player-boxscores-dir data/player_boxscores/historical `
    --output data/games_predict_2024_2025_with_players.jsonl `
    --seasons 2024-2025 `
    --include-players `
    --debug

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ FAILED at 2024-2025 data generation" -ForegroundColor Red
    exit 1
}

$Step3Duration = (Get-Date) - $Step3Start
Write-Host "✓ Step 3 completed in $([math]::Round($Step3Duration.TotalMinutes, 1)) minutes" -ForegroundColor Green
Write-Host ""

# Step 4: Apply league normalization
Write-Host "[Step 4/9] Applying league normalization..." -ForegroundColor Yellow

$Step4Start = Get-Date
python scripts/normalize_all_data.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ FAILED at normalization" -ForegroundColor Red
    exit 1
}

$Step4Duration = (Get-Date) - $Step4Start
Write-Host "✓ Step 4 completed in $([math]::Round($Step4Duration.TotalSeconds, 1)) seconds" -ForegroundColor Green
Write-Host ""

Write-Host "=" -NoNewline -ForegroundColor Green
Write-Host ("=" * 99) -ForegroundColor Green
Write-Host "DATA REGENERATION COMPLETE - NOW TRAINING MODELS" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Green
Write-Host ("=" * 99) -ForegroundColor Green
Write-Host ""

# Step 5: Train OLD model (21-24)
Write-Host "[Step 5/9] Training OLD model (21-24 data)..." -ForegroundColor Yellow
Write-Host "Expected: 1-2 minutes" -ForegroundColor Gray
Write-Host ""

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
Write-Host "Expected: 1-2 minutes" -ForegroundColor Gray
Write-Host ""

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
Write-Host "Step durations:" -ForegroundColor Gray
Write-Host "  1. Training data:     $([math]::Round($Step1Duration.TotalMinutes, 1)) min" -ForegroundColor Gray
Write-Host "  2. Train/val split:   $([math]::Round($Step2Duration.TotalSeconds, 1)) sec" -ForegroundColor Gray
Write-Host "  3. 24-25 test data:   $([math]::Round($Step3Duration.TotalMinutes, 1)) min" -ForegroundColor Gray
Write-Host "  4. Normalization:     $([math]::Round($Step4Duration.TotalSeconds, 1)) sec" -ForegroundColor Gray
Write-Host "  5. OLD model train:   $([math]::Round($Step5Duration.TotalSeconds, 1)) sec" -ForegroundColor Gray
Write-Host "  6. NEW model train:   $([math]::Round($Step6Duration.TotalSeconds, 1)) sec" -ForegroundColor Gray
Write-Host "  7. 24-25 predictions: $([math]::Round($Step7Duration.TotalSeconds, 1)) sec" -ForegroundColor Gray
Write-Host "  8. 25-26 predictions: $([math]::Round($Step8Duration.TotalSeconds, 1)) sec" -ForegroundColor Gray
Write-Host "  9. Evaluation:        $([math]::Round($Step9Duration.TotalSeconds, 1)) sec" -ForegroundColor Gray
Write-Host ""
Write-Host "Results saved to: predictions/model_comparison_fixed.csv" -ForegroundColor Cyan
Write-Host ""
Write-Host "Finished at: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
Write-Host "="*100 -ForegroundColor Green
