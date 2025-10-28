# Regenerate training data with player availability features
# This will create new JSONL files that include player data

$StartTime = Get-Date

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "Regenerating Training Data with Player Availability" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "Started at: $($StartTime.ToString('HH:mm:ss'))" -ForegroundColor Gray
Write-Host ""

# Training data (2021-2022, 2022-2023, 2023-2024)
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "STEP 1/3: Generating training data (2021-2022, 2022-2023, 2023-2024)" -ForegroundColor Yellow
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "Expected: 5-10 minutes for ~3,700 games" -ForegroundColor Gray
Write-Host "Progress logged every 100 games..." -ForegroundColor Gray
Write-Host ""

$Step1Start = Get-Date
python prepare_data.py `
    --team-boxscores-dir data/team_boxscores/historical `
    --player-boxscores-dir data/player_boxscores/historical `
    --output data/games_train_with_players.jsonl `
    --seasons 2021-2022 2022-2023 2023-2024 `
    --include-players `
    --debug

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "✗ Failed to generate training data" -ForegroundColor Red
    exit 1
}

$Step1Duration = (Get-Date) - $Step1Start
Write-Host ""
Write-Host "✓ Step 1 completed in $([math]::Round($Step1Duration.TotalMinutes, 1)) minutes" -ForegroundColor Green
Write-Host ""

# Validation data (subset for quick checks)
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "STEP 2/3: Splitting into train 90% / validation 10%" -ForegroundColor Yellow
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host ""

$Step2Start = Get-Date
$pythonScript = @'
import json
import random
random.seed(42)
data = [json.loads(line) for line in open('data/games_train_with_players.jsonl')]
random.shuffle(data)
split = int(len(data) * 0.9)
print(f'Total games: {len(data)}')
print(f'Training: {split} games')
print(f'Validation: {len(data) - split} games')
with open('data/games_train_with_players_90.jsonl', 'w') as f:
    for r in data[:split]:
        f.write(json.dumps(r) + chr(10))
with open('data/games_val_with_players.jsonl', 'w') as f:
    for r in data[split:]:
        f.write(json.dumps(r) + chr(10))
'@

python -c $pythonScript

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "✗ Failed to split train/val data" -ForegroundColor Red
    exit 1
}

$Step2Duration = (Get-Date) - $Step2Start
Write-Host ""
Write-Host "✓ Step 2 completed in $([math]::Round($Step2Duration.TotalSeconds, 1)) seconds" -ForegroundColor Green
Write-Host ""

# 2024-2025 prediction data
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "STEP 3/3: Generating 2024-2025 prediction data" -ForegroundColor Yellow
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "Expected: 2-3 minutes for ~1,300 games" -ForegroundColor Gray
Write-Host "Progress logged every 100 games..." -ForegroundColor Gray
Write-Host ""

$Step3Start = Get-Date
python prepare_data.py `
    --team-boxscores-dir data/team_boxscores/historical `
    --player-boxscores-dir data/player_boxscores/historical `
    --output data/games_predict_2024_2025_with_players.jsonl `
    --seasons 2024-2025 `
    --include-players

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "✗ Failed to generate 2024-2025 data" -ForegroundColor Red
    exit 1
}

$Step3Duration = (Get-Date) - $Step3Start
Write-Host ""
Write-Host "✓ Step 3 completed in $([math]::Round($Step3Duration.TotalMinutes, 1)) minutes" -ForegroundColor Green
Write-Host ""

Write-Host ""
Write-Host "==============================================================" -ForegroundColor Green
Write-Host "✓ Data Generation Complete!" -ForegroundColor Green
Write-Host "==============================================================" -ForegroundColor Green

$TotalDuration = (Get-Date) - $StartTime
Write-Host ""
Write-Host "Total time: $([math]::Round($TotalDuration.TotalMinutes, 1)) minutes" -ForegroundColor Cyan
Write-Host "  Step 1 (Training data): $([math]::Round($Step1Duration.TotalMinutes, 1)) min" -ForegroundColor Gray
Write-Host "  Step 2 (Train/val split): $([math]::Round($Step2Duration.TotalSeconds, 1)) sec" -ForegroundColor Gray
Write-Host "  Step 3 (2024-2025 data): $([math]::Round($Step3Duration.TotalMinutes, 1)) min" -ForegroundColor Gray
Write-Host ""
Write-Host "Generated files:" -ForegroundColor Cyan
Write-Host "  - data/games_train_with_players_90.jsonl (training set, 90%)"
Write-Host "  - data/games_val_with_players.jsonl (validation set, 10%)"
Write-Host "  - data/games_predict_2024_2025_with_players.jsonl (2024-2025 for evaluation)"
Write-Host ""

# Show file sizes and game counts
Write-Host "File statistics:" -ForegroundColor Cyan
$trainFile = Get-Item "data/games_train_with_players_90.jsonl"
$valFile = Get-Item "data/games_val_with_players.jsonl"
$predFile = Get-Item "data/games_predict_2024_2025_with_players.jsonl"

$trainLines = (Get-Content "data/games_train_with_players_90.jsonl" | Measure-Object -Line).Lines
$valLines = (Get-Content "data/games_val_with_players.jsonl" | Measure-Object -Line).Lines
$predLines = (Get-Content "data/games_predict_2024_2025_with_players.jsonl" | Measure-Object -Line).Lines

Write-Host "  Training: $trainLines games ($([math]::Round($trainFile.Length/1MB, 1)) MB)"
Write-Host "  Validation: $valLines games ($([math]::Round($valFile.Length/1MB, 1)) MB)"
Write-Host "  Prediction: $predLines games ($([math]::Round($predFile.Length/1MB, 1)) MB)"
Write-Host ""

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Inspect a few games to verify player data:"
Write-Host "     python -c ""import json; print(json.dumps(json.loads(open('data/games_train_with_players_90.jsonl').readline()), indent=2))"""
Write-Host ""
Write-Host "  2. Retrain the model with player features:"
Write-Host "     python train.py --data data/games_train_with_players_90.jsonl --val data/games_val_with_players.jsonl --output artifacts/run_with_players"
Write-Host ""
Write-Host "  3. Evaluate on 2024-2025:"
Write-Host "     python evaluate.py --data data/games_predict_2024_2025_with_players.jsonl --model artifacts/run_with_players --reports-dir reports/run_2425_with_players --fast-eval --sharpen 0.9"
Write-Host ""
Write-Host "Finished at: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
Write-Host ""

