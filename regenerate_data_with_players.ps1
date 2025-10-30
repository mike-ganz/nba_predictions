# Regenerate training data with player availability features
# This will create new JSONL files that include player data

$StartTime = Get-Date
$ErrorActionPreference = "Stop"

# Ensure we run from the repository root (script directory)
try {
    Set-Location -Path $PSScriptRoot
} catch {}

# Robust script root fallback and scripts dir setup
if (-not $PSScriptRoot) {
    $PSScriptRoot = (Resolve-Path ".").Path
}
$scriptsDir = Join-Path $PSScriptRoot "scripts"
New-Item -ItemType Directory -Force -Path $scriptsDir | Out-Null

# Show which Python is being used
$pythonExe = (Get-Command python).Source
$pythonVer = (& python --version)
Write-Host "Using Python: $pythonVer ($pythonExe)" -ForegroundColor Gray

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
python -B .\prepare_data.py `
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
# Write the splitter script to a temp file to avoid -c quoting issues
$splitScriptPath = Join-Path $scriptsDir "split_train_val_tmp.py"
$pythonScript = @'
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
Set-Content -Path $splitScriptPath -Value $pythonScript -Encoding UTF8

python -B $splitScriptPath

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "✗ Failed to split train/val data" -ForegroundColor Red
    exit 1
}

# Verify split outputs exist
if (-not (Test-Path "data/games_train_with_players_90.jsonl") -or -not (Test-Path "data/games_val_with_players.jsonl")) {
    Write-Host "✗ Split did not produce expected files (train_90 and/or val missing)" -ForegroundColor Red
    exit 1
}

# Clean up temp script
Remove-Item -Force $splitScriptPath -ErrorAction SilentlyContinue

$Step2Duration = (Get-Date) - $Step2Start
Write-Host ""
Write-Host "✓ Step 2 completed in $([math]::Round($Step2Duration.TotalSeconds, 1)) seconds" -ForegroundColor Green
Write-Host ""

# Quick sanity check: ensure player usage is not stuck at 0.2
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "SANITY CHECK: Verifying player usage distribution (training set)" -ForegroundColor Yellow
Write-Host "==============================================================" -ForegroundColor Cyan

$usageCheck = @'
import json, itertools, os, sys

path = 'data/games_train_with_players_90.jsonl'
if not os.path.exists(path):
    print('MISSING_TRAIN_90_FILE')
    sys.exit(0)

def iterate_usages(path, max_games=200):
    n = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            g = json.loads(line)
            if 'players' not in g:
                continue
            for side in ('A','H'):
                for p in g['players'].get(side, []):
                    yield p.get('baseline_usage_rate')
            n += 1
            if n >= max_games:
                break

rates = list(itertools.islice(iterate_usages(path), 0, 5000))
uniq = sorted(set(rates))
print('Sampled players:', len(rates))
print('Unique usage values (first 10):', uniq[:10])
if rates and all(abs(r - 0.2) < 1e-9 for r in rates):
    print('ALL_USAGE_0_2')
'@

$usageScriptPath = Join-Path $scriptsDir "usage_check_tmp.py"
Set-Content -Path $usageScriptPath -Value $usageCheck -Encoding UTF8
$checkOutput = & python -B $usageScriptPath | Out-String
Write-Host $checkOutput.Trim() -ForegroundColor Gray
if ($checkOutput -match 'MISSING_TRAIN_90_FILE') {
    Write-Host "✗ Train split file not found: data/games_train_with_players_90.jsonl" -ForegroundColor Red
    Write-Host "  Split step likely failed (quoting/encoding)." -ForegroundColor Red
    exit 1
}
if ($checkOutput -match 'ALL_USAGE_0_2') {
    Write-Host "✗ Detected all player usage rates = 0.2 in training set." -ForegroundColor Red
    Write-Host "  This usually means the wrong Python or working directory was used, or player data wasn't loaded." -ForegroundColor Red
    Write-Host "  Confirm you're running from the repo root and the Excel files exist under data/player_boxscores/historical." -ForegroundColor Red
    exit 1
}
Remove-Item -Force $usageScriptPath -ErrorAction SilentlyContinue

# 2024-2025 prediction data
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "STEP 3/3: Generating 2024-2025 prediction data" -ForegroundColor Yellow
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "Expected: 2-3 minutes for ~1,300 games" -ForegroundColor Gray
Write-Host "Progress logged every 100 games..." -ForegroundColor Gray
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
if (Test-Path "data/games_train_with_players_90.jsonl") {
$trainFile = Get-Item "data/games_train_with_players_90.jsonl"
    $trainLines = (Get-Content "data/games_train_with_players_90.jsonl" | Measure-Object -Line).Lines
    Write-Host "  Training: $trainLines games ($([math]::Round($trainFile.Length/1MB, 1)) MB)"
} else {
    Write-Host "  Training: 0 games (0 MB)" -ForegroundColor Yellow
}
if (Test-Path "data/games_val_with_players.jsonl") {
$valFile = Get-Item "data/games_val_with_players.jsonl"
    $valLines = (Get-Content "data/games_val_with_players.jsonl" | Measure-Object -Line).Lines
    Write-Host "  Validation: $valLines games ($([math]::Round($valFile.Length/1MB, 1)) MB)"
} else {
    Write-Host "  Validation: 0 games (0 MB)" -ForegroundColor Yellow
}
if (Test-Path "data/games_predict_2024_2025_with_players.jsonl") {
$predFile = Get-Item "data/games_predict_2024_2025_with_players.jsonl"
$predLines = (Get-Content "data/games_predict_2024_2025_with_players.jsonl" | Measure-Object -Line).Lines
Write-Host "  Prediction: $predLines games ($([math]::Round($predFile.Length/1MB, 1)) MB)"
} else {
    Write-Host "  Prediction: 0 games (0 MB)" -ForegroundColor Yellow
}
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

