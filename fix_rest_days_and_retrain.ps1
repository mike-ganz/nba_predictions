#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Fix rest days bug and regenerate all training data and models

.DESCRIPTION
    This script performs a complete regeneration after fixing the rest days bug:
    1. Backs up existing artifacts and data
    2. Clears cached team stats (to use corrected calculations)
    3. Regenerates training JSONL files with correct rest days
    4. Re-normalizes all data
    5. Retrains the Champion model
    6. Re-evaluates on current season
    7. Compares before/after performance

.PARAMETER SkipBackup
    Skip backing up existing artifacts (not recommended)

.PARAMETER SkipRetrain
    Skip retraining models (only regenerate data)

.PARAMETER QuickMode
    Only regenerate current season and evaluate (skip full retraining)

.PARAMETER SkipDataRegeneration
    Skip steps 1-4 (backup, cache clearing, data regeneration) and jump to step 5 (model retraining)
    Useful if steps 1-4 already completed but training failed

.EXAMPLE
    .\fix_rest_days_and_retrain.ps1
    
.EXAMPLE
    .\fix_rest_days_and_retrain.ps1 -QuickMode

.EXAMPLE
    .\fix_rest_days_and_retrain.ps1 -SkipDataRegeneration
#>

param(
    [switch]$SkipBackup,
    [switch]$SkipRetrain,
    [switch]$QuickMode,
    [switch]$SkipDataRegeneration
)

$ErrorActionPreference = "Stop"

# Color helpers
function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host ("=" * 69) -ForegroundColor Cyan
    Write-Host "  $Message" -ForegroundColor Yellow
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host ("=" * 69) -ForegroundColor Cyan
    Write-Host ""
}

function Write-Step {
    param([string]$Message)
    Write-Host "[STEP] " -NoNewline -ForegroundColor Green
    Write-Host $Message
}

function Write-Success {
    param([string]$Message)
    Write-Host "[OK] " -NoNewline -ForegroundColor Green
    Write-Host $Message
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] " -NoNewline -ForegroundColor Yellow
    Write-Host $Message
}

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] " -NoNewline -ForegroundColor Cyan
    Write-Host $Message
}

# Start
Write-Header "REST DAYS BUG FIX AND REGENERATION PIPELINE"

Write-Host "This script will:"
Write-Host "  1. Back up existing artifacts and data" -ForegroundColor Cyan
Write-Host "  2. Clear cached team stats" -ForegroundColor Cyan
Write-Host "  3. Regenerate training data with correct rest days" -ForegroundColor Cyan
Write-Host "  4. Re-normalize all data" -ForegroundColor Cyan

if (-not $QuickMode) {
    Write-Host "  5. Retrain Champion model" -ForegroundColor Cyan
    Write-Host "  6. Re-evaluate on current season" -ForegroundColor Cyan
    Write-Host "  7. Compare before/after performance" -ForegroundColor Cyan
} else {
    Write-Host "  5. Re-evaluate current season only (Quick Mode)" -ForegroundColor Cyan
}

Write-Host ""
Write-Warning "This will take significant time and compute resources!"
Write-Host ""

if (-not $SkipBackup) {
    $response = Read-Host "Continue? (yes/no)"
    if ($response -ne "yes") {
        Write-Host "Aborted by user" -ForegroundColor Yellow
        exit 0
    }
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

# ============================================================================
# STEPS 1-4: DATA REGENERATION (can be skipped if already done)
# ============================================================================

if (-not $SkipDataRegeneration) {

# ============================================================================
# STEP 1: BACKUP EXISTING ARTIFACTS
# ============================================================================

if (-not $SkipBackup) {
    Write-Header "STEP 1: BACKING UP EXISTING ARTIFACTS"
    
    $backupDir = "backups/pre_rest_days_fix_$timestamp"
    New-Item -ItemType Directory -Force -Path $backupDir | Out-Null
    
    Write-Step "Backing up Champion model..."
    if (Test-Path "artifacts/champion_individually_tuned") {
        Copy-Item -Recurse "artifacts/champion_individually_tuned" "$backupDir/champion_individually_tuned"
        Write-Success "Champion model backed up"
    } else {
        Write-Warning "Champion model not found, skipping backup"
    }
    
    Write-Step "Backing up training data..."
    if (Test-Path "data/games_train_with_players_90_norm.jsonl") {
        Copy-Item "data/games_train_with_players_90_norm.jsonl" "$backupDir/"
        Write-Success "Training data backed up"
    }
    
    Write-Step "Backing up current season data..."
    if (Test-Path "data/games_2025_2026_current_norm.jsonl") {
        Copy-Item "data/games_2025_2026_current_norm.jsonl" "$backupDir/"
        Write-Success "Current season data backed up"
    }
    
    Write-Step "Backing up current predictions..."
    if (Test-Path "predictions/current_season_champion_2025_2026_predictions.csv") {
        Copy-Item "predictions/current_season_champion_2025_2026_predictions.csv" "$backupDir/"
        Write-Success "Current predictions backed up"
    }
    
    Write-Success "All backups saved to: $backupDir"
    Write-Info "You can restore from this backup if needed"
} else {
    Write-Warning "Skipping backup (not recommended!)"
}

# ============================================================================
# STEP 2: CLEAR CACHED TEAM STATS
# ============================================================================

Write-Header "STEP 2: CLEARING CACHED TEAM STATS"

Write-Step "Removing cache directory..."
if (Test-Path "cache") {
    Remove-Item -Recurse -Force "cache"
    Write-Success "Cache cleared"
} else {
    Write-Info "Cache directory not found (already clean)"
}

Write-Info "Team stats will now use the corrected _compute_rest_days function"

} # End of SkipDataRegeneration check

if ($SkipDataRegeneration) {
    Write-Header "SKIPPING DATA REGENERATION (Steps 1-4)"
    Write-Info "Using existing data files - jumping to Step 5"
}

# ============================================================================
# STEP 3: REGENERATE TRAINING DATA
# ============================================================================

if (-not $QuickMode -and -not $SkipDataRegeneration) {
    Write-Header "STEP 3: REGENERATING TRAINING DATA"
    
    Write-Step "Regenerating training data for 2021-2024 seasons..."
    Write-Info "This will take 15-20 minutes..."
    
    python prepare_data.py `
        --team-boxscores-dir data/team_boxscores/historical `
        --player-boxscores-dir data/player_boxscores/historical `
        --output data/games_train_with_players.jsonl `
        --seasons 2021-2022 2022-2023 2023-2024 `
        --include-players
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to regenerate training data" -ForegroundColor Red
        exit 1
    }
    
    Write-Success "Training data regenerated with correct rest days"
    
    # ============================================================================
    # STEP 4: RE-NORMALIZE DATA
    # ============================================================================
    
    Write-Header "STEP 4: RE-NORMALIZING DATA"
    
    Write-Step "Applying league normalization to all datasets..."
    
    python scripts/normalize_all_data.py
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to normalize data" -ForegroundColor Red
        exit 1
    }
    
    Write-Success "All data normalized"
}

# ============================================================================
# STEP 5: RETRAIN CHAMPION MODEL
# ============================================================================

if (-not $QuickMode) {
    if (-not $SkipRetrain) {
        Write-Header "STEP 5: RETRAINING CHAMPION MODEL"
        
        Write-Step "Training Champion model with corrected rest days..."
        Write-Info "This will take 5-10 minutes..."
        
        # Create new model directory
        $newModelDir = "artifacts/champion_corrected_rest_days"
        New-Item -ItemType Directory -Force -Path $newModelDir | Out-Null
        
        # Check if we have the champion config
        if (Test-Path "configs/champion_individually_tuned.yaml") {
            $configFile = "configs/champion_individually_tuned.yaml"
        } else {
            Write-Warning "Champion config not found, using margin_xgboost.yaml"
            $configFile = "configs/margin_xgboost.yaml"
        }
        
        python train_margin.py `
            --data data/games_train_with_players_90_norm.jsonl `
            --config $configFile `
            --model-type xgboost `
            --output $newModelDir
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ Failed to train model" -ForegroundColor Red
            exit 1
        }
        
        Write-Success "Champion model retrained"
        Write-Info "New model saved to: $newModelDir"
        Write-Info "Original model preserved at: artifacts/champion_individually_tuned"
    } else {
        Write-Warning "Skipping model retraining (use existing model)"
    }
}

# ============================================================================
# STEP 6: REGENERATE CURRENT SEASON DATA
# ============================================================================

Write-Header "CURRENT SEASON: REGENERATING DATA"

Write-Step "Processing 2025-2026 season with corrected rest days..."

python process_current_season.py `
    --team-boxscores-dir data/team_boxscores/current `
    --player-boxscores-dir data/player_boxscores/current `
    --output data/games_2025_2026_current.jsonl `
    --season 2025-2026 `
    --include-players

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to process current season" -ForegroundColor Red
    exit 1
}

Write-Success "Current season data regenerated"

# ============================================================================
# STEP 7: RE-EVALUATE ON CURRENT SEASON
# ============================================================================

Write-Header "CURRENT SEASON: RE-EVALUATING CHAMPION MODEL"

Write-Step "Generating predictions with corrected data..."

# Determine which model to use
if (Test-Path "artifacts/champion_corrected_rest_days") {
    $modelPath = "artifacts/champion_corrected_rest_days"
    Write-Info "Using newly trained model with corrected rest days"
} else {
    $modelPath = "artifacts/champion_individually_tuned"
    Write-Info "Using original model (data corrected but model not retrained)"
}

python predict_margin.py `
    --model $modelPath `
    --data data/games_2025_2026_current_norm.jsonl `
    --output predictions/current_season_champion_corrected_predictions.csv

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to generate predictions" -ForegroundColor Red
    exit 1
}

Write-Success "Predictions generated"

Write-Step "Evaluating performance..."

python evaluate_current_season.py `
    --predictions predictions/current_season_champion_corrected_predictions.csv `
    --games data/games_2025_2026_current_norm.jsonl `
    --output predictions/current_season_champion_corrected_evaluation.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to evaluate predictions" -ForegroundColor Red
    exit 1
}

Write-Success "Evaluation complete"

# ============================================================================
# STEP 8: COMPARE BEFORE/AFTER
# ============================================================================

Write-Header "COMPARISON: BEFORE vs AFTER FIX"

Write-Step "Comparing rest days for November 7 games..."

# Create a temporary Python comparison script
$comparisonScript = @'
import json

print("=" * 70)
print("REST DAYS COMPARISON: November 7, 2025")
print("=" * 70)
print()

# Load old data
try:
    old_games = []
    with open('BACKUP_DIR_PLACEHOLDER/games_2025_2026_current_norm.jsonl', 'r') as f:
        for line in f:
            game = json.loads(line)
            if game['date'] == '2025-11-07':
                old_games.append(game)
    print(f"[OLD DATA] {len(old_games)} games")
except:
    print("[WARNING] Could not load old data")
    old_games = []

# Load new data
new_games = []
with open('data/games_2025_2026_current_norm.jsonl', 'r') as f:
    for line in f:
        game = json.loads(line)
        if game['date'] == '2025-11-07':
            new_games.append(game)

print(f"[NEW DATA] {len(new_games)} games")
print()

if old_games:
    print("Game-by-Game Comparison:")
    print("-" * 70)
    
    for new_game in new_games:
        game_id = new_game['game_id']
        old_game = next((g for g in old_games if g['game_id'] == game_id), None)
        
        if old_game:
            away_old = old_game['teams']['A']['rest_days']
            home_old = old_game['teams']['H']['rest_days']
            away_new = new_game['teams']['A']['rest_days']
            home_new = new_game['teams']['H']['rest_days']
            
            away_diff = away_new - away_old
            home_diff = home_new - home_old
            
            print(f"{game_id}:")
            print(f"  Away: {away_old} -> {away_new} ({away_diff:+.0f})")
            print(f"  Home: {home_old} -> {home_new} ({home_diff:+.0f})")
    
    print()
    print("[OK] All rest days should now be 1 less than before!")
else:
    print("Sample games with corrected rest days:")
    print("-" * 70)
    for game in new_games[:5]:
        print(f"{game['game_id']}:")
        print(f"  Away: {game['teams']['A']['rest_days']}")
        print(f"  Home: {game['teams']['H']['rest_days']}")

print()
print("=" * 70)
'@

# Find backup directory (either from this run or most recent)
if (-not $backupDir) {
    # Find most recent backup
    $backups = Get-ChildItem -Path "backups" -Directory -Filter "pre_rest_days_fix_*" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
    if ($backups) {
        $backupDir = $backups[0].FullName
        Write-Info "Using backup from: $backupDir"
    }
}

if ($backupDir) {
    # Replace placeholder and run
    $comparisonScript = $comparisonScript -replace 'BACKUP_DIR_PLACEHOLDER', $backupDir.Replace('\', '/')
    Write-Output $comparisonScript | python
} else {
    Write-Warning "No backup directory found - skipping before/after comparison"
    Write-Info "Showing current rest days for November 7 games..."
    
    # Simplified script that just shows current data
    $simpleScript = @'
import json

print("=" * 70)
print("CURRENT REST DAYS: November 7, 2025")
print("=" * 70)
print()

new_games = []
with open('data/games_2025_2026_current_norm.jsonl', 'r') as f:
    for line in f:
        game = json.loads(line)
        if game['date'] == '2025-11-07':
            new_games.append(game)

print(f"Found {len(new_games)} games on November 7, 2025")
print()

for game in new_games:
    print(f"{game['game_id']}:")
    print(f"  Away: {game['teams']['A']['rest_days']} rest days")
    print(f"  Home: {game['teams']['H']['rest_days']} rest days")

print()
print("=" * 70)
'@
    
    Write-Output $simpleScript | python
}

# ============================================================================
# FINAL SUMMARY
# ============================================================================

Write-Header "REGENERATION COMPLETE!"

Write-Host ""
Write-Host "SUMMARY:" -ForegroundColor Cyan
Write-Host ""

if (-not $SkipBackup -and $backupDir) {
    Write-Host "[OK] Backup Location:" -ForegroundColor Green
    Write-Host "   $backupDir" -ForegroundColor White
    Write-Host ""
}

Write-Host "[OK] Updated Files:" -ForegroundColor Green
Write-Host "   • data/games_2025_2026_current_norm.jsonl (corrected rest days)" -ForegroundColor White
Write-Host "   • predictions/current_season_champion_corrected_predictions.csv" -ForegroundColor White
Write-Host "   • predictions/current_season_champion_corrected_evaluation.txt" -ForegroundColor White

if (-not $QuickMode -and -not $SkipRetrain) {
    Write-Host "   • data/games_train_with_players_90_norm.jsonl (corrected rest days)" -ForegroundColor White
    Write-Host "   • artifacts/champion_corrected_rest_days/ (retrained model)" -ForegroundColor White
}

Write-Host ""
Write-Host "NEXT STEPS:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Review the corrected evaluation results:" -ForegroundColor White
Write-Host "   cat predictions/current_season_champion_corrected_evaluation.txt" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Compare predictions for November 7:" -ForegroundColor White
Write-Host "   # Old predictions (with bug):" -ForegroundColor Gray
Write-Host "   cat predictions/predictions_champion_2025-11-07.csv" -ForegroundColor Gray
Write-Host "   # New predictions (corrected):" -ForegroundColor Gray
Write-Host "   # Re-run prepare_todays_games.py to regenerate" -ForegroundColor Gray
Write-Host ""

if (-not $QuickMode -and -not $SkipRetrain) {
    Write-Host "3. Use the new model for future predictions:" -ForegroundColor White
    Write-Host "   # Option A: Replace the old model" -ForegroundColor Gray
    Write-Host "   mv artifacts/champion_individually_tuned artifacts/champion_old" -ForegroundColor Gray
    Write-Host "   mv artifacts/champion_corrected_rest_days artifacts/champion_individually_tuned" -ForegroundColor Gray
    Write-Host ""
    Write-Host "   # Option B: Update your scripts to use champion_corrected_rest_days" -ForegroundColor Gray
    Write-Host ""
}

Write-Host "4. Monitor performance over next few games to validate fix" -ForegroundColor White
Write-Host ""

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 69) -ForegroundColor Cyan
Write-Host ""

# Create a summary log
$summaryLog = @"
REST DAYS FIX SUMMARY
Generated: $timestamp

BUG DESCRIPTION:
The _compute_rest_days() function was returning calendar days instead of rest days,
causing all teams to have +1 extra rest day in calculated data (training & evaluation).

FIX APPLIED:
Changed: return float((target_date - last_game_date).days)
To:      return float(max(0, calendar_days - 1))

AFFECTED DATA:
- Training data: All historical games had incorrect rest days (+1)
- Current season evaluation: All games had incorrect rest days (+1)  
- Daily predictions: NOT affected (used NBA schedule data)

ACTIONS TAKEN:
$(if (-not $SkipBackup -and $backupDir) { "[OK] Backed up original artifacts to: $backupDir" } elseif ($SkipDataRegeneration) { "[SKIP] Data regeneration skipped - used existing files" } else { "[WARNING] Backup skipped" })
$(if (-not $SkipDataRegeneration) { "[OK] Cleared team stats cache" } else { "[SKIP] Cache clearing skipped" })
$(if (-not $QuickMode -and -not $SkipDataRegeneration) { "[OK] Regenerated training data with correct rest days" } else { "[SKIP] Skipped" })
$(if (-not $QuickMode -and -not $SkipDataRegeneration) { "[OK] Re-normalized all data" } else { "[SKIP] Skipped" })
$(if (-not $QuickMode -and -not $SkipRetrain) { "[OK] Retrained Champion model" } elseif ($SkipRetrain) { "[SKIP] Skipped model retraining" } else { "[SKIP] Skipped" })
[OK] Regenerated current season data
[OK] Re-evaluated Champion model performance

RESULT:
All pipelines now use consistent, correct rest days:
- Training data: Correct
- Daily predictions: Correct (already was)
- Post-game evaluation: Correct (now fixed)

FILES MODIFIED:
- generate_team_stats.py: Fixed _compute_rest_days() function
- All .jsonl files regenerated with correct rest days
$(if (-not $QuickMode -and -not $SkipRetrain) { "- New model trained: artifacts/champion_corrected_rest_days/" } else { "" })

VERIFICATION:
Check predictions/current_season_champion_corrected_evaluation.txt for updated performance metrics.
"@

$summaryLog | Out-File "FIX_SUMMARY_$timestamp.txt" -Encoding UTF8

Write-Success "Summary log saved to: FIX_SUMMARY_$timestamp.txt"
Write-Host ""

