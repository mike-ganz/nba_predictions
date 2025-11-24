#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Check prerequisites for model experiments

.DESCRIPTION
    Verifies all required data files exist before running experiments
#>

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "  EXPERIMENT PREREQUISITES CHECK" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""

$AllGood = $true

# Check historical team boxscores
Write-Host "[1/5] Checking historical team boxscores..." -ForegroundColor White
$ExpectedSeasons = @("2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025")
$BoxscoreDir = "data/team_boxscores/historical"

foreach ($season in $ExpectedSeasons) {
    $Pattern = "*$season*.xlsx"
    $Files = Get-ChildItem -Path $BoxscoreDir -Filter $Pattern -ErrorAction SilentlyContinue
    
    if ($Files.Count -gt 0) {
        Write-Host "  ✓ $season boxscore found" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $season boxscore MISSING" -ForegroundColor Red
        $AllGood = $false
    }
}
Write-Host ""

# Check historical player boxscores
Write-Host "[2/5] Checking historical player boxscores..." -ForegroundColor White
$PlayerBoxscoreDir = "data/player_boxscores/historical"

foreach ($season in $ExpectedSeasons) {
    $Pattern = "*$season*.xlsx"
    $Files = Get-ChildItem -Path $PlayerBoxscoreDir -Filter $Pattern -ErrorAction SilentlyContinue
    
    if ($Files.Count -gt 0) {
        Write-Host "  ✓ $season player data found" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $season player data MISSING" -ForegroundColor Red
        $AllGood = $false
    }
}
Write-Host ""

# Check current season data
Write-Host "[3/5] Checking current season data..." -ForegroundColor White
$CurrentSeasonFiles = @(
    "data/games_2025_2026_current.jsonl",
    "data/games_2025_2026_current_norm.jsonl"
)

foreach ($file in $CurrentSeasonFiles) {
    if (Test-Path $file) {
        $LineCount = (Get-Content $file).Count
        Write-Host "  ✓ $(Split-Path $file -Leaf) exists ($LineCount games)" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $(Split-Path $file -Leaf) MISSING" -ForegroundColor Red
        $AllGood = $false
    }
}
Write-Host ""

# Check current champion predictions
Write-Host "[4/5] Checking current champion predictions..." -ForegroundColor White
$ChampionPredictions = "predictions/current_season_champion_2025_2026_predictions.csv"

if (Test-Path $ChampionPredictions) {
    $Lines = (Get-Content $ChampionPredictions).Count
    Write-Host "  ✓ Champion predictions exist ($Lines rows)" -ForegroundColor Green
} else {
    Write-Host "  ⚠ Champion predictions not found (OK if first run)" -ForegroundColor Yellow
}
Write-Host ""

# Check disk space
Write-Host "[5/5] Checking disk space..." -ForegroundColor White
$Drive = (Get-Location).Drive
$FreeSpace = (Get-PSDrive $Drive.Name).Free / 1GB

if ($FreeSpace -gt 20) {
    Write-Host "  ✓ Sufficient disk space: $([math]::Round($FreeSpace, 1)) GB free" -ForegroundColor Green
} elseif ($FreeSpace -gt 10) {
    Write-Host "  ⚠ Limited disk space: $([math]::Round($FreeSpace, 1)) GB free" -ForegroundColor Yellow
} else {
    Write-Host "  ✗ Low disk space: $([math]::Round($FreeSpace, 1)) GB free (need 20+ GB)" -ForegroundColor Red
    $AllGood = $false
}
Write-Host ""

# Summary
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan

if ($AllGood) {
    Write-Host "  ✓ ALL PREREQUISITES MET - Ready to run experiments!" -ForegroundColor Green
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host ("=" * 79) -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Run experiments with:" -ForegroundColor White
    Write-Host "  .\run_model_experiments.ps1" -ForegroundColor Cyan
    Write-Host ""
    exit 0
} else {
    Write-Host "  ✗ MISSING PREREQUISITES - Cannot run experiments" -ForegroundColor Red
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host ("=" * 79) -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Fix issues above before running experiments" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

