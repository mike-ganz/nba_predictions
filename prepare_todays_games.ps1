# Prepare game data for today's NBA games
# This is a convenience wrapper around prepare_future_games.py

param(
    [string]$Date = "",  # Optional: specify date in YYYY-MM-DD format, defaults to today
    [string]$Season = "2025-2026",  # Season
    [switch]$NoPlayers,  # Exclude player data (faster)
    [switch]$Debug  # Enable debug logging
)

# Use today's date if not specified
if ($Date -eq "") {
    $Date = Get-Date -Format "yyyy-MM-dd"
    Write-Host "Using today's date: $Date"
} else {
    Write-Host "Using specified date: $Date"
}

# Build output filename
$OutputFile = "data/games_future_$Date.jsonl"

Write-Host ""
Write-Host "======================================================================"
Write-Host "Preparing game data for $Date"
Write-Host "======================================================================"
Write-Host ""

# Build command arguments
$args = @(
    "prepare_future_games.py",
    "--schedule", "data/schedules/current/2025-2026_NBA_Regular_Season_Schedule_Updated.xlsx",
    "--injuries", "data/injuries/current_injuries.json",
    "--market", "data/market/current_spreads.json",
    "--output", $OutputFile,
    "--season", $Season,
    "--start-date", $Date,
    "--end-date", $Date
)

# Add optional flags
if (-not $NoPlayers) {
    $args += "--include-players"
}

if ($Debug) {
    $args += "--debug"
}

# Run the pipeline
python @args

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "======================================================================"
    Write-Host "SUCCESS: Game data prepared for $Date"
    Write-Host "======================================================================"
    Write-Host ""
    Write-Host "Output file: $OutputFile"
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "  1. Review the prepared game data"
    Write-Host "  2. Generate predictions:"
    Write-Host ""
    Write-Host "     python predict_margin.py \"
    Write-Host "       --data $OutputFile \"
    Write-Host "       --model artifacts/margin_normalized \"
    Write-Host "       --output predictions/predictions_$Date.csv"
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "ERROR: Failed to prepare game data" -ForegroundColor Red
    Write-Host ""
    Write-Host "Common issues:"
    Write-Host "  - Missing market data for games on $Date"
    Write-Host "  - Update data/market/current_spreads.json with today's lines"
    Write-Host ""
    exit 1
}

