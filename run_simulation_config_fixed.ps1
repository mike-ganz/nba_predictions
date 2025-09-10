#!/usr/bin/env pwsh
# Run NBA Simulation with Configuration File - FIXED VERSION

param(
    [Parameter(Mandatory=$true)]
    [string]$ConfigFile
)

Write-Host ("=" * 80)
Write-Host "RUNNING NBA SIMULATION WITH CONFIG FILE"
Write-Host ("=" * 80)

# Check if config file exists
if (-not (Test-Path $ConfigFile)) {
    Write-Host "ERROR: Configuration file not found: $ConfigFile" -ForegroundColor Red
    exit 1
}

# Load and display config
$config = Get-Content $ConfigFile | ConvertFrom-Json
Write-Host "`nCONFIGURATION:" -ForegroundColor Cyan
Write-Host "   File: $ConfigFile"
Write-Host "   Games: $($config.games.Count) games"
Write-Host "   First few games: $($config.games[0..4] -join ', ')..."
Write-Host "   Runs per game: $($config.runs_per_game)"
Write-Host "   Max iterations: $($config.max_iterations_per_run)"
Write-Host "   Threads: $($config.max_threads)"
Write-Host "   Platform: $($config.platform)"
Write-Host "   Output database: $($config.output_db)"

# Calculate estimates
$total_simulations = $config.games.Count * $config.runs_per_game
$estimated_time_minutes = [math]::Round(($total_simulations * $config.max_iterations_per_run * 0.08 / $config.max_threads) / 60, 1)
$estimated_cost = [math]::Round($estimated_time_minutes * 0.05, 2)

Write-Host "`nESTIMATES:" -ForegroundColor Yellow
Write-Host "   Total simulations: $total_simulations"
Write-Host "   Estimated time: $estimated_time_minutes minutes"
Write-Host "   Estimated cost: ~$($estimated_cost) (VM + API calls)"

# Check prerequisites
$BUCKET_NAME = (Get-Content "bucket_name.txt" -Raw).Trim()
$vmStatus = gcloud compute instances list --filter="name:nba-orchestrator" --format="value(status)" --project=utopian-outlook-470922-q2 2>$null

if ($vmStatus -ne "RUNNING") {
    Write-Host "`nVM is not running. Starting VM..." -ForegroundColor Yellow
    gcloud compute instances start nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2
    Write-Host "   Waiting for VM to start..."
    Start-Sleep 30
}

# Confirm execution
$confirm = Read-Host "`nProceed with simulation? (y/n)"
if ($confirm -ne "y") {
    Write-Host "Cancelled." -ForegroundColor Yellow
    exit 0
}

# Upload config to VM
Write-Host "`nUploading configuration to VM..." -ForegroundColor Cyan
gcloud compute scp $ConfigFile nba-orchestrator:current_config.json --zone=us-central1-a --project=utopian-outlook-470922-q2 --scp-flag="-batch"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to upload config file" -ForegroundColor Red
    exit 1
}

Write-Host "SUCCESS: Configuration uploaded" -ForegroundColor Green

# Start simulation
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
Write-Host "`nStarting simulation..." -ForegroundColor Cyan
Write-Host "   Timestamp: $timestamp"
Write-Host "   Log file: simulation_$timestamp.log"
Write-Host "   Database: $($config.output_db)"

# Create the SSH command as a single string
$logFile = "simulation_$timestamp.log"
$sshCommand = "source ~/.bashrc && source ~/venv/bin/activate && cd ~ && nohup python3 orchestrator.py --config current_config.json > $logFile 2>&1 &"

Write-Host "`nExecuting on VM..." -ForegroundColor Yellow
gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command=$sshCommand

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nSUCCESS: SIMULATION STARTED!" -ForegroundColor Green
} else {
    Write-Host "`nERROR: Failed to start simulation" -ForegroundColor Red
    exit 1
}

Write-Host "   Running in background on VM"
Write-Host "   Use Ctrl+C to disconnect (simulation continues)"

Write-Host "`nMONITORING COMMANDS:" -ForegroundColor Cyan
Write-Host "   Monitor dashboard: .\monitor_fixed.ps1"
Write-Host "   Live logs: gcloud compute ssh nba-orchestrator --zone=us-central1-a --ssh-flag='-batch' --command='tail -f $logFile'"
Write-Host "   Check progress: gcloud compute ssh nba-orchestrator --zone=us-central1-a --ssh-flag='-batch' --command='ps aux | grep orchestrator'"

Write-Host "`nWHEN COMPLETE - DOWNLOAD RESULTS:" -ForegroundColor Cyan
Write-Host "   Database: gcloud compute scp nba-orchestrator:$($config.output_db) ./ --zone=us-central1-a --scp-flag='-batch'"
Write-Host "   Logs: gcloud compute scp nba-orchestrator:$logFile ./ --zone=us-central1-a --scp-flag='-batch'"

Write-Host "`nANALYSIS COMMANDS:" -ForegroundColor Cyan
Write-Host "   Run analysis: .\run_actual_results_analysis.ps1"
Write-Host "   Local analysis: python results_analysis.py (after downloading database)"

Write-Host "`n" + ("=" * 80)
Write-Host "SIMULATION LAUNCHED SUCCESSFULLY!" -ForegroundColor Green
Write-Host ("=" * 80)
