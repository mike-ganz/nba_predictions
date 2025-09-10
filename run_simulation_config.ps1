#!/usr/bin/env pwsh
# Run NBA Simulation with Configuration File

param(
    [Parameter(Mandatory=$true)]
    [string]$ConfigFile
)

Write-Host "=" * 80
Write-Host "RUNNING NBA SIMULATION WITH CONFIG FILE"
Write-Host "=" * 80

# Check if config file exists
if (-not (Test-Path $ConfigFile)) {
    Write-Host "❌ Configuration file not found: $ConfigFile"
    exit 1
}

# Load and display config
$config = Get-Content $ConfigFile | ConvertFrom-Json
Write-Host "`n📋 CONFIGURATION:"
Write-Host "   File: $ConfigFile"
Write-Host "   Games: $($config.games.Count) games ($($config.games -join ', '))"
Write-Host "   Runs per game: $($config.runs_per_game)"
Write-Host "   Max iterations: $($config.max_iterations_per_run)"
Write-Host "   Threads: $($config.max_threads)"
Write-Host "   Platform: $($config.platform)"
Write-Host "   Output database: $($config.output_db)"

# Calculate estimates
$total_simulations = $config.games.Count * $config.runs_per_game
$estimated_time_minutes = [math]::Round(($total_simulations * $config.max_iterations_per_run * 0.08 / $config.max_threads) / 60, 1)

Write-Host "`n📊 ESTIMATES:"
Write-Host "   Total simulations: $total_simulations"
Write-Host "   Estimated time: $estimated_time_minutes minutes"
Write-Host "   Estimated cost: ~$([math]::Round($estimated_time_minutes * 0.05, 2)) (VM + API calls)"

# Check prerequisites
$BUCKET_NAME = (Get-Content "bucket_name.txt" -Raw).Trim()
$vmStatus = gcloud compute instances list --filter="name:nba-orchestrator" --format="value(status)" --project=utopian-outlook-470922-q2 2>$null

if ($vmStatus -ne "RUNNING") {
    Write-Host "`n❌ VM is not running. Starting VM..."
    gcloud compute instances start nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2
    Write-Host "   Waiting for VM to start..."
    Start-Sleep 30
}

# Confirm execution
$confirm = Read-Host "`nProceed with simulation? (y/n)"
if ($confirm -ne "y") {
    Write-Host "Cancelled."
    exit 0
}

# Upload config to VM
Write-Host "`n📤 Uploading configuration to VM..."
gcloud compute scp $ConfigFile nba-orchestrator:~/current_config.json --zone=us-central1-a --project=utopian-outlook-470922-q2 --scp-flag="-batch"

# Enable multithreading if not already done and config uses multiple threads
if ($config.max_threads -gt 1) {
    Write-Host "`n⚡ Ensuring multithreading is enabled..."
    gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="if [ ! -f orchestrator_multithreaded.py ]; then echo 'Multithreading not enabled. Run enable_multithreading.ps1 first.'; fi"
}

# Start simulation
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
Write-Host "`n🚀 Starting simulation..."
Write-Host "   Timestamp: $timestamp"
Write-Host "   Log file: simulation_$timestamp.log"
Write-Host "   Database: $($config.output_db)"

# Run in background
$sshCommand = 'source ~/.bashrc && source ~/venv/bin/activate && cd ~ && nohup python3 orchestrator.py --config current_config.json > simulation_' + $timestamp + '.log 2>&1 &'
gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command=$sshCommand

Write-Host "`n✅ SIMULATION STARTED!"
Write-Host "   Running in background on VM"
Write-Host "   Use Ctrl+C to disconnect (simulation continues)"

Write-Host "`n📊 MONITORING COMMANDS:"
Write-Host "   Monitor dashboard: ./monitor.ps1"
Write-Host "   Live logs: gcloud compute ssh nba-orchestrator --zone=us-central1-a --ssh-flag='-batch' --command='tail -f simulation_$timestamp.log'"
Write-Host "   Check progress: gcloud compute ssh nba-orchestrator --zone=us-central1-a --ssh-flag='-batch' --command='ps aux | grep orchestrator'"

Write-Host "`n📥 WHEN COMPLETE - DOWNLOAD RESULTS:"
Write-Host "   Database: gcloud compute scp nba-orchestrator:~/$($config.output_db) ./ --zone=us-central1-a --scp-flag='-batch'"
Write-Host "   Logs: gcloud compute scp nba-orchestrator:~/simulation_$timestamp.log ./ --zone=us-central1-a --scp-flag='-batch'"

Write-Host "`n📋 ANALYSIS COMMANDS:"
Write-Host "   Run analysis: ./run_analysis_on_gcp.ps1"
Write-Host "   Local analysis: python results_analysis.py (after downloading database)"
