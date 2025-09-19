#!/usr/bin/env pwsh
# Start Multithreaded NBA Simulations on GCP
# Provides options for different scales of multithreaded execution

Write-Host "=" * 80
Write-Host "MULTITHREADED NBA SIMULATION LAUNCHER"
Write-Host "=" * 80

$BUCKET_NAME = (Get-Content "bucket_name.txt" -Raw).Trim()
Write-Host "Using bucket: $BUCKET_NAME"

Write-Host "`n🚀 MULTITHREADED SIMULATION OPTIONS:"
Write-Host "1 - Quick test (2 games, 4 runs each, 2 threads, 200 iterations)"
Write-Host "2 - Medium run (3 games, 8 runs each, 4 threads, 500 iterations)"
Write-Host "3 - Large scale (5 games, 12 runs each, 4 threads, 1000 iterations)"
Write-Host "4 - Custom configuration"

$choice = Read-Host "`nSelect option (1-4)"

switch ($choice) {
    "1" {
        $games = '"22200001","22200002"'
        $runs = 4
        $threads = 2
        $iterations = 200
        $description = "Quick multithreaded test"
    }
    "2" {
        $games = '"22200001","22200002","22200003"'
        $runs = 8
        $threads = 4
        $iterations = 500
        $description = "Medium multithreaded run"
    }
    "3" {
        $games = '"22200001","22200002","22200003","22200004","22200005"'
        $runs = 12
        $threads = 4
        $iterations = 1000
        $description = "Large scale multithreaded"
    }
    "4" {
        $games = Read-Host "Enter game IDs (comma-separated, quoted)"
        $runs = Read-Host "Runs per game"
        $threads = Read-Host "Number of threads (1-4 recommended)"
        $iterations = Read-Host "Max iterations per run"
        $description = "Custom multithreaded run"
    }
    default {
        Write-Host "Invalid option. Exiting."
        exit 1
    }
}

$total_simulations = ($games.Split(',').Count * $runs)
$estimated_time = ($total_simulations * $iterations * 0.1 / $threads) / 60  # Rough estimate

Write-Host "`n📊 SIMULATION PLAN:"
Write-Host "   Description: $description"
Write-Host "   Games: $games"
Write-Host "   Runs per game: $runs"
Write-Host "   Total simulations: $total_simulations"
Write-Host "   Threads: $threads"
Write-Host "   Iterations per run: $iterations"
Write-Host "   Estimated time: $([math]::Round($estimated_time, 1)) minutes"

$confirm = Read-Host "`nProceed with this configuration? (y/n)"
if ($confirm -ne "y") {
    Write-Host "Cancelled."
    exit 0
}

# Create custom config for this run
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$config = @{
    season_year = "2023-2024"
    games = $games.Split(',') | ForEach-Object { $_.Trim().Trim('"') }
    runs_per_game = [int]$runs
    max_iterations_per_run = [int]$iterations
    max_threads = [int]$threads
    skip_stage1 = $true
    platform = "gemini"
    output_db = "enhanced_simulation_results_multithreaded.db"
    log_level = "INFO"
    resume_on_error = $true
    timeout_minutes = 60
}

$configJson = $config | ConvertTo-Json -Depth 10
$configJson | Out-File -FilePath "temp_config.json" -Encoding UTF8

# Upload config to VM
gcloud compute scp temp_config.json nba-orchestrator:~/orchestrator_config_current.json --zone=us-central1-a --project=utopian-outlook-470922-q2 --scp-flag="-batch"
Remove-Item "temp_config.json"

Write-Host "`n🚀 Starting multithreaded simulation on GCP..."
Write-Host "This will run in the background. Use Ctrl+C to disconnect (simulation continues)."
Write-Host "Use ./monitor.ps1 to check progress."

# Start the simulation (enhanced multithreaded orchestrator)
gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="source ~/.bashrc && source ~/venv/bin/activate && cd ~ && nohup python3 enhanced_orchestrator.py --config orchestrator_config_current.json --threads $threads > big_run_multithreaded.log 2>&1 &"

Write-Host "`n✅ Multithreaded simulation started!"
Write-Host "📁 Output database: enhanced_simulation_results_multithreaded.db"
Write-Host "📋 Log file: big_run_multithreaded.log"
Write-Host "`n📊 To monitor progress:"
Write-Host "   ./monitor.ps1"
Write-Host "`n📥 To download results when complete:"
Write-Host "   gcloud compute scp nba-orchestrator:~/enhanced_simulation_results_multithreaded.db ./ --zone=us-central1-a --scp-flag='-batch'"
