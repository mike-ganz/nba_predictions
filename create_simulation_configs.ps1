#!/usr/bin/env pwsh
# Create Different Simulation Configuration Templates

Write-Host "=" * 80
Write-Host "NBA SIMULATION CONFIGURATION GENERATOR"
Write-Host "=" * 80

Write-Host "`n📋 Choose configuration template:"
Write-Host "1 - Single thread (1 thread, good for testing)"
Write-Host "2 - Multithreaded (4 threads, fastest)"
Write-Host "3 - Research grade (many runs, high quality)"
Write-Host "4 - Quick batch (many games, few runs each)"
Write-Host "5 - Custom configuration"

$choice = Read-Host "`nSelect template (1-5)"

switch ($choice) {
    "1" {
        $config = @{
            season_year = "2023-2024"
            games = @("22200001", "22200002", "22200003")
            runs_per_game = 5
            max_iterations_per_run = 800
            max_threads = 1
            skip_stage1 = $true
            platform = "gemini"
            output_db = "simulation_results_single_thread.db"
            log_level = "INFO"
            resume_on_error = $true
            timeout_minutes = 45
        }
        $filename = "config_single_thread.json"
        $description = "Single Thread Configuration"
    }
    "2" {
        $config = @{
            season_year = "2023-2024"
            games = @("22200001", "22200002", "22200003", "22200004")
            runs_per_game = 8
            max_iterations_per_run = 1000
            max_threads = 4
            skip_stage1 = $true
            platform = "gemini"
            output_db = "simulation_results_multithreaded.db"
            log_level = "INFO"
            resume_on_error = $true
            timeout_minutes = 60
        }
        $filename = "config_multithreaded.json"
        $description = "4-Thread Multithreaded Configuration"
    }
    "3" {
        $config = @{
            season_year = "2023-2024"
            games = @("22200001", "22200002", "22200003")
            runs_per_game = 25
            max_iterations_per_run = 2000
            max_threads = 4
            skip_stage1 = $true
            platform = "gemini"
            output_db = "simulation_results_research.db"
            log_level = "WARNING"
            resume_on_error = $true
            timeout_minutes = 120
        }
        $filename = "config_research.json"
        $description = "Research Grade Configuration (High Quality)"
    }
    "4" {
        $config = @{
            season_year = "2023-2024"
            games = @("22200001", "22200002", "22200003", "22200004", "22200005", "22200006", "22200007", "22200008")
            runs_per_game = 3
            max_iterations_per_run = 600
            max_threads = 4
            skip_stage1 = $true
            platform = "gemini"
            output_db = "simulation_results_batch.db"
            log_level = "WARNING"
            resume_on_error = $true
            timeout_minutes = 45
        }
        $filename = "config_quick_batch.json"
        $description = "Quick Batch Configuration (Many Games, Fast)"
    }
    "5" {
        Write-Host "`n🛠️  CUSTOM CONFIGURATION"
        $games_input = Read-Host "Enter game IDs (comma-separated): "
        $games_array = $games_input.Split(',') | ForEach-Object { $_.Trim() }
        
        $runs = Read-Host "Runs per game (1-50)"
        $iterations = Read-Host "Max iterations per run (100-3000)"
        $threads = Read-Host "Number of threads (1-4)"
        $timeout = Read-Host "Timeout minutes (30-120)"
        $log_level = Read-Host "Log level (INFO/DEBUG/WARNING)"
        
        $config = @{
            season_year = "2023-2024"
            games = $games_array
            runs_per_game = [int]$runs
            max_iterations_per_run = [int]$iterations
            max_threads = [int]$threads
            skip_stage1 = $true
            platform = "gemini"
            output_db = "simulation_results_custom.db"
            log_level = $log_level.ToUpper()
            resume_on_error = $true
            timeout_minutes = [int]$timeout
        }
        $filename = "config_custom.json"
        $description = "Custom Configuration"
    }
    default {
        Write-Host "Invalid option. Exiting."
        exit 1
    }
}

# Calculate estimates
$total_simulations = $config.games.Count * $config.runs_per_game
$estimated_time_minutes = [math]::Round(($total_simulations * $config.max_iterations_per_run * 0.08 / $config.max_threads) / 60, 1)

Write-Host "`n📊 CONFIGURATION SUMMARY:"
Write-Host "   Description: $description"
Write-Host "   Games: $($config.games.Count) games"
Write-Host "   Total simulations: $total_simulations"
Write-Host "   Threads: $($config.max_threads)"
Write-Host "   Estimated time: $estimated_time_minutes minutes"

# Save configuration
$config | ConvertTo-Json -Depth 10 | Out-File -FilePath $filename -Encoding UTF8

Write-Host "`n✅ Configuration saved to: $filename"
Write-Host "`n🚀 To run this configuration:"
Write-Host "   gcloud compute scp $filename nba-orchestrator:~/ --zone=us-central1-a --project=utopian-outlook-470922-q2 --scp-flag='-batch'"
Write-Host "   gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag='-batch' --command='source ~/venv/bin/activate && python3 orchestrator.py --config $filename'"

Write-Host "`n📋 Or use the run_simulation_config.ps1 script (if you create it below):"
Write-Host "   .\run_simulation_config.ps1 $filename"
