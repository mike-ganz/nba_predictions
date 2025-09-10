# NBA Predictions Simulation Starter
# Starts full simulation runs on GCP

Write-Host "NBA Predictions Simulation Starter" -ForegroundColor Green

# Check prerequisites
Write-Host "`nChecking prerequisites..." -ForegroundColor Cyan

# Check bucket name
if (Test-Path "bucket_name.txt") {
    $BUCKET_NAME = Get-Content "bucket_name.txt" -Raw
    $BUCKET_NAME = $BUCKET_NAME.Trim()
    Write-Host "SUCCESS: Bucket: $BUCKET_NAME" -ForegroundColor Green
} else {
    Write-Host "ERROR: bucket_name.txt not found! Please run setup scripts first." -ForegroundColor Red
    exit 1
}

# Check VM status
$vmStatus = gcloud compute instances list --filter="name:nba-orchestrator" --format="value(status)" --project=utopian-outlook-470922-q2 2>$null
if ($vmStatus -eq "RUNNING") {
    Write-Host "SUCCESS: VM is running" -ForegroundColor Green
} else {
    Write-Host "ERROR: VM is not running: $vmStatus" -ForegroundColor Red
    Write-Host "Start VM with: gcloud compute instances start nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2" -ForegroundColor Yellow
    exit 1
}

# Simulation options
Write-Host "`n🎮 Simulation Options:" -ForegroundColor Cyan
Write-Host "1 - Quick test (3 games, 2 runs each, 100 iterations)" -ForegroundColor White
Write-Host "2 - Medium run (5 games, 5 runs each, 500 iterations)" -ForegroundColor White
Write-Host "3 - Full simulation (from config file - orchestrator_config.json)" -ForegroundColor White
Write-Host "4 - Custom simulation" -ForegroundColor White

$choice = Read-Host "`nChoose simulation type (1-4)"

switch ($choice) {
    "1" {
        Write-Host "`nStarting Quick Test..." -ForegroundColor Green
        $games = "22200001,22200002,22200003"
        $runsPerGame = 2
        $maxIterations = 100
        $description = "Quick Test"
    }
    "2" {
        Write-Host "`nStarting Medium Run..." -ForegroundColor Green
        $games = "22200001,22200002,22200003,22200004,22200005"
        $runsPerGame = 5
        $maxIterations = 500
        $description = "Medium Run"
    }
    "3" {
        Write-Host "`nStarting Full Simulation from Config..." -ForegroundColor Green
        if (Test-Path "orchestrator_config.json") {
            # Copy config to VM first
            gcloud compute scp orchestrator_config.json nba-orchestrator:~/ --zone=us-central1-a --project=utopian-outlook-470922-q2 --scp-flag="-batch"
            $useConfigFile = $true
            $description = "Full Simulation (Config)"
        } else {
            Write-Host "ERROR: orchestrator_config.json not found!" -ForegroundColor Red
            exit 1
        }
    }
    "4" {
        Write-Host "`n🎮 Custom Simulation Setup:" -ForegroundColor Cyan
        $games = Read-Host "Enter game IDs (comma-separated, e.g., 22200001,22200002)"
        $runsPerGame = [int](Read-Host "Enter runs per game (e.g., 5)")
        $maxIterations = [int](Read-Host "Enter max iterations per run (e.g., 2000)")
        $description = "Custom Simulation"
    }
    default {
        Write-Host "ERROR: Invalid choice" -ForegroundColor Red
        exit 1
    }
}

# Start simulation based on choice
if ($choice -eq "3") {
    # Use config file
    Write-Host "`nStarting simulation with config file..." -ForegroundColor Cyan
    $simulationCommand = "
source ~/.bashrc &&
source ~/venv/bin/activate &&
cd ~ &&
nohup python3 orchestrator.py --config orchestrator_config.json > orchestrator_gcp.log 2>&1 &
echo \$! > simulation.pid &&
echo 'Simulation started with PID:' &&
cat simulation.pid
"
} else {
    # Use command line parameters
    Write-Host "`nStarting simulation: $description" -ForegroundColor Cyan
    Write-Host "Games: $games" -ForegroundColor Yellow
    Write-Host "Runs per game: $runsPerGame" -ForegroundColor Yellow
    Write-Host "Max iterations: $maxIterations" -ForegroundColor Yellow
    
    $simulationCommand = "source ~/.bashrc && source ~/venv/bin/activate && cd ~ && nohup python3 orchestrator.py --games `"$games`" --runs-per-game $runsPerGame --max-iterations $maxIterations --log-level INFO > orchestrator_gcp.log 2>&1 & echo \$! > simulation.pid && echo 'Simulation started with PID:' && cat simulation.pid"
}

# Execute simulation start
Write-Host "`nLaunching simulation on GCP VM..." -ForegroundColor Green
$result = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command=$simulationCommand

Write-Host $result -ForegroundColor White

if ($result -like "*Simulation started with PID:*") {
    Write-Host "`nSUCCESS: Simulation launched successfully!" -ForegroundColor Green
    
    # Extract PID
    $pid = ($result -split "Simulation started with PID:")[1].Trim()
    Write-Host "Process ID: $pid" -ForegroundColor Yellow
    
    Write-Host "`nSimulation Details:" -ForegroundColor Cyan
    Write-Host "Description: $description" -ForegroundColor White
    Write-Host "VM: nba-orchestrator (us-central1-a)" -ForegroundColor White
    Write-Host "Log file: orchestrator_gcp.log" -ForegroundColor White
    Write-Host "Process ID: $pid" -ForegroundColor White
    
    Write-Host "`nMonitoring Options:" -ForegroundColor Cyan
    Write-Host "1. Run: .\monitor.ps1 - Interactive monitoring dashboard" -ForegroundColor White
    Write-Host "2. View logs: gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --command='tail -f orchestrator_gcp.log'" -ForegroundColor White
    Write-Host "3. Check status: gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --command='ps aux | grep orchestrator'" -ForegroundColor White
    
    # Quick initial status check
    Write-Host "`nChecking initial status (waiting 10 seconds)..." -ForegroundColor Cyan
    Start-Sleep -Seconds 10
    
    $statusCheck = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="
if ps -p $pid > /dev/null 2>&1; then
    echo 'Process running SUCCESS'
    echo 'Log tail:'
    tail -5 orchestrator_gcp.log 2>/dev/null || echo 'Log not ready yet'
else
    echo 'Process not found ERROR - check for errors'
    tail -10 orchestrator_gcp.log 2>/dev/null || echo 'No log file'
fi
"
    
    Write-Host $statusCheck -ForegroundColor White
    
    Write-Host "`nSimulation is running! Use .\monitor.ps1 to track progress." -ForegroundColor Green
    
} else {
    Write-Host "`nERROR: Failed to start simulation" -ForegroundColor Red
    Write-Host "Check the error details above" -ForegroundColor Yellow
    exit 1
}

# Ask if user wants to start monitoring
$monitor = Read-Host "`nStart monitoring dashboard now? (y/n)"
if ($monitor.ToLower() -eq "y") {
    Write-Host "`nStarting monitoring dashboard..." -ForegroundColor Cyan
    & ".\monitor.ps1"
}
