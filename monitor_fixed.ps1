#!/usr/bin/env pwsh
# Fixed NBA Predictions GCP Monitoring Script - No Unicode, Proper PowerShell Syntax

Write-Host ("=" * 80) -ForegroundColor Green
Write-Host "NBA PREDICTIONS MONITORING DASHBOARD" -ForegroundColor Green  
Write-Host ("=" * 80) -ForegroundColor Green

$BUCKET_NAME = (Get-Content "bucket_name.txt" -Raw).Trim()

# Check VM Status
Write-Host "`n[VM] STATUS:" -ForegroundColor Cyan
$vmStatus = gcloud compute instances list --filter="name:nba-orchestrator" --format="value(status,machineType)" --project=utopian-outlook-470922-q2 2>$null
if ($vmStatus) {
    $status, $machineType = $vmStatus.Split("`t")
    Write-Host "   Status: $status" -ForegroundColor $(if($status -eq "RUNNING") {"Green"} else {"Red"})
    Write-Host "   Machine Type: $machineType" -ForegroundColor White
} else {
    Write-Host "   ERROR: VM not found or not accessible" -ForegroundColor Red
}

# Check Running Processes
Write-Host "`n[PROCESS] RUNNING SIMULATIONS:" -ForegroundColor Cyan
$processCount = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="ps aux | grep orchestrator | grep -v grep | wc -l" 2>$null
if ($processCount -and [int]$processCount -gt 0) {
    Write-Host "   SUCCESS: $processCount orchestrator process(es) running" -ForegroundColor Green
    
    # Get process details
    $processDetails = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="ps aux | grep orchestrator | grep -v grep | head -3" 2>$null
    if ($processDetails) {
        Write-Host "   Process details:" -ForegroundColor White
        $processDetails.Split("`n") | ForEach-Object { Write-Host "   $_" -ForegroundColor Gray }
    }
} else {
    Write-Host "   WARNING: No orchestrator processes running" -ForegroundColor Yellow
}

# Check Database Status
Write-Host "`n[DATABASE] STATUS:" -ForegroundColor Cyan
$dbExists = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="test -f enhanced_simulation_results.db && echo 'EXISTS' || echo 'NOT_FOUND'" 2>$null

if ($dbExists -eq "EXISTS") {
    Write-Host "   SUCCESS: Database file exists" -ForegroundColor Green
    
    # Get file size
    $dbSize = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="ls -lh enhanced_simulation_results.db | awk '{print `$5}'" 2>$null
    Write-Host "   File size: $dbSize" -ForegroundColor Gray
    
    # Get simple count
    $simCount = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="sqlite3 enhanced_simulation_results.db 'SELECT COUNT(*) FROM simulation_runs;'" 2>$null
    
    if ($simCount -and $simCount -match '^\d+$') {
        Write-Host "   TOTAL SIMULATIONS: $simCount" -ForegroundColor White
        
        # Get successful count (game_ended + completed)
        $successfulCount = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="sqlite3 enhanced_simulation_results.db 'SELECT COUNT(*) FROM simulation_runs WHERE status IN (''completed'', ''game_ended'');'" 2>$null
        
        if ($successfulCount -and $successfulCount -match '^\d+$') {
            $successRate = [math]::Round(([int]$successfulCount / [int]$simCount) * 100, 1)
            Write-Host "   SUCCESS: $successfulCount successful ($successRate%)" -ForegroundColor Green
        }
    }
} else {
    Write-Host "   ERROR: Database file not found" -ForegroundColor Red
}

# Check Recent Logs
Write-Host "`n[LOGS] RECENT ENTRIES:" -ForegroundColor Cyan

# Try to find log files
$logFile = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="find . -name '*.log' -type f | head -1" 2>$null

if ($logFile) {
    Write-Host "   Log file: $logFile" -ForegroundColor White
    $recentLogs = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="tail -5 '$logFile'" 2>$null
    
    if ($recentLogs) {
        $recentLogs.Split("`n") | ForEach-Object { 
            if ($_ -match "ERROR|Failed") {
                Write-Host "   $_" -ForegroundColor Red
            } elseif ($_ -match "SUCCESS|Complete") {
                Write-Host "   $_" -ForegroundColor Green
            } else {
                Write-Host "   $_" -ForegroundColor Gray
            }
        }
    }
} else {
    Write-Host "   WARNING: No log files found" -ForegroundColor Yellow
}

# System Resources
Write-Host "`n[SYSTEM] RESOURCES:" -ForegroundColor Cyan
$diskUsage = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="df -h / | tail -1" 2>$null
if ($diskUsage) {
    Write-Host "   Disk usage: $diskUsage" -ForegroundColor White
}

$memUsage = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="free -h | head -2 | tail -1" 2>$null
if ($memUsage) {
    Write-Host "   Memory: $memUsage" -ForegroundColor White
}

Write-Host ("`n" + "=" * 80) -ForegroundColor Green
Write-Host "DASHBOARD COMPLETE - Refresh with: ./monitor_fixed.ps1" -ForegroundColor Green
Write-Host ("=" * 80) -ForegroundColor Green
