#!/usr/bin/env pwsh
# Working NBA Predictions GCP Monitoring Script

Write-Host "=" * 80 -ForegroundColor Green
Write-Host "NBA PREDICTIONS MONITORING DASHBOARD" -ForegroundColor Green  
Write-Host "=" * 80 -ForegroundColor Green

$BUCKET_NAME = (Get-Content "bucket_name.txt" -Raw).Trim()

# Check VM Status
Write-Host "`n🖥️  VM STATUS:" -ForegroundColor Cyan
$vmStatus = gcloud compute instances list --filter="name:nba-orchestrator" --format="value(status,machineType)" --project=utopian-outlook-470922-q2 2>$null
if ($vmStatus) {
    $status, $machineType = $vmStatus.Split("`t")
    Write-Host "   Status: $status" -ForegroundColor $(if($status -eq "RUNNING") {"Green"} else {"Red"})
    Write-Host "   Machine Type: $machineType" -ForegroundColor White
} else {
    Write-Host "   ❌ VM not found or not accessible" -ForegroundColor Red
}

# Check Running Processes
Write-Host "`n🔄 RUNNING PROCESSES:" -ForegroundColor Cyan
$processes = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="ps aux | grep orchestrator | grep -v grep | wc -l" 2>$null
if ($processes -and $processes -gt 0) {
    Write-Host "   ✅ $processes orchestrator process(es) running" -ForegroundColor Green
    
    # Get process details
    $processDetails = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="ps aux | grep orchestrator | grep -v grep | head -3" 2>$null
    if ($processDetails) {
        Write-Host "   Process details:" -ForegroundColor White
        $processDetails.Split("`n") | ForEach-Object { Write-Host "   $_" -ForegroundColor Gray }
    }
} else {
    Write-Host "   ⚠️  No orchestrator processes running" -ForegroundColor Yellow
}

# Check Database Status
Write-Host "`n🗄️  DATABASE STATUS:" -ForegroundColor Cyan
$dbCheck = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="ls -lh enhanced_simulation_results.db 2>/dev/null || echo 'NOT_FOUND'" 2>$null

if ($dbCheck -and $dbCheck -ne "NOT_FOUND") {
    Write-Host "   ✅ Database file exists" -ForegroundColor Green
    Write-Host "   File info: $dbCheck" -ForegroundColor Gray
    
    # Get simple count
    $simCount = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="sqlite3 enhanced_simulation_results.db 'SELECT COUNT(*) FROM simulation_runs;' 2>/dev/null || echo 'ERROR'" 2>$null
    
    if ($simCount -and $simCount -ne "ERROR") {
        Write-Host "   📊 Total simulation runs: $simCount" -ForegroundColor White
        
        # Get completed count
        $completedCount = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="sqlite3 enhanced_simulation_results.db 'SELECT COUNT(*) FROM simulation_runs WHERE status=\"completed\";' 2>/dev/null || echo 'ERROR'" 2>$null
        
        if ($completedCount -and $completedCount -ne "ERROR") {
            Write-Host "   ✅ Completed simulations: $completedCount" -ForegroundColor Green
        }
    }
} else {
    Write-Host "   ❌ Database file not found" -ForegroundColor Red
}

# Check Recent Logs
Write-Host "`n📋 RECENT LOG ENTRIES:" -ForegroundColor Cyan

# Try multiple possible log file names
$logFiles = @("orchestrator.log", "orchestrator_gcp.log", "simulation_*.log")
$foundLog = $false

foreach ($logFile in $logFiles) {
    $logCheck = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="ls $logFile 2>/dev/null | head -1" 2>$null
    
    if ($logCheck) {
        Write-Host "   📄 Log file: $logCheck" -ForegroundColor White
        $recentLogs = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="tail -5 $logCheck 2>/dev/null" 2>$null
        
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
            $foundLog = $true
            break
        }
    }
}

if (-not $foundLog) {
    Write-Host "   ⚠️  No log files found" -ForegroundColor Yellow
}

# System Resources
Write-Host "`n💾 SYSTEM RESOURCES:" -ForegroundColor Cyan
$diskUsage = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="df -h / | tail -1" 2>$null
if ($diskUsage) {
    Write-Host "   Disk usage: $diskUsage" -ForegroundColor White
}

$memUsage = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="free -h | head -2 | tail -1" 2>$null
if ($memUsage) {
    Write-Host "   Memory: $memUsage" -ForegroundColor White
}

Write-Host "`n" + "=" * 80 -ForegroundColor Green
Write-Host "DASHBOARD COMPLETE - Refresh with: ./monitor_working.ps1" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Green
