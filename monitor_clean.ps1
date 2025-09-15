#!/usr/bin/env pwsh
# NBA Predictions GCP Monitoring Script - PURE PowerShell (no bash syntax)

function Get-VMStatus {
    Write-Host "`n🖥️  VM Status:" -ForegroundColor Cyan
    try {
        $vmStatus = gcloud compute instances list --filter="name:nba-orchestrator" --format="value(status)" --project=utopian-outlook-470922-q2 2>$null
        if ($vmStatus -eq "RUNNING") {
            Write-Host "   ✅ VM is running" -ForegroundColor Green
        } else {
            Write-Host "   ❌ VM status: $vmStatus" -ForegroundColor Red
        }
    } catch {
        Write-Host "   ❌ Cannot check VM status" -ForegroundColor Red
    }
}

function Get-ProcessStatus {
    Write-Host "`n🔄 Process Status:" -ForegroundColor Cyan
    try {
        $cmd = "ps aux | grep enhanced_orchestrator | grep -v grep"
        $processes = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command=$cmd 2>$null
        
        if ($processes -and $processes.Trim()) {
            Write-Host "   ✅ Enhanced orchestrator running:" -ForegroundColor Green
            $processes -split "`n" | ForEach-Object {
                if ($_ -match "enhanced_orchestrator" -and $_ -notmatch "grep") {
                    $parts = $_ -split '\s+', 11
                    if ($parts.Length -ge 3) {
                        $cpu = $parts[2]
                        $mem = $parts[3]
                        Write-Host "     CPU: $cpu% Memory: $mem%" -ForegroundColor White
                    }
                }
            }
        } else {
            Write-Host "   ❌ No enhanced orchestrator processes running" -ForegroundColor Red
        }
    } catch {
        Write-Host "   ❌ Cannot check process status" -ForegroundColor Red
    }
}

function Get-RecentLogs {
    Write-Host "`n📋 Recent Log Entries:" -ForegroundColor Cyan
    try {
        # Try each log file individually using separate commands
        $logFiles = @("big_run_fixed.log", "big_run_multithreaded.log", "big_run_simulation.log")
        $foundLog = $false
        
        foreach ($logFile in $logFiles) {
            # Check if file exists (separate command)
            $testCmd = "test -f $logFile"
            $fileExists = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command=$testCmd 2>$null
            $exitCode = $LASTEXITCODE
            
            if ($exitCode -eq 0) {
                # File exists, get the logs (separate command)
                $logCmd = "tail -10 $logFile | strings | grep -E 'ITERATION|NEW PLAY|Game State|ERROR|threads' | tail -5"
                $logs = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command=$logCmd 2>$null
                
                if ($logs -and $logs.Trim()) {
                    Write-Host "   📄 From $logFile:" -ForegroundColor Yellow
                    $logs -split "`n" | ForEach-Object {
                        if ($_ -and $_.Trim()) {
                            Write-Host "     $_" -ForegroundColor White
                        }
                    }
                    $foundLog = $true
                    break
                }
            }
        }
        
        if (-not $foundLog) {
            Write-Host "   ⚠️  No recent log entries found" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "   ❌ Cannot retrieve logs: $_" -ForegroundColor Red
    }
}

function Get-DatabaseStats {
    Write-Host "`n🗄️  Database Status:" -ForegroundColor Cyan
    try {
        $dbCmd = "ls -lh *.db 2>/dev/null"
        $dbFiles = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command=$dbCmd 2>$null
        
        if ($dbFiles -and $dbFiles.Trim()) {
            Write-Host "   📊 Database files:" -ForegroundColor Green
            $dbFiles -split "`n" | Where-Object { $_ -match "\.db" } | ForEach-Object {
                $parts = $_ -split '\s+', 9
                if ($parts.Length -ge 5) {
                    $size = $parts[4]
                    $name = $parts[8]
                    Write-Host "     $name ($size)" -ForegroundColor White
                }
            }
        } else {
            Write-Host "   ⚠️  No database files found" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "   ❌ Cannot check database files: $_" -ForegroundColor Red
    }
}

function Show-Menu {
    Write-Host "`n🎮 Options:" -ForegroundColor Cyan
    Write-Host "   r - Refresh dashboard"
    Write-Host "   l - View live logs"
    Write-Host "   s - SSH to VM"
    Write-Host "   d - Download latest database"
    Write-Host "   q - Quit"
    
    $choice = Read-Host "`nEnter choice (r/l/s/d/q)"
    return $choice
}

function Start-LiveLogs {
    Write-Host "`nStarting live log view (Ctrl+C to return)..." -ForegroundColor Yellow
    Start-Sleep -Seconds 1
    
    # Try each log file until we find one
    $logFiles = @("big_run_fixed.log", "big_run_multithreaded.log", "big_run_simulation.log")
    
    foreach ($logFile in $logFiles) {
        $testCmd = "test -f $logFile"
        gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command=$testCmd 2>$null
        
        if ($LASTEXITCODE -eq 0) {
            $tailCmd = "tail -f $logFile"
            gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command=$tailCmd
            return
        }
    }
    
    Write-Host "No log files found to monitor" -ForegroundColor Red
}

# Main monitoring loop
do {
    Clear-Host
    Write-Host "NBA Predictions GCP Monitoring Dashboard" -ForegroundColor Green
    Write-Host "=" * 60
    Write-Host "Last updated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    
    Get-VMStatus
    Get-ProcessStatus  
    Get-RecentLogs
    Get-DatabaseStats
    
    $choice = Show-Menu
    
    switch ($choice) {
        "l" {
            Start-LiveLogs
        }
        "s" {
            Write-Host "`nConnecting to VM..." -ForegroundColor Yellow
            gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2
        }
        "d" {
            Write-Host "`nDownloading latest database..." -ForegroundColor Yellow
            $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
            gcloud compute scp "nba-orchestrator:enhanced_simulation_results_multithreaded.db" "./enhanced_simulation_results_$timestamp.db" --zone=us-central1-a --project=utopian-outlook-470922-q2 --scp-flag="-batch"
            Write-Host "Download completed as enhanced_simulation_results_$timestamp.db!" -ForegroundColor Green
            Read-Host "Press Enter to continue"
        }
        "q" {
            Write-Host "`nExiting monitor..." -ForegroundColor Yellow
            break
        }
        default {
            Start-Sleep -Seconds 1
        }
    }
} while ($choice -ne "q")

Write-Host "`nMonitoring session ended." -ForegroundColor Gray
