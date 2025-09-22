# NBA Predictions GCP Monitoring Script - Simple PowerShell

param(
    [string]$LogFile = "",
    [string]$DatabaseFile = "enhanced_simulation_results_multithreaded.db",
    [switch]$Help
)

# Show help if requested
if ($Help) {
    Write-Host "NBA Predictions GCP Monitoring Script" -ForegroundColor Green
    Write-Host "Usage: .\monitor.ps1 [-LogFile <filename>] [-DatabaseFile <filename>] [-Help]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Parameters:" -ForegroundColor Cyan
    Write-Host "  -LogFile <filename>     : Specific log file to monitor (optional)" -ForegroundColor White
    Write-Host "  -DatabaseFile <filename>: Database file to download (default: enhanced_simulation_results_multithreaded.db)" -ForegroundColor White
    Write-Host "  -Help                   : Show this help message" -ForegroundColor White
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Cyan
    Write-Host "  .\monitor.ps1" -ForegroundColor White
    Write-Host "  .\monitor.ps1 -LogFile test_2threads_orchestrator.log" -ForegroundColor White
    Write-Host "  .\monitor.ps1 -LogFile my_test.log -DatabaseFile test_results.db" -ForegroundColor White
    exit 0
}

# Global variables for script parameters
$global:TargetLogFile = $LogFile
$global:TargetDatabaseFile = $DatabaseFile

function Get-VMStatus {
    Write-Host "`nVM Status:" -ForegroundColor Cyan
    try {
        $vmStatus = gcloud compute instances list --filter="name:nba-orchestrator" --format="value(status)" --project=utopian-outlook-470922-q2 2>$null
        if ($vmStatus -eq "RUNNING") {
            Write-Host "   Status: VM is running" -ForegroundColor Green
        } else {
            Write-Host "   Status: $vmStatus" -ForegroundColor Red
        }
    } catch {
        Write-Host "   Error: Cannot check VM status" -ForegroundColor Red
    }
}

function Get-ProcessStatus {
    Write-Host "`nProcess Status:" -ForegroundColor Cyan
    try {
        $cmd = "ps aux | grep enhanced_orchestrator | grep -v grep"
        $processes = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command=$cmd 2>$null
        
        if ($processes -and $processes.Trim()) {
            Write-Host "   Enhanced orchestrator is running:" -ForegroundColor Green
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
            Write-Host "   No enhanced orchestrator processes running" -ForegroundColor Red
        }
    } catch {
        Write-Host "   Error: Cannot check process status" -ForegroundColor Red
    }
}

function Get-RecentLogs {
    Write-Host "`nRecent Log Entries:" -ForegroundColor Cyan
    try {
        # Build log files array with priority to specified log file
        $logFiles = @()
        if ($global:TargetLogFile -ne "") {
            $logFiles += $global:TargetLogFile
        }
        $logFiles += @("orchestrator.log", "enhanced_orchestrator.log", "big_run_multithreaded.log")
        
        # Remove duplicates while preserving order
        $logFiles = $logFiles | Select-Object -Unique
        $foundLog = $false
        
        foreach ($logFile in $logFiles) {
            $testCmd = "test -f $logFile"
            gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command=$testCmd 2>$null
            
            if ($LASTEXITCODE -eq 0) {
                $logCmd = "tail -10 $logFile"
                $logs = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command=$logCmd 2>$null
                
                if ($logs -and $logs.Trim()) {
                    Write-Host "   From ${logFile}:" -ForegroundColor Yellow
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
            Write-Host "   No recent log entries found" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "   Error retrieving logs: $_" -ForegroundColor Red
    }
}

function Get-DatabaseStats {
    Write-Host "`nDatabase Status:" -ForegroundColor Cyan
    try {
        $dbCmd = "ls -lh *.db 2>/dev/null"
        $dbFiles = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command=$dbCmd 2>$null
        
        if ($dbFiles -and $dbFiles.Trim()) {
            Write-Host "   Database files:" -ForegroundColor Green
            $dbFiles -split "`n" | Where-Object { $_ -match "\.db" } | ForEach-Object {
                $parts = $_ -split '\s+', 9
                if ($parts.Length -ge 5) {
                    $size = $parts[4]
                    $name = $parts[8]
                    Write-Host "     $name ($size)" -ForegroundColor White
                }
            }
        } else {
            Write-Host "   No database files found" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "   Error checking database files: $_" -ForegroundColor Red
    }
}

function Show-Menu {
    Write-Host "`nOptions:" -ForegroundColor Cyan
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
    
    # Build log files array with priority to specified log file
    $logFiles = @()
    if ($global:TargetLogFile -ne "") {
        $logFiles += $global:TargetLogFile
    }
    $logFiles += @("orchestrator.log", "enhanced_orchestrator.log", "big_run_multithreaded.log")
    
    # Remove duplicates while preserving order
    $logFiles = $logFiles | Select-Object -Unique
    
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
    
    # Show current configuration
    if ($global:TargetLogFile -ne "") {
        Write-Host "Target Log: $($global:TargetLogFile)" -ForegroundColor Cyan
    } else {
        Write-Host "Target Log: Auto-detect" -ForegroundColor Cyan
    }
    Write-Host "Target DB: $($global:TargetDatabaseFile)" -ForegroundColor Cyan
    
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
            
            # Flush WAL -> main DB via Python (sqlite3 CLI might not be installed on VM)
            $checkpointCmd = @"
source ~/.bashrc && source ~/venv/bin/activate && python3 -c "import sqlite3; p='/home/micha/$($global:TargetDatabaseFile)'; con=sqlite3.connect(p, check_same_thread=False); con.execute('PRAGMA wal_checkpoint(FULL);'); con.commit(); con.close(); print('CHECKPOINT_DONE')"
"@
            try {
                gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command=$checkpointCmd 2>$null | Out-Null
            } catch {
                Write-Host "   Warning: Could not run WAL checkpoint (will download anyway)" -ForegroundColor Yellow
            }

            # Download the main DB after checkpoint
            $localDbName = $global:TargetDatabaseFile -replace '\.db$', '_current.db'
            gcloud compute scp "nba-orchestrator:$($global:TargetDatabaseFile)" "./$localDbName" --zone=us-central1-a --project=utopian-outlook-470922-q2 --scp-flag="-batch"
            Write-Host "Download completed as $localDbName!" -ForegroundColor Green
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
