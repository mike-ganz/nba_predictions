# NBA Predictions GCP Monitoring Script
# Project: utopian-outlook-470922-q2

Write-Host "NBA Predictions Monitoring Dashboard" -ForegroundColor Green

# Function to check VM status
function Get-VMStatus {
    Write-Host "`nVM Status:" -ForegroundColor Cyan
    $vmStatus = gcloud compute instances list --filter="name:nba-orchestrator" --format="table(name,status,machineType.scope(machineTypes),zone.scope(zones))" --project=utopian-outlook-470922-q2
    Write-Host $vmStatus -ForegroundColor White
}

# Function to check running processes
function Get-RunningProcesses {
    Write-Host "`nRunning Processes:" -ForegroundColor Cyan
    $processes = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="ps aux | grep -E '(orchestrator|python)' | grep -v grep" 2>$null
    if ($processes) {
        Write-Host $processes -ForegroundColor White
    } else {
        Write-Host "No orchestrator processes running" -ForegroundColor Yellow
    }
}

# Function to check log tail
function Get-LogTail {
    Write-Host "`nRecent Log Entries:" -ForegroundColor Cyan
    $logs = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="tail -20 orchestrator_gcp.log 2>/dev/null || echo 'Log file not found'"
    Write-Host $logs -ForegroundColor White
}

# Function to check database status
function Get-DatabaseStatus {
    Write-Host "`nDatabase Status:" -ForegroundColor Cyan
    $dbStatus = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="
    if [ -f enhanced_simulation_results.db ]; then
        echo 'Database file exists'
        python3 -c \"
import sqlite3
try:
    conn = sqlite3.connect('enhanced_simulation_results.db')
    cursor = conn.execute('SELECT COUNT(*) FROM simulation_runs')
    count = cursor.fetchone()[0]
    print(f'Total simulation runs: {count}')
    
    cursor = conn.execute('SELECT COUNT(*) FROM simulation_runs WHERE status = \"completed\"')
    completed = cursor.fetchone()[0]
    print(f'Completed runs: {completed}')
    
    cursor = conn.execute('SELECT COUNT(*) FROM simulation_runs WHERE created_at > datetime(\"now\", \"-1 hour\")')
    recent = cursor.fetchone()[0]
    print(f'Runs in last hour: {recent}')
    
    conn.close()
except Exception as e:
    print(f'Database error: {e}')
\"
    else
        echo 'Database file not found'
    fi
    "
    Write-Host $dbStatus -ForegroundColor White
}

# Function to check bucket status
function Get-BucketStatus {
    if (Test-Path "bucket_name.txt") {
        $bucketName = Get-Content "bucket_name.txt" -Raw
        $bucketName = $bucketName.Trim()
        Write-Host "`nCloud Storage Status:" -ForegroundColor Cyan
        Write-Host "Bucket: $bucketName" -ForegroundColor Yellow
        
        $bucketInfo = gsutil du -s gs://$bucketName 2>$null
        if ($bucketInfo) {
            Write-Host $bucketInfo -ForegroundColor White
        } else {
            Write-Host "Could not get bucket info" -ForegroundColor Yellow
        }
    } else {
        Write-Host "`nBucket info not available" -ForegroundColor Yellow
    }
}

# Main monitoring loop
do {
    Clear-Host
    Write-Host "NBA Predictions Monitoring Dashboard - $(Get-Date)" -ForegroundColor Green
    Write-Host "=" * 60 -ForegroundColor Green
    
    Get-VMStatus
    Get-RunningProcesses
    Get-LogTail
    Get-DatabaseStatus
    Get-BucketStatus
    
    Write-Host "`n🎮 Options:" -ForegroundColor Cyan
    Write-Host "R - Refresh dashboard" -ForegroundColor White
    Write-Host "L - View full logs" -ForegroundColor White
    Write-Host "S - SSH to VM" -ForegroundColor White
    Write-Host "D - Download database" -ForegroundColor White
    Write-Host "Q - Quit monitoring" -ForegroundColor White
    
    $choice = Read-Host "`nChoose option"
    
    switch ($choice.ToUpper()) {
        "L" {
            Write-Host "`nFull log viewer..." -ForegroundColor Cyan
            gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="less orchestrator_gcp.log"
        }
        "S" {
            Write-Host "`nOpening SSH connection..." -ForegroundColor Cyan
            gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch"
        }
        "D" {
            Write-Host "`nDownloading database..." -ForegroundColor Cyan
            $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
            gcloud compute scp nba-orchestrator:~/enhanced_simulation_results.db "enhanced_simulation_results_$timestamp.db" --zone=us-central1-a --project=utopian-outlook-470922-q2 --scp-flag="-batch"
            if ($LASTEXITCODE -eq 0) {
                Write-Host "SUCCESS: Database downloaded to enhanced_simulation_results_$timestamp.db" -ForegroundColor Green
            } else {
                Write-Host "ERROR: Download failed" -ForegroundColor Red
            }
            Read-Host "Press Enter to continue"
        }
        "Q" {
            Write-Host "`nExiting monitoring dashboard..." -ForegroundColor Green
            exit
        }
        default {
            # Refresh (default action)
            Start-Sleep -Seconds 1
        }
    }
} while ($true)
