#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Automated workflow to scrape NBA odds from oddschecker.com
    
.DESCRIPTION
    This script automates the process of:
    1. Finding the latest browser snapshot file
    2. Running the odds scraper
    3. Saving output to data/market/current_spreads.json
    
.PARAMETER SnapshotFile
    Path to a specific snapshot file. If not provided, uses the most recent snapshot.
    
.PARAMETER Date
    Filter odds for a specific date (YYYY-MM-DD). Default: today
    
.PARAMETER Output
    Output file path. Default: data/market/current_spreads.json
    
.EXAMPLE
    .\scrape_odds_automated.ps1
    
.EXAMPLE
    .\scrape_odds_automated.ps1 -Date 2025-11-02
    
.EXAMPLE
    .\scrape_odds_automated.ps1 -SnapshotFile "C:\path\to\snapshot.log"
#>

param(
    [Parameter(Mandatory=$false)]
    [string]$SnapshotFile,
    
    [Parameter(Mandatory=$false)]
    [string]$Date,
    
    [Parameter(Mandatory=$false)]
    [string]$Output = "data/market/current_spreads.json"
)

Write-Host ""
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 68) -ForegroundColor Cyan
Write-Host "NBA ODDS SCRAPER - Automated Workflow" -ForegroundColor Cyan
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 68) -ForegroundColor Cyan
Write-Host ""

# Find snapshot file if not provided
if (-not $SnapshotFile) {
    $snapshotDir = "$env:USERPROFILE\.cursor\browser-logs"
    
    if (-not (Test-Path $snapshotDir)) {
        Write-Host "ERROR: Browser snapshot directory not found: $snapshotDir" -ForegroundColor Red
        Write-Host ""
        Write-Host "You need to first navigate to the odds page and take a snapshot:" -ForegroundColor Yellow
        Write-Host "  1. Open Cursor chat" -ForegroundColor Yellow
        Write-Host "  2. Use browser navigation to go to: https://www.oddschecker.com/us/basketball/nba" -ForegroundColor Yellow
        Write-Host "  3. Take a snapshot of the page" -ForegroundColor Yellow
        Write-Host "  4. Then run this script again" -ForegroundColor Yellow
        Write-Host ""
        exit 1
    }
    
    # Find the most recent snapshot file
    $snapshotFiles = Get-ChildItem -Path $snapshotDir -Filter "snapshot-*.log" | Sort-Object LastWriteTime -Descending
    
    if ($snapshotFiles.Count -eq 0) {
        Write-Host "ERROR: No snapshot files found in $snapshotDir" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please take a browser snapshot first (see instructions above)." -ForegroundColor Yellow
        Write-Host ""
        exit 1
    }
    
    $SnapshotFile = $snapshotFiles[0].FullName
    Write-Host "Using most recent snapshot:" -ForegroundColor Green
    Write-Host "  $SnapshotFile" -ForegroundColor Gray
    Write-Host "  Created: $($snapshotFiles[0].LastWriteTime)" -ForegroundColor Gray
    Write-Host ""
}

# Build command
$cmd = "python scrape_odds.py --snapshot `"$SnapshotFile`" --output `"$Output`""

if ($Date) {
    $cmd += " --date $Date"
}

Write-Host "Running odds scraper..." -ForegroundColor Cyan
Write-Host ""

# Run scraper
Invoke-Expression $cmd

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=" -NoNewline -ForegroundColor Green
    Write-Host ("=" * 68) -ForegroundColor Green
    Write-Host "SUCCESS! Odds scraped and saved to:" -ForegroundColor Green
    Write-Host "  $Output" -ForegroundColor White
    Write-Host "=" -NoNewline -ForegroundColor Green
    Write-Host ("=" * 68) -ForegroundColor Green
    Write-Host ""
    
    # Show next steps
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Review the odds data (optional)" -ForegroundColor Gray
    Write-Host "  2. Run injury scraper: .\scrape_injuries.bat" -ForegroundColor Gray
    Write-Host "  3. Prepare games: python prepare_todays_games.py" -ForegroundColor Gray
    Write-Host "  4. Generate predictions (command will be shown after step 3)" -ForegroundColor Gray
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "ERROR: Odds scraping failed!" -ForegroundColor Red
    Write-Host ""
    exit 1
}

