#!/usr/bin/env pwsh
$ErrorActionPreference = 'Stop'

Write-Host "Packaging latest repo for GCP deploy..." -ForegroundColor Cyan

# Remove existing zip if present
if (Test-Path 'nba_predictions_gcp.zip') {
    Remove-Item 'nba_predictions_gcp.zip' -Force
}

# Collect files, excluding VCS, caches, data, dbs, logs, and error logs
$files = Get-ChildItem -Recurse -File |
    Where-Object {
        $_.FullName -notmatch '\\.git' -and
        $_.FullName -notmatch '\\__pycache__\\' -and
        $_.FullName -notmatch '\\data\\' -and
        $_.Extension -ne '.db' -and
        $_.Extension -ne '.log' -and
        $_.Name -notlike 'database_errors*' -and
        $_.Name -ne 'nba_predictions_gcp.zip'
    }

$paths = $files | Select-Object -ExpandProperty FullName
if (-not $paths -or $paths.Count -eq 0) {
    throw 'No files to package'
}

Compress-Archive -Path $paths -DestinationPath 'nba_predictions_gcp.zip' -Force

# Ensure requirements.txt included/updated
if (Test-Path 'requirements.txt') {
    Compress-Archive -Path 'requirements.txt' -Update -DestinationPath 'nba_predictions_gcp.zip'
}

$sizeBytes = (Get-Item 'nba_predictions_gcp.zip').Length
Write-Host ("Packaged zip size (bytes): {0}" -f $sizeBytes) -ForegroundColor Green
Write-Host "Done." -ForegroundColor Green


