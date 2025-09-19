#!/usr/bin/env pwsh
$ErrorActionPreference = 'Stop'

$zone = 'us-central1-a'
$project = 'utopian-outlook-470922-q2'
$instance = 'nba-orchestrator'
$remoteBase = '/home/micha/data'

Write-Host "Ensuring remote data directory exists..." -ForegroundColor Cyan
gcloud compute ssh $instance --zone $zone --project $project --strict-host-key-checking=no --command "mkdir -p $remoteBase"

if (!(Test-Path -LiteralPath './data')) {
    throw "Local './data' directory not found"
}

$excludeNames = @('training','__pycache__','cache')

Write-Host "Uploading data subdirectories (excluding: $($excludeNames -join ', '))..." -ForegroundColor Cyan
$dirs = Get-ChildItem -LiteralPath './data' -Directory -ErrorAction Stop
foreach ($d in $dirs) {
    if ($excludeNames -contains $d.Name) { continue }
    $src = $d.FullName
    $dest = "${instance}:${remoteBase}/"
    Write-Host (" - " + $d.Name) -ForegroundColor Yellow
    & gcloud compute scp --recurse --strict-host-key-checking=no "$src" "$dest" --zone $zone --project $project
}

Write-Host "Uploading files at data/ root..." -ForegroundColor Cyan
$files = Get-ChildItem -LiteralPath './data' -File -ErrorAction Stop
foreach ($f in $files) {
    $src = $f.FullName
    $dest = "${instance}:${remoteBase}/"
    Write-Host (" - " + $f.Name) -ForegroundColor Yellow
    & gcloud compute scp --strict-host-key-checking=no "$src" "$dest" --zone $zone --project $project
}

Write-Host "Data upload complete." -ForegroundColor Green


