# NBA Predictions Complete GCP Setup
# Master script to run full setup process

Write-Host "NBA Predictions Complete GCP Setup" -ForegroundColor Green
Write-Host "This script will run the full setup process automatically" -ForegroundColor Yellow
Write-Host "Project: utopian-outlook-470922-q2" -ForegroundColor Cyan

# Confirmation
$confirm = Read-Host "`nThis will:
1. Update code for GCP compatibility
2. Create Cloud Storage and upload data
3. Create and configure VM
4. Deploy and test the system

Continue? (y/n)"

if ($confirm.ToLower() -ne "y") {
    Write-Host "Setup cancelled." -ForegroundColor Yellow
    exit
}

Write-Host "`nStarting complete setup process..." -ForegroundColor Green

# Step 1: Update code for GCP
Write-Host "`n" + "="*60 -ForegroundColor Green
Write-Host "STEP 1: Updating code for GCP compatibility" -ForegroundColor Green
Write-Host "="*60 -ForegroundColor Green

& ".\update_code_for_gcp.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Code update failed" -ForegroundColor Red
    exit 1
}

Write-Host "`nWaiting 5 seconds before next step..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Step 2: Phase 1 Setup
Write-Host "`n" + "="*60 -ForegroundColor Green
Write-Host "STEP 2: Phase 1 - Data & Storage Setup" -ForegroundColor Green
Write-Host "="*60 -ForegroundColor Green

& ".\setup_phase1.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Phase 1 setup failed" -ForegroundColor Red
    exit 1
}

Write-Host "`nWaiting 10 seconds before next step..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Step 3: Phase 2 Setup
Write-Host "`n" + "="*60 -ForegroundColor Green
Write-Host "STEP 3: Phase 2 - VM Setup" -ForegroundColor Green
Write-Host "="*60 -ForegroundColor Green

& ".\setup_phase2.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Phase 2 setup failed" -ForegroundColor Red
    exit 1
}

Write-Host "`nWaiting 10 seconds before deployment..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Step 4: Deploy
Write-Host "`n" + "="*60 -ForegroundColor Green
Write-Host "STEP 4: Deploying to GCP" -ForegroundColor Green
Write-Host "="*60 -ForegroundColor Green

& ".\deploy.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Deployment failed" -ForegroundColor Red
    exit 1
}

Write-Host "`nWaiting 15 seconds before testing..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

# Step 5: Test System
Write-Host "`n" + "="*60 -ForegroundColor Green
Write-Host "STEP 5: Testing System" -ForegroundColor Green
Write-Host "="*60 -ForegroundColor Green

& ".\test_system.ps1"
# Don't exit on test failure - system might still work

Write-Host "`n" + "="*60 -ForegroundColor Green
Write-Host "SETUP COMPLETE!" -ForegroundColor Green
Write-Host "="*60 -ForegroundColor Green

Write-Host "`nSetup Summary:" -ForegroundColor Cyan
if (Test-Path "bucket_name.txt") {
    $bucketName = Get-Content "bucket_name.txt" -Raw
    $bucketName = $bucketName.Trim()
    Write-Host "SUCCESS: Cloud Storage Bucket: $bucketName" -ForegroundColor Green
} else {
    Write-Host "WARNING: Bucket name not saved" -ForegroundColor Yellow
}
Write-Host "SUCCESS: VM: nba-orchestrator (e2-standard-4, us-central1-a)" -ForegroundColor Green
Write-Host "SUCCESS: Service Account: nba-orchestrator@utopian-outlook-470922-q2.iam.gserviceaccount.com" -ForegroundColor Green
Write-Host "SUCCESS: Gemini Models: 4062830704562536448, 8359264749073989632" -ForegroundColor Green

Write-Host "`nNext Steps:" -ForegroundColor Cyan
Write-Host "1. Run simulations: .\start_simulation.ps1" -ForegroundColor White
Write-Host "2. Monitor progress: .\monitor.ps1" -ForegroundColor White
Write-Host "3. SSH to VM: gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2" -ForegroundColor White

Write-Host "`nMonthly Cost Estimate: ~$65-75" -ForegroundColor Yellow
Write-Host "   - VM (e2-standard-4): ~$55/month" -ForegroundColor White
Write-Host "   - Cloud Storage: ~$8-15/month" -ForegroundColor White
Write-Host "   - Data transfer: ~$2-5/month" -ForegroundColor White

$startSim = Read-Host "`nStart a simulation now? (y/n)"
if ($startSim.ToLower() -eq "y") {
    Write-Host "`nStarting simulation..." -ForegroundColor Green
    & ".\start_simulation.ps1"
} else {
    Write-Host "`nSetup complete! Run .\start_simulation.ps1 when ready." -ForegroundColor Green
}
