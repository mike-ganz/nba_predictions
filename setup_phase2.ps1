# NBA Predictions GCP Setup - Phase 2: VM Setup
# Project: utopian-outlook-470922-q2
# Account: ganzy9@gmail.com

Write-Host "Starting NBA Predictions GCP Setup - Phase 2" -ForegroundColor Green

# Read bucket name from Phase 1
if (Test-Path "bucket_name.txt") {
    $BUCKET_NAME = Get-Content "bucket_name.txt" -Raw
    $BUCKET_NAME = $BUCKET_NAME.Trim()
    Write-Host "Using bucket: $BUCKET_NAME" -ForegroundColor Yellow
} else {
    Write-Host "ERROR: bucket_name.txt not found! Please run Phase 1 first." -ForegroundColor Red
    exit 1
}

# Step 2.1: Create Compute Instance
Write-Host "`nCreating VM instance..." -ForegroundColor Cyan
Write-Host "Specs: e2-standard-4, 50GB disk, us-central1-a" -ForegroundColor Yellow
Write-Host "This will take 2-3 minutes..." -ForegroundColor Yellow

gcloud compute instances create nba-orchestrator --zone=us-central1-a --machine-type=e2-standard-4 --boot-disk-size=50GB --boot-disk-type=pd-standard --service-account=nba-orchestrator@utopian-outlook-470922-q2.iam.gserviceaccount.com --scopes=https://www.googleapis.com/auth/cloud-platform --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud --project=utopian-outlook-470922-q2

if ($LASTEXITCODE -eq 0) {
    Write-Host "SUCCESS: VM created successfully!" -ForegroundColor Green
} else {
    Write-Host "ERROR: Failed to create VM" -ForegroundColor Red
    exit 1
}

# Step 2.2: Create Requirements File
Write-Host "`n📄 Creating requirements.txt..." -ForegroundColor Cyan

$requirements = @"
pandas>=1.5.0
numpy>=1.21.0
openai>=1.0.0
google-cloud-aiplatform>=1.34.0
python-dotenv>=0.19.0
ujson>=5.0.0
gcsfs>=2023.1.0
google-generativeai>=0.3.0
"@

$requirements | Out-File -FilePath "requirements.txt" -Encoding UTF8
Write-Host "SUCCESS: requirements.txt created" -ForegroundColor Green

# Step 2.3: Create Deployment Package  
Write-Host "`nCreating deployment package..." -ForegroundColor Cyan
Write-Host "Excluding: .git, __pycache__, .db files, data folder, logs" -ForegroundColor Yellow

# Get all files except excluded patterns
$excludePatterns = @("*.git*", "*__pycache__*", "*.db", "data", "*.log", "database_errors*")
$files = Get-ChildItem -Recurse | Where-Object { 
    $item = $_
    $shouldExclude = $false
    foreach ($pattern in $excludePatterns) {
        if ($item.FullName -like "*$pattern*") {
            $shouldExclude = $true
            break
        }
    }
    -not $shouldExclude -and -not $item.PSIsContainer
}

# Create zip file
Write-Host "Found $($files.Count) files to package..." -ForegroundColor Yellow
Compress-Archive -Path $files.FullName -DestinationPath "nba_predictions_gcp.zip" -Force

# Add requirements.txt specifically
Compress-Archive -Path "requirements.txt" -Update -DestinationPath "nba_predictions_gcp.zip"

if (Test-Path "nba_predictions_gcp.zip") {
    $zipSize = [math]::Round((Get-Item "nba_predictions_gcp.zip").Length / 1MB, 2)
    Write-Host "SUCCESS: Deployment package created: nba_predictions_gcp.zip ($zipSize MB)" -ForegroundColor Green
} else {
    Write-Host "ERROR: Failed to create deployment package" -ForegroundColor Red
    exit 1
}

# Step 2.4: Wait for VM to be ready
Write-Host "`nWaiting for VM to be ready..." -ForegroundColor Cyan
Start-Sleep -Seconds 30

# Check VM status
$vmStatus = gcloud compute instances list --filter="name:nba-orchestrator" --format="value(status)" 2>$null
if ($vmStatus -eq "RUNNING") {
    Write-Host "SUCCESS: VM is running and ready!" -ForegroundColor Green
} else {
    Write-Host "WARNING: VM status: $vmStatus - continuing anyway..." -ForegroundColor Yellow
}

Write-Host "`nPhase 2 Complete!" -ForegroundColor Green
Write-Host "Summary:" -ForegroundColor Yellow
Write-Host "  - VM: nba-orchestrator (e2-standard-4) in us-central1-a" -ForegroundColor White
Write-Host "  - Deployment package: nba_predictions_gcp.zip ($zipSize MB)" -ForegroundColor White
Write-Host "  - Requirements: requirements.txt created" -ForegroundColor White
Write-Host "`nReady for deployment!" -ForegroundColor Green

# Explicitly exit with success code
exit 0
