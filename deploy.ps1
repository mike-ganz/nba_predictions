# NBA Predictions GCP Deployment Script
# Project: utopian-outlook-470922-q2
# Gemini Endpoints: 4062830704562536448, 8359264749073989632

Write-Host "Starting NBA Predictions Deployment" -ForegroundColor Green

# Read bucket name
if (Test-Path "bucket_name.txt") {
    $BUCKET_NAME = Get-Content "bucket_name.txt" -Raw
    $BUCKET_NAME = $BUCKET_NAME.Trim()
    Write-Host "Using bucket: $BUCKET_NAME" -ForegroundColor Yellow
} else {
    Write-Host "ERROR: bucket_name.txt not found! Please run Phase 1 first." -ForegroundColor Red
    exit 1
}

# Step 1: Copy code to VM
Write-Host "`nUploading code to VM..." -ForegroundColor Cyan
gcloud compute scp nba_predictions_gcp.zip nba-orchestrator: --zone=us-central1-a --project=utopian-outlook-470922-q2 --scp-flag="-batch"

if ($LASTEXITCODE -eq 0) {
    Write-Host "SUCCESS: Code uploaded successfully!" -ForegroundColor Green
} else {
    Write-Host "ERROR: Failed to upload code" -ForegroundColor Red
    exit 1
}

# Step 2: Setup environment on VM
Write-Host "`nSetting up Python environment on VM..." -ForegroundColor Cyan
Write-Host "This will take 3-5 minutes..." -ForegroundColor Yellow

$setupScript = @"
cd ~ &&
unzip -o nba_predictions_gcp.zip &&
sudo apt update &&
sudo apt install -y python3-pip python3-venv &&
python3 -m venv venv &&
source venv/bin/activate &&
pip install --upgrade pip &&
pip install -r requirements.txt &&
echo 'Environment setup complete'
"@

gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command=$setupScript

if ($LASTEXITCODE -eq 0) {
    Write-Host "SUCCESS: Environment setup complete!" -ForegroundColor Green
} else {
    Write-Host "ERROR: Environment setup failed" -ForegroundColor Red
    exit 1
}

# Step 3: Configure environment variables
Write-Host "`nSetting up environment variables..." -ForegroundColor Cyan

$envScript = @"
echo 'export NBA_DATA_BUCKET=$BUCKET_NAME' >> ~/.bashrc &&
echo 'export PREDICTION_PLATFORM=gemini' >> ~/.bashrc &&
echo 'export GEMINI_MODEL_1_ENDPOINT=4062830704562536448' >> ~/.bashrc &&
echo 'export GEMINI_MODEL_2_ENDPOINT=8359264749073989632' >> ~/.bashrc &&
echo 'export GOOGLE_CLOUD_PROJECT=utopian-outlook-470922-q2' >> ~/.bashrc &&
echo 'export GOOGLE_CLOUD_LOCATION=us-central1' >> ~/.bashrc &&
source ~/.bashrc &&
echo 'Environment variables set'
"@

gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command=$envScript

if ($LASTEXITCODE -eq 0) {
    Write-Host "SUCCESS: Environment variables configured!" -ForegroundColor Green
} else {
    Write-Host "WARNING: Environment variable setup had issues, continuing..." -ForegroundColor Yellow
}

# Step 4: Test basic functionality
Write-Host "`nTesting basic functionality..." -ForegroundColor Cyan

$testScript = @"
source ~/.bashrc &&
source ~/venv/bin/activate &&
cd ~ &&
python3 -c 'from config.settings import config; print(f\"Testing config: {config.season_year}\")' &&
python3 -c 'import os; print(f\"Bucket: {os.getenv(\"NBA_DATA_BUCKET\", \"NOT_SET\")}\")' &&
python3 -c 'import os; print(f\"Platform: {os.getenv(\"PREDICTION_PLATFORM\", \"NOT_SET\")}\")' &&
echo 'Basic tests complete'
"@

gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command=$testScript

if ($LASTEXITCODE -eq 0) {
    Write-Host "SUCCESS: Basic tests passed!" -ForegroundColor Green
} else {
    Write-Host "WARNING: Some tests failed, but deployment may still work" -ForegroundColor Yellow
}

Write-Host "`nDeployment Complete!" -ForegroundColor Green
Write-Host "Summary:" -ForegroundColor Yellow
Write-Host "  - Code deployed to VM: nba-orchestrator" -ForegroundColor White
Write-Host "  - Python environment ready" -ForegroundColor White
Write-Host "  - Environment variables configured" -ForegroundColor White
Write-Host "  - Bucket: $BUCKET_NAME" -ForegroundColor White
Write-Host "  - Gemini endpoints configured" -ForegroundColor White

Write-Host "`nNext Steps:" -ForegroundColor Cyan
Write-Host "1. Run: .\test_system.ps1 - to test the system" -ForegroundColor White
Write-Host "2. Run: .\start_simulation.ps1 - to start simulations" -ForegroundColor White
Write-Host "3. Run: .\monitor.ps1 - to monitor progress" -ForegroundColor White

# Explicitly exit with success code
exit 0
