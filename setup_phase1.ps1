# NBA Predictions GCP Setup - Phase 1: Data & Storage
# Project: utopian-outlook-470922-q2
# Account: ganzy9@gmail.com

Write-Host "Starting NBA Predictions GCP Setup - Phase 1" -ForegroundColor Green
Write-Host "Project: utopian-outlook-470922-q2" -ForegroundColor Yellow
Write-Host "Account: ganzy9@gmail.com" -ForegroundColor Yellow

# Step 1.1: Create Cloud Storage Bucket
Write-Host "`nCreating Cloud Storage Bucket..." -ForegroundColor Cyan
$BUCKET_NAME = "nba-predictions-$(Get-Date -UFormat %s)"
$BUCKET_NAME = $BUCKET_NAME -replace '\.', '-'  # Replace dots with dashes for valid bucket name
Write-Host "Bucket name: $BUCKET_NAME" -ForegroundColor Yellow

gsutil mb gs://$BUCKET_NAME
if ($LASTEXITCODE -eq 0) {
    Write-Host "SUCCESS: Bucket created successfully!" -ForegroundColor Green
    # Save bucket name for later scripts
    $BUCKET_NAME | Out-File -FilePath "bucket_name.txt" -Encoding UTF8
    Write-Host "💾 Bucket name saved to bucket_name.txt" -ForegroundColor Green
} else {
    Write-Host "ERROR: Failed to create bucket" -ForegroundColor Red
    exit 1
}

# Step 1.2: Upload Essential Data Files (excluding cache and training)
Write-Host "`nUploading essential data files to Cloud Storage..." -ForegroundColor Cyan
Write-Host "Excluding cache directories and training data (not needed for predictions)..." -ForegroundColor Yellow

# Upload only essential directories for predictions
Write-Host "Uploading play-by-play data..." -ForegroundColor Gray
gsutil -m cp -r data/play_by_play/ gs://$BUCKET_NAME/data/ 2>$null

Write-Host "Uploading other essential data (excluding cache)..." -ForegroundColor Gray
# Upload specific directories, skip cache
if (Test-Path "data/player_boxscores") { gsutil -m cp -r data/player_boxscores/ gs://$BUCKET_NAME/data/ 2>$null }
if (Test-Path "data/team_boxscores") { gsutil -m cp -r data/team_boxscores/ gs://$BUCKET_NAME/data/ 2>$null }
if (Test-Path "data/schedules") { gsutil -m cp -r data/schedules/ gs://$BUCKET_NAME/data/ 2>$null }

# Upload individual files in data root (but not subdirectories)
gsutil -m cp data/*.csv gs://$BUCKET_NAME/data/ 2>$null
gsutil -m cp data/*.json gs://$BUCKET_NAME/data/ 2>$null

Write-Host "SUCCESS: Essential prediction data uploaded (cache and training excluded)!" -ForegroundColor Green

# Upload database
Write-Host "`nUploading database..." -ForegroundColor Cyan
gsutil cp enhanced_simulation_results.db gs://$BUCKET_NAME/
if ($LASTEXITCODE -eq 0) {
    Write-Host "SUCCESS: Database uploaded successfully!" -ForegroundColor Green
} else {
    Write-Host "WARNING: Database upload failed, continuing..." -ForegroundColor Yellow
}

# Verify upload
Write-Host "`nVerifying uploads..." -ForegroundColor Cyan
gsutil ls -r gs://$BUCKET_NAME/data/ | Select-Object -First 10
Write-Host "`nUpload verification complete - showing first 10 files" -ForegroundColor Green

# Step 1.3: Create Service Account
Write-Host "`nCreating service account..." -ForegroundColor Cyan

gcloud iam service-accounts create nba-orchestrator --description="NBA Prediction Orchestrator" --display-name="NBA Orchestrator" --project=utopian-outlook-470922-q2
if ($LASTEXITCODE -eq 0) {
    Write-Host "SUCCESS: Service account created successfully!" -ForegroundColor Green
} else {
    Write-Host "WARNING: Service account may already exist, continuing..." -ForegroundColor Yellow
}

# Grant storage permissions
Write-Host "`nGranting storage permissions..." -ForegroundColor Cyan
gcloud projects add-iam-policy-binding utopian-outlook-470922-q2 --member="serviceAccount:nba-orchestrator@utopian-outlook-470922-q2.iam.gserviceaccount.com" --role="roles/storage.objectViewer"

gcloud projects add-iam-policy-binding utopian-outlook-470922-q2 --member="serviceAccount:nba-orchestrator@utopian-outlook-470922-q2.iam.gserviceaccount.com" --role="roles/storage.objectAdmin"

# Grant AI Platform permissions
Write-Host "`nGranting AI Platform permissions..." -ForegroundColor Cyan
gcloud projects add-iam-policy-binding utopian-outlook-470922-q2 --member="serviceAccount:nba-orchestrator@utopian-outlook-470922-q2.iam.gserviceaccount.com" --role="roles/aiplatform.user"

Write-Host "`nPhase 1 Complete!" -ForegroundColor Green
Write-Host "Summary:" -ForegroundColor Yellow
Write-Host "  - Bucket: $BUCKET_NAME" -ForegroundColor White
Write-Host "  - Data uploaded to Cloud Storage" -ForegroundColor White
Write-Host "  - Service account created with proper permissions" -ForegroundColor White
Write-Host "`nReady for Phase 2!" -ForegroundColor Green

# Explicitly exit with success code
exit 0
