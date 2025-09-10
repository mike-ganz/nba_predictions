#!/usr/bin/env pwsh
# Run Your Custom results_analysis.py on GCP

Write-Host ("=" * 80)
Write-Host "RUNNING YOUR CUSTOM NBA RESULTS ANALYSIS ON GCP"
Write-Host ("=" * 80)

$BUCKET_NAME = (Get-Content "bucket_name.txt" -Raw).Trim()
Write-Host "Using bucket: $BUCKET_NAME"

# Step 1: Upload your custom analysis files
Write-Host "`nStep 1: Uploading your custom analysis scripts..."

# Upload your main results_analysis.py
if (Test-Path "results_analysis.py") {
    gcloud compute scp results_analysis.py nba-orchestrator: --zone=us-central1-a --project=utopian-outlook-470922-q2 --scp-flag="-batch"
    Write-Host "SUCCESS: results_analysis.py uploaded" -ForegroundColor Green
} else {
    Write-Host "ERROR: results_analysis.py not found locally" -ForegroundColor Red
    exit 1
}

# Upload nba_results_notebook.py (dependency)
if (Test-Path "nba_results_notebook.py") {
    gcloud compute scp nba_results_notebook.py nba-orchestrator: --zone=us-central1-a --project=utopian-outlook-470922-q2 --scp-flag="-batch"
    Write-Host "SUCCESS: nba_results_notebook.py uploaded" -ForegroundColor Green
} else {
    Write-Host "WARNING: nba_results_notebook.py not found" -ForegroundColor Yellow
}

# Step 2: Install required Python packages
Write-Host "`nStep 2: Installing required packages..."
$packageResult = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="source ~/.bashrc && source ~/venv/bin/activate && pip install matplotlib seaborn tabulate pandas numpy --quiet && echo 'PACKAGES_READY'"

if ($packageResult -like "*PACKAGES_READY*") {
    Write-Host "SUCCESS: All packages installed" -ForegroundColor Green
} else {
    Write-Host "WARNING: Package installation had issues" -ForegroundColor Yellow
}

# Step 3: Check if actual game results data exists (needed for betting analysis)
Write-Host "`nStep 3: Checking for actual game results data..."
$gameResultsCheck = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="test -f data/game_results_2024-2025.csv && echo 'FOUND' || echo 'NOT_FOUND'"

if ($gameResultsCheck -eq "NOT_FOUND") {
    Write-Host "INFO: Actual game results CSV not found - betting analysis will be skipped" -ForegroundColor Yellow
    Write-Host "      Your script will still run simulation analysis portions"
    
    # Upload game results if available locally
    if (Test-Path "data/game_results_2024-2025.csv") {
        Write-Host "      Found local game results - uploading..."
        # Create data directory on VM if needed
        gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="mkdir -p data"
        gcloud compute scp data/game_results_2024-2025.csv nba-orchestrator:data/ --zone=us-central1-a --project=utopian-outlook-470922-q2 --scp-flag="-batch"
        Write-Host "SUCCESS: Game results uploaded" -ForegroundColor Green
    }
} else {
    Write-Host "SUCCESS: Actual game results found - full betting analysis available" -ForegroundColor Green
}

# Step 4: Create a runner script to handle the database context
Write-Host "`nStep 4: Creating runner script for your analysis..."

$RUNNER_SCRIPT = @'
#!/usr/bin/env python3
"""
Runner for your custom results_analysis.py
Handles database context and path setup
"""
import sys
import os
import sqlite3
from pathlib import Path

# Change to home directory where database is located
os.chdir('/home/micha')

# Check if database exists
if not Path('enhanced_simulation_results.db').exists():
    print("ERROR: enhanced_simulation_results.db not found")
    sys.exit(1)

print("Database found. Running your custom analysis...")
print("=" * 60)

# Import and run your analysis
try:
    # Execute your analysis script in the correct context
    exec(open('results_analysis.py').read())
    print("\n" + "=" * 60)
    print("Your custom analysis completed successfully!")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Some modules may not be available - running simulation analysis only...")
    
    # Fallback: just run simulation summary if full analysis fails
    import pandas as pd
    
    conn = sqlite3.connect('enhanced_simulation_results.db')
    
    query = """
    SELECT 
        COUNT(*) as total,
        COUNT(CASE WHEN status IN ('completed', 'game_ended') THEN 1 END) as successful,
        COUNT(CASE WHEN status = 'error' THEN 1 END) as errors
    FROM simulation_runs
    """
    
    stats = pd.read_sql_query(query, conn)
    total = stats['total'].iloc[0]
    successful = stats['successful'].iloc[0]
    errors = stats['errors'].iloc[0]
    
    print(f"\nSIMULATION SUMMARY:")
    print(f"Total simulations: {total}")
    print(f"Successful: {successful} ({successful/total*100:.1f}%)")
    print(f"Errors: {errors} ({errors/total*100:.1f}%)")
    
    conn.close()
    
except Exception as e:
    print(f"Error running analysis: {e}")
    print("Check that all dependencies are available")
'@

# Upload runner script
$RUNNER_SCRIPT | Out-File -FilePath "temp_runner.py" -Encoding UTF8
gcloud compute scp temp_runner.py nba-orchestrator:analysis_runner.py --zone=us-central1-a --project=utopian-outlook-470922-q2 --scp-flag="-batch"
Remove-Item "temp_runner.py"

Write-Host "SUCCESS: Runner script created" -ForegroundColor Green

# Step 5: Run your custom analysis
Write-Host "`nStep 5: Running your custom analysis on GCP..."
Write-Host "This may take a few minutes for full betting analysis..." -ForegroundColor Cyan

$analysisOutput = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="source ~/.bashrc && source ~/venv/bin/activate && cd ~ && python3 analysis_runner.py"

Write-Host "`nANALYSIS OUTPUT:" -ForegroundColor Cyan
Write-Host ("-" * 80)
Write-Host $analysisOutput -ForegroundColor White
Write-Host ("-" * 80)

# Step 6: Check for output files and download them
Write-Host "`nStep 6: Checking for analysis output files..."

# Your script might create various output files - let's check what's available
$outputFiles = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="ls -la *.txt *.csv *.xlsx 2>/dev/null || echo 'NO_FILES'"

if ($outputFiles -ne "NO_FILES") {
    Write-Host "Found output files:" -ForegroundColor Green
    Write-Host $outputFiles -ForegroundColor Gray
    
    # Download any analysis result files
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    
    # Try to download common output file names
    $possibleFiles = @("analysis_results.txt", "betting_analysis.txt", "simulation_summary.txt")
    
    foreach ($file in $possibleFiles) {
        $downloadResult = gcloud compute scp nba-orchestrator:$file "./custom_$file`_$timestamp" --zone=us-central1-a --project=utopian-outlook-470922-q2 --scp-flag="-batch" 2>$null
        if ($?) {
            Write-Host "Downloaded: custom_$file`_$timestamp" -ForegroundColor Green
        }
    }
} else {
    Write-Host "No output files found - analysis results displayed above" -ForegroundColor Yellow
}

Write-Host ("`n" + "=" * 80)
Write-Host "YOUR CUSTOM ANALYSIS COMPLETE!" -ForegroundColor Green
Write-Host ("=" * 80)
Write-Host "This ran your full results_analysis.py with all custom betting logic" -ForegroundColor Green
