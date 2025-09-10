#!/usr/bin/env pwsh
# Fixed NBA Results Analysis - No Unicode, Proper PowerShell Syntax

Write-Host ("=" * 80)
Write-Host "NBA RESULTS ANALYSIS (FIXED FOR POWERSHELL)"
Write-Host ("=" * 80)

$BUCKET_NAME = (Get-Content "bucket_name.txt" -Raw).Trim()
Write-Host "Using bucket: $BUCKET_NAME"

# Create fixed analysis script
Write-Host "`nCreating analysis script on VM..."

$ANALYSIS_SCRIPT = @'
#!/usr/bin/env python3
"""
Fixed NBA Results Analysis - Treats game_ended as successful completion
"""
import pandas as pd
import sqlite3
from pathlib import Path

def analyze_simulation_results():
    """Run analysis on enhanced_simulation_results.db"""
    
    db_path = "enhanced_simulation_results.db"
    
    if not Path(db_path).exists():
        print("ERROR: Database not found: enhanced_simulation_results.db")
        return
    
    print("Loading simulation results...")
    
    conn = sqlite3.connect(db_path)
    
    # Get basic statistics with corrected success definition
    query = """
    SELECT 
        COUNT(*) as total_simulations,
        COUNT(CASE WHEN status IN ('completed', 'game_ended') THEN 1 END) as successful,
        COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
        COUNT(CASE WHEN status = 'game_ended' THEN 1 END) as game_ended,
        COUNT(CASE WHEN status = 'error' THEN 1 END) as errors,
        COUNT(CASE WHEN status = 'timeout' THEN 1 END) as timeouts,
        AVG(duration_seconds) as avg_duration,
        COUNT(DISTINCT game_id) as unique_games,
        AVG(CASE WHEN status IN ('completed', 'game_ended') THEN scoring_rate END) as avg_scoring_rate,
        AVG(CASE WHEN status IN ('completed', 'game_ended') THEN successful_predictions END) as avg_predictions
    FROM simulation_runs
    """
    
    basic_stats = pd.read_sql_query(query, conn)
    
    total = basic_stats['total_simulations'].iloc[0]
    successful = basic_stats['successful'].iloc[0] 
    completed = basic_stats['completed'].iloc[0]
    game_ended = basic_stats['game_ended'].iloc[0]
    errors = basic_stats['errors'].iloc[0]
    
    print(f"\nSIMULATION SUMMARY")
    print("=" * 50)
    print(f"Total simulations: {total}")
    print(f"SUCCESS: {successful} ({(successful/total*100):.1f}%)")
    print(f"  - Game ended naturally: {game_ended}")
    print(f"  - Completed (max iterations): {completed}")
    print(f"ERRORS: {errors} ({(errors/total*100):.1f}%)")
    print(f"Unique games: {basic_stats['unique_games'].iloc[0]}")
    print(f"Average duration: {basic_stats['avg_duration'].iloc[0]:.1f} seconds")
    
    if basic_stats['avg_scoring_rate'].iloc[0]:
        print(f"Average scoring rate: {basic_stats['avg_scoring_rate'].iloc[0]:.1f}%")
    if basic_stats['avg_predictions'].iloc[0]:
        print(f"Average predictions per simulation: {basic_stats['avg_predictions'].iloc[0]:.1f}")
    
    # Get per-game breakdown with corrected success definition
    game_query = """
    SELECT 
        game_id,
        season_year,
        COUNT(*) as total_runs,
        COUNT(CASE WHEN status IN ('completed', 'game_ended') THEN 1 END) as successful_runs,
        COUNT(CASE WHEN status = 'error' THEN 1 END) as error_runs,
        AVG(CASE WHEN status IN ('completed', 'game_ended') THEN scoring_rate END) as avg_scoring_rate
    FROM simulation_runs 
    GROUP BY game_id, season_year
    ORDER BY successful_runs DESC, game_id
    """
    
    game_stats = pd.read_sql_query(game_query, conn)
    
    print(f"\nTOP 10 PERFORMING GAMES")
    print("=" * 50)
    for _, row in game_stats.head(10).iterrows():
        success_rate = (row['successful_runs'] / row['total_runs']) * 100
        print(f"Game {row['game_id']}: {row['successful_runs']}/{row['total_runs']} ({success_rate:.1f}% success)")
        if row['avg_scoring_rate']:
            print(f"  Average scoring: {row['avg_scoring_rate']:.1f}%")
        print()
    
    # Recent activity
    recent_query = """
    SELECT 
        run_id,
        game_id, 
        status,
        scoring_rate,
        start_time
    FROM simulation_runs 
    ORDER BY start_time DESC 
    LIMIT 10
    """
    
    recent_runs = pd.read_sql_query(recent_query, conn)
    
    print(f"RECENT SIMULATIONS")
    print("=" * 50)
    for _, row in recent_runs.iterrows():
        if row['status'] in ['completed', 'game_ended']:
            status_text = f"SUCCESS ({row['status']})"
            if row['scoring_rate']:
                status_text += f" - {row['scoring_rate']:.1f}% scoring"
        else:
            status_text = f"ERROR ({row['status']})"
            
        print(f"{row['run_id'][:30]}... | Game: {row['game_id']} | {status_text}")
    
    conn.close()
    
    # Save results
    with open('analysis_fixed_results.txt', 'w') as f:
        f.write("NBA SIMULATION ANALYSIS RESULTS\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total simulations: {total}\n")
        f.write(f"Successful simulations: {successful} ({(successful/total*100):.1f}%)\n")
        f.write(f"  - Game ended: {game_ended}\n")
        f.write(f"  - Completed: {completed}\n") 
        f.write(f"Error simulations: {errors} ({(errors/total*100):.1f}%)\n")
        f.write(f"Average duration: {basic_stats['avg_duration'].iloc[0]:.1f} seconds\n\n")
        f.write("TOP GAMES:\n")
        f.write(game_stats.head(20).to_string())
        f.write("\n\nRECENT RUNS:\n")
        f.write(recent_runs.to_string())
    
    print(f"\nResults saved to: analysis_fixed_results.txt")

if __name__ == "__main__":
    analyze_simulation_results()
'@

# Create the analysis script on VM using a different approach to avoid syntax issues
Write-Host "Creating analysis script file..."
$ANALYSIS_SCRIPT | Out-File -FilePath "temp_analysis.py" -Encoding UTF8

# Upload the file instead of trying to create it via SSH
Write-Host "Uploading analysis script to VM..."
gcloud compute scp temp_analysis.py nba-orchestrator:analysis_fixed.py --zone=us-central1-a --project=utopian-outlook-470922-q2 --scp-flag="-batch"

# Clean up local temp file
Remove-Item "temp_analysis.py"

Write-Host "SUCCESS: Analysis script uploaded to VM" -ForegroundColor Green

# Run the analysis
Write-Host "`nRunning analysis on GCP..."
$analysisResult = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="source ~/.bashrc && source ~/venv/bin/activate && cd ~ && python3 analysis_fixed.py"

Write-Host $analysisResult -ForegroundColor White

# Download results
Write-Host "`nDownloading analysis results..."
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
gcloud compute scp nba-orchestrator:analysis_fixed_results.txt "./analysis_fixed_$timestamp.txt" --zone=us-central1-a --project=utopian-outlook-470922-q2 --scp-flag="-batch"

if (Test-Path "./analysis_fixed_$timestamp.txt") {
    Write-Host "SUCCESS: Analysis results downloaded: analysis_fixed_$timestamp.txt" -ForegroundColor Green
    
    Write-Host "`nPreview of results:" -ForegroundColor Cyan
    Write-Host ("-" * 40)
    Get-Content "./analysis_fixed_$timestamp.txt" -Head 15
    Write-Host "`n... (full results in analysis_fixed_$timestamp.txt)"
    
} else {
    Write-Host "WARNING: Could not download analysis results" -ForegroundColor Yellow
}

Write-Host ("`n" + "=" * 80)
Write-Host "ANALYSIS COMPLETE!" -ForegroundColor Green
Write-Host ("=" * 80)
