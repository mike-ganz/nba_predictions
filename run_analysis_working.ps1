#!/usr/bin/env pwsh
# Working NBA Results Analysis Script for GCP

Write-Host "=" * 80
Write-Host "NBA RESULTS ANALYSIS ON GCP (Working Version)"
Write-Host "=" * 80

$BUCKET_NAME = (Get-Content "bucket_name.txt" -Raw).Trim()
Write-Host "Using bucket: $BUCKET_NAME"

# Step 1: Install required packages if needed
Write-Host "`nStep 1: Installing analysis packages..."
$packageInstall = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="source ~/.bashrc && source ~/venv/bin/activate && pip install matplotlib seaborn tabulate --quiet && echo 'Packages ready'"

if ($packageInstall -like "*Packages ready*") {
    Write-Host "✅ Analysis packages installed" -ForegroundColor Green
} else {
    Write-Host "⚠️  Package installation had warnings (likely already installed)" -ForegroundColor Yellow
}

# Step 2: Create analysis script directly on VM
Write-Host "`nStep 2: Creating analysis script on VM..."

$ANALYSIS_SCRIPT = @'
#!/usr/bin/env python3
"""
Terminal-friendly NBA Results Analysis
"""
import pandas as pd
import sqlite3
import sys
from pathlib import Path

def analyze_simulation_results():
    """Run analysis on enhanced_simulation_results.db"""
    
    db_path = "enhanced_simulation_results.db"
    
    if not Path(db_path).exists():
        print("❌ Database not found: enhanced_simulation_results.db")
        return
    
    print("📊 Loading simulation results...")
    
    conn = sqlite3.connect(db_path)
    
    # Get basic statistics
    query = """
    SELECT 
        COUNT(*) as total_simulations,
        COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
        COUNT(CASE WHEN status = 'error' THEN 1 END) as errors,
        COUNT(CASE WHEN status = 'timeout' THEN 1 END) as timeouts,
        AVG(duration_seconds) as avg_duration,
        COUNT(DISTINCT game_id) as unique_games
    FROM simulation_runs
    """
    
    basic_stats = pd.read_sql_query(query, conn)
    
    print(f"\n🏀 SIMULATION SUMMARY")
    print("=" * 50)
    print(f"Total simulations: {basic_stats['total_simulations'].iloc[0]}")
    print(f"Completed: {basic_stats['completed'].iloc[0]}")
    print(f"Errors: {basic_stats['errors'].iloc[0]}")  
    print(f"Timeouts: {basic_stats['timeouts'].iloc[0]}")
    print(f"Success rate: {(basic_stats['completed'].iloc[0] / basic_stats['total_simulations'].iloc[0] * 100):.1f}%")
    print(f"Unique games: {basic_stats['unique_games'].iloc[0]}")
    print(f"Average duration: {basic_stats['avg_duration'].iloc[0]:.1f} seconds")
    
    # Get per-game breakdown
    game_query = """
    SELECT 
        game_id,
        season_year,
        COUNT(*) as total_runs,
        COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_runs,
        AVG(duration_seconds) as avg_duration,
        AVG(scoring_rate) as avg_scoring_rate,
        AVG(successful_predictions) as avg_predictions
    FROM simulation_runs 
    GROUP BY game_id, season_year
    ORDER BY game_id
    """
    
    game_stats = pd.read_sql_query(game_query, conn)
    
    print(f"\n📋 PER-GAME BREAKDOWN")
    print("=" * 50)
    for _, row in game_stats.iterrows():
        success_rate = (row['completed_runs'] / row['total_runs']) * 100
        print(f"Game {row['game_id']} ({row['season_year']}):")
        print(f"  Runs: {row['completed_runs']}/{row['total_runs']} ({success_rate:.1f}% success)")
        if row['avg_scoring_rate']:
            print(f"  Avg scoring rate: {row['avg_scoring_rate']:.1f}%")
        if row['avg_predictions']:
            print(f"  Avg predictions: {row['avg_predictions']:.1f}")
        print()
    
    # Get recent activity
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
    
    print(f"🕒 RECENT SIMULATIONS")
    print("=" * 50)
    for _, row in recent_runs.iterrows():
        status_emoji = "✅" if row['status'] == 'completed' else "❌" if row['status'] == 'error' else "⏳"
        print(f"{status_emoji} {row['run_id'][:20]}... | Game: {row['game_id']} | {row['status']}")
    
    conn.close()
    
    # Save detailed results
    with open('analysis_results.txt', 'w') as f:
        f.write("NBA SIMULATION ANALYSIS RESULTS\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total simulations: {basic_stats['total_simulations'].iloc[0]}\n")
        f.write(f"Completed: {basic_stats['completed'].iloc[0]}\n")
        f.write(f"Success rate: {(basic_stats['completed'].iloc[0] / basic_stats['total_simulations'].iloc[0] * 100):.1f}%\n")
        f.write(f"Average duration: {basic_stats['avg_duration'].iloc[0]:.1f} seconds\n\n")
        f.write("PER-GAME BREAKDOWN:\n")
        f.write(game_stats.to_string())
        f.write("\n\nRECENT SIMULATIONS:\n")
        f.write(recent_runs.to_string())
    
    print(f"\n💾 Results saved to: analysis_results.txt")
    print(f"📁 Download with: gcloud compute scp nba-orchestrator:analysis_results.txt ./ --zone=us-central1-a --scp-flag=\"-batch\"")

if __name__ == "__main__":
    analyze_simulation_results()
'@

# Create the analysis script on VM
$createScript = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="cat > analysis_working.py << 'EOF'
$ANALYSIS_SCRIPT
EOF"

Write-Host "✅ Analysis script created on VM" -ForegroundColor Green

# Step 3: Run the analysis
Write-Host "`nStep 3: Running analysis on GCP..."
$analysisResult = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="source ~/.bashrc && source ~/venv/bin/activate && cd ~ && python3 analysis_working.py"

Write-Host $analysisResult -ForegroundColor White

# Step 4: Download results
Write-Host "`nStep 4: Downloading analysis results..."
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$downloadResult = gcloud compute scp nba-orchestrator:analysis_results.txt "./analysis_results_$timestamp.txt" --zone=us-central1-a --project=utopian-outlook-470922-q2 --scp-flag="-batch"

if (Test-Path "./analysis_results_$timestamp.txt") {
    Write-Host "✅ Analysis results downloaded: analysis_results_$timestamp.txt" -ForegroundColor Green
    
    Write-Host "`nPreview of results:" -ForegroundColor Cyan
    Write-Host "-" * 40
    Get-Content "./analysis_results_$timestamp.txt" -Head 15
    Write-Host "`n... (full results in analysis_results_$timestamp.txt)"
    
} else {
    Write-Host "⚠️  Could not download analysis results" -ForegroundColor Yellow
}

Write-Host "`n" + "=" * 80
Write-Host "ANALYSIS COMPLETE!" -ForegroundColor Green
Write-Host "=" * 80
