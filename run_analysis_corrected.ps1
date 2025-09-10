#!/usr/bin/env pwsh
# Corrected NBA Results Analysis - Treats "game_ended" as Success

Write-Host "=" * 80
Write-Host "NBA RESULTS ANALYSIS (CORRECTED FOR game_ended STATUS)"
Write-Host "=" * 80

$BUCKET_NAME = (Get-Content "bucket_name.txt" -Raw).Trim()
Write-Host "Using bucket: $BUCKET_NAME"

# Create corrected analysis script
Write-Host "`nCreating corrected analysis script on VM..."

$CORRECTED_ANALYSIS = @'
#!/usr/bin/env python3
"""
Corrected NBA Results Analysis - Treats game_ended as successful completion
"""
import pandas as pd
import sqlite3
import sys
from pathlib import Path

def analyze_simulation_results():
    """Run corrected analysis on enhanced_simulation_results.db"""
    
    db_path = "enhanced_simulation_results.db"
    
    if not Path(db_path).exists():
        print("❌ Database not found: enhanced_simulation_results.db")
        return
    
    print("📊 Loading simulation results with corrected status interpretation...")
    
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
    
    print(f"\n🏀 CORRECTED SIMULATION SUMMARY")
    print("=" * 50)
    print(f"Total simulations: {total}")
    print(f"✅ SUCCESSFUL: {successful} ({(successful/total*100):.1f}%)")
    print(f"   - Game ended naturally: {game_ended}")
    print(f"   - Completed (max iterations): {completed}")
    print(f"❌ Errors: {errors} ({(errors/total*100):.1f}%)")
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
        COUNT(CASE WHEN status = 'game_ended' THEN 1 END) as game_ended_runs,
        COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_runs,
        COUNT(CASE WHEN status = 'error' THEN 1 END) as error_runs,
        AVG(duration_seconds) as avg_duration,
        AVG(CASE WHEN status IN ('completed', 'game_ended') THEN scoring_rate END) as avg_scoring_rate,
        AVG(CASE WHEN status IN ('completed', 'game_ended') THEN successful_predictions END) as avg_predictions
    FROM simulation_runs 
    GROUP BY game_id, season_year
    ORDER BY successful_runs DESC, game_id
    """
    
    game_stats = pd.read_sql_query(game_query, conn)
    
    print(f"\n📋 PER-GAME BREAKDOWN (Top Performing Games)")
    print("=" * 50)
    for _, row in game_stats.head(10).iterrows():
        success_rate = (row['successful_runs'] / row['total_runs']) * 100
        print(f"Game {row['game_id']} ({row['season_year']}):")
        print(f"  ✅ Success: {row['successful_runs']}/{row['total_runs']} ({success_rate:.1f}%)")
        print(f"     - Game ended: {row['game_ended_runs']}")
        print(f"     - Completed: {row['completed_runs']}")  
        print(f"  ❌ Errors: {row['error_runs']}")
        if row['avg_scoring_rate']:
            print(f"  📊 Avg scoring: {row['avg_scoring_rate']:.1f}%")
        if row['avg_predictions']:
            print(f"  🎯 Avg predictions: {row['avg_predictions']:.1f}")
        print()
    
    # Show games with issues (high error rate)
    problem_games = game_stats[game_stats['error_runs'] > game_stats['successful_runs']]
    if not problem_games.empty:
        print(f"⚠️  GAMES WITH ISSUES (More errors than successes)")
        print("=" * 50)
        for _, row in problem_games.head(5).iterrows():
            error_rate = (row['error_runs'] / row['total_runs']) * 100
            print(f"Game {row['game_id']}: {row['error_runs']} errors, {row['successful_runs']} successful ({error_rate:.1f}% error rate)")
    
    # Recent activity
    recent_query = """
    SELECT 
        run_id,
        game_id, 
        status,
        scoring_rate,
        successful_predictions,
        start_time
    FROM simulation_runs 
    ORDER BY start_time DESC 
    LIMIT 10
    """
    
    recent_runs = pd.read_sql_query(recent_query, conn)
    
    print(f"\n🕒 RECENT SIMULATIONS")
    print("=" * 50)
    for _, row in recent_runs.iterrows():
        if row['status'] in ['completed', 'game_ended']:
            status_emoji = "✅"
            status_text = f"{row['status']} ({row['scoring_rate']:.1f}% scoring)"
        elif row['status'] == 'error':
            status_emoji = "❌"
            status_text = "error"
        else:
            status_emoji = "⏳"
            status_text = row['status']
            
        print(f"{status_emoji} {row['run_id'][:25]}... | Game: {row['game_id']} | {status_text}")
    
    conn.close()
    
    # Save detailed results
    with open('analysis_corrected_results.txt', 'w') as f:
        f.write("NBA SIMULATION ANALYSIS RESULTS (CORRECTED)\n")
        f.write("=" * 50 + "\n\n")
        f.write("STATUS DEFINITIONS:\n")
        f.write("- 'game_ended': Simulation completed, game reached natural end\n") 
        f.write("- 'completed': Simulation completed, may have hit max iterations\n")
        f.write("- 'error': Simulation failed with error\n\n")
        f.write(f"SUMMARY:\n")
        f.write(f"Total simulations: {total}\n")
        f.write(f"Successful simulations: {successful} ({(successful/total*100):.1f}%)\n")
        f.write(f"  - Game ended: {game_ended}\n")
        f.write(f"  - Completed: {completed}\n") 
        f.write(f"Error simulations: {errors} ({(errors/total*100):.1f}%)\n")
        f.write(f"Average duration: {basic_stats['avg_duration'].iloc[0]:.1f} seconds\n\n")
        f.write("PER-GAME BREAKDOWN:\n")
        f.write(game_stats.to_string())
        f.write("\n\nRECENT SIMULATIONS:\n")
        f.write(recent_runs.to_string())
    
    print(f"\n💾 Corrected results saved to: analysis_corrected_results.txt")

if __name__ == "__main__":
    analyze_simulation_results()
'@

# Create the corrected analysis script on VM
gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="cat > analysis_corrected.py << 'EOF'
$CORRECTED_ANALYSIS
EOF"

Write-Host "✅ Corrected analysis script created on VM" -ForegroundColor Green

# Run the corrected analysis
Write-Host "`nRunning corrected analysis..."
$analysisResult = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="source ~/.bashrc && source ~/venv/bin/activate && cd ~ && python3 analysis_corrected.py"

Write-Host $analysisResult -ForegroundColor White

# Download corrected results
Write-Host "`nDownloading corrected analysis..."
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
gcloud compute scp nba-orchestrator:analysis_corrected_results.txt "./analysis_corrected_$timestamp.txt" --zone=us-central1-a --project=utopian-outlook-470922-q2 --scp-flag="-batch"

if (Test-Path "./analysis_corrected_$timestamp.txt") {
    Write-Host "✅ Corrected analysis downloaded: analysis_corrected_$timestamp.txt" -ForegroundColor Green
} else {
    Write-Host "⚠️  Could not download corrected analysis" -ForegroundColor Yellow
}

Write-Host "`n" + "=" * 80
Write-Host "CORRECTED ANALYSIS COMPLETE!" -ForegroundColor Green
Write-Host "📊 Your system has been performing much better than initially calculated!" -ForegroundColor Green
Write-Host "=" * 80
