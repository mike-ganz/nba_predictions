#!/usr/bin/env pwsh
# NBA Results Analysis on GCP
# Uploads analysis scripts and runs comprehensive results analysis

Write-Host "=" * 80
Write-Host "RUNNING NBA RESULTS ANALYSIS ON GCP"
Write-Host "=" * 80

$BUCKET_NAME = (Get-Content "bucket_name.txt" -Raw).Trim()
Write-Host "Using bucket: $BUCKET_NAME"

# Step 1: Upload analysis files to VM
Write-Host "`nStep 1: Uploading analysis files..."

# Upload main analysis script
if (Test-Path "results_analysis.py") {
    gcloud compute scp results_analysis.py nba-orchestrator:~/ --zone=us-central1-a --project=utopian-outlook-470922-q2 --scp-flag="-batch"
    Write-Host "SUCCESS: results_analysis.py uploaded"
} else {
    Write-Host "WARNING: results_analysis.py not found locally"
}

# Upload notebook functions (required dependency)
if (Test-Path "nba_results_notebook.py") {
    gcloud compute scp nba_results_notebook.py nba-orchestrator:~/ --zone=us-central1-a --project=utopian-outlook-470922-q2 --scp-flag="-batch"
    Write-Host "SUCCESS: nba_results_notebook.py uploaded"
} else {
    Write-Host "WARNING: nba_results_notebook.py not found locally"
}

# Upload any actual game results data if available
if (Test-Path "data/game_results_2024-2025.csv") {
    Write-Host "Uploading actual game results data..."
    gcloud compute scp data/game_results_2024-2025.csv nba-orchestrator:~/data/ --zone=us-central1-a --project=utopian-outlook-470922-q2 --scp-flag="-batch"
    Write-Host "SUCCESS: Game results data uploaded"
}

# Step 2: Install required Python packages on VM
Write-Host "`nStep 2: Installing analysis packages..."
gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="source ~/.bashrc && source ~/venv/bin/activate && pip install matplotlib seaborn tabulate"

# Step 3: Create simplified analysis script that works in terminal
Write-Host "`nStep 3: Creating terminal-friendly analysis script..."

$TERMINAL_ANALYSIS = @'
#!/usr/bin/env python3
"""
Terminal-friendly NBA Results Analysis
Runs analysis and saves results to text files (no GUI plots)
"""

import pandas as pd
import sqlite3
import sys
from pathlib import Path

def load_simulation_results(db_path="simulation_results.db"):
    """Load simulation results from database."""
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return pd.DataFrame()
    
    conn = sqlite3.connect(db_path)
    query = """
    SELECT 
        run_id, game_id, season_year, platform, status,
        start_time, end_time, duration_seconds,
        total_predictions, successful_predictions,
        final_score, final_quarter, final_time,
        termination_reason, scoring_plays, non_scoring_plays, scoring_rate
    FROM simulation_runs 
    ORDER BY game_id, start_time
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def analyze_results():
    """Run comprehensive analysis and save to files."""
    print("📊 Loading simulation results...")
    
    # Load results
    results_df = load_simulation_results()
    
    if results_df.empty:
        print("❌ No simulation results found!")
        return
    
    print(f"📈 Found {len(results_df)} simulation runs")
    
    # Basic statistics
    print(f"\n🏀 SIMULATION SUMMARY")
    print("=" * 50)
    
    total_runs = len(results_df)
    completed_runs = len(results_df[results_df['status'] == 'completed'])
    success_rate = (completed_runs / total_runs) * 100 if total_runs > 0 else 0
    
    print(f"Total simulation runs: {total_runs}")
    print(f"Completed runs: {completed_runs}")
    print(f"Success rate: {success_rate:.1f}%")
    
    # Game breakdown
    games_summary = results_df.groupby('game_id').agg({
        'run_id': 'count',
        'status': lambda x: (x == 'completed').sum(),
        'duration_seconds': 'mean',
        'scoring_rate': 'mean'
    }).round(2)
    
    games_summary.columns = ['Total_Runs', 'Completed_Runs', 'Avg_Duration_Sec', 'Avg_Scoring_Rate']
    games_summary['Success_Rate'] = (games_summary['Completed_Runs'] / games_summary['Total_Runs'] * 100).round(1)
    
    print(f"\n📋 PER-GAME BREAKDOWN")
    print("=" * 50)
    print(games_summary.to_string())
    
    # Completed runs analysis
    completed_df = results_df[results_df['status'] == 'completed']
    if not completed_df.empty:
        print(f"\n🎯 COMPLETED RUNS ANALYSIS")
        print("=" * 50)
        print(f"Average duration: {completed_df['duration_seconds'].mean():.1f} seconds")
        print(f"Average predictions per run: {completed_df['total_predictions'].mean():.1f}")
        print(f"Average successful predictions: {completed_df['successful_predictions'].mean():.1f}")
        print(f"Average scoring rate: {completed_df['scoring_rate'].mean():.1f}%")
        
        # Termination reasons
        termination_counts = completed_df['termination_reason'].value_counts()
        print(f"\n🛑 TERMINATION REASONS")
        print("=" * 50)
        for reason, count in termination_counts.items():
            print(f"{reason}: {count} runs ({count/len(completed_df)*100:.1f}%)")
    
    # Save detailed results to file
    output_file = "analysis_results.txt"
    with open(output_file, 'w') as f:
        f.write("NBA SIMULATION RESULTS ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total simulation runs: {total_runs}\n")
        f.write(f"Completed runs: {completed_runs}\n")
        f.write(f"Success rate: {success_rate:.1f}%\n\n")
        f.write("PER-GAME BREAKDOWN:\n")
        f.write(games_summary.to_string())
        f.write("\n\n")
        
        if not completed_df.empty:
            f.write("COMPLETED RUNS DETAILS:\n")
            f.write(completed_df.to_string())
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    print(f"📁 Download with: gcloud compute scp nba-orchestrator:~/{output_file} ./ --zone=us-central1-a --scp-flag=\"-batch\"")

if __name__ == "__main__":
    analyze_results()
'@

# Upload the terminal analysis script
$TERMINAL_ANALYSIS | Out-File -FilePath "terminal_analysis.py" -Encoding UTF8
gcloud compute scp terminal_analysis.py nba-orchestrator:~/ --zone=us-central1-a --project=utopian-outlook-470922-q2 --scp-flag="-batch"
Remove-Item "terminal_analysis.py"

# Step 4: Run the analysis
Write-Host "`nStep 4: Running results analysis on GCP..."
gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="source ~/.bashrc && source ~/venv/bin/activate && cd ~ && python3 terminal_analysis.py"

# Step 5: Download results
Write-Host "`nStep 5: Downloading analysis results..."
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
gcloud compute scp nba-orchestrator:~/analysis_results.txt "./analysis_results_$timestamp.txt" --zone=us-central1-a --project=utopian-outlook-470922-q2 --scp-flag="-batch"

if (Test-Path "./analysis_results_$timestamp.txt") {
    Write-Host "SUCCESS: Analysis results downloaded to analysis_results_$timestamp.txt"
    Write-Host "`nPreview of results:"
    Write-Host "-" * 40
    Get-Content "./analysis_results_$timestamp.txt" -Head 20
    Write-Host "`n... (full results in analysis_results_$timestamp.txt)"
} else {
    Write-Host "WARNING: Could not download analysis results file"
}

Write-Host "`n" + "=" * 80
Write-Host "ANALYSIS COMPLETE!"
Write-Host "=" * 80
Write-Host "Next steps:"
Write-Host "1. Review analysis_results_$timestamp.txt for detailed breakdown"
Write-Host "2. Run betting analysis with your actual game results CSV"
Write-Host "3. Use ./monitor.ps1 to check VM status and logs"
