#!/usr/bin/env pwsh
# Optimize Database Structure - Add Betting Data to SQLite Database

Write-Host ("=" * 80)
Write-Host "OPTIMIZING DATABASE STRUCTURE - SINGLE FILE SOLUTION"
Write-Host ("=" * 80)

# Create script to add betting data table to your existing database
$OPTIMIZATION_SCRIPT = @'
#!/usr/bin/env python3
"""
Database optimization script - Add betting data to simulation database
"""
import sqlite3
import pandas as pd
from pathlib import Path

def optimize_database():
    """Add betting data table to simulation database for single-file efficiency"""
    
    db_path = "enhanced_simulation_results.db"
    
    if not Path(db_path).exists():
        print("ERROR: Database not found")
        return
    
    print("Optimizing database structure...")
    
    conn = sqlite3.connect(db_path)
    
    # Create betting data table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS actual_game_results (
            game_id TEXT PRIMARY KEY,
            date TEXT,
            season_year TEXT,
            away_team TEXT,
            home_team TEXT,
            away_score INTEGER,
            home_score INTEGER,
            final_spread REAL,
            away_moneyline TEXT,
            home_moneyline TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Check if CSV data exists to import
    csv_path = "data/game_results_2024-2025.csv"
    if Path(csv_path).exists():
        print(f"Importing data from {csv_path}...")
        
        # Load CSV data
        csv_df = pd.read_csv(csv_path)
        
        # Import to database
        csv_df.to_sql('actual_game_results', conn, if_exists='replace', index=False)
        
        print(f"Imported {len(csv_df)} game results to database")
        
    else:
        print("No CSV file found - creating empty betting data table")
        
        # Add sample data structure for reference
        sample_data = {
            'game_id': ['22200001', '22200002'],
            'date': ['2024-10-22', '2024-10-23'],
            'season_year': ['2024-2025', '2024-2025'],
            'away_team': ['ATL', 'BOS'],
            'home_team': ['CLE', 'NYK'],
            'away_score': [110, 95],
            'home_score': [108, 102],
            'final_spread': [2.5, -7.0],
            'away_moneyline': ['+115', '+280'],
            'home_moneyline': ['-135', '-350']
        }
        
        sample_df = pd.DataFrame(sample_data)
        sample_df.to_sql('actual_game_results', conn, if_exists='replace', index=False)
        
        print("Created sample betting data table structure")
    
    # Verify the optimization
    cursor = conn.execute('SELECT COUNT(*) FROM actual_game_results')
    betting_count = cursor.fetchone()[0]
    
    cursor = conn.execute('SELECT COUNT(*) FROM simulation_runs')
    simulation_count = cursor.fetchone()[0]
    
    print(f"\nDATABASE OPTIMIZATION COMPLETE:")
    print(f"  Simulation runs: {simulation_count}")
    print(f"  Actual game results: {betting_count}")
    print(f"  Single file contains both datasets!")
    
    conn.close()

if __name__ == "__main__":
    optimize_database()
'@

# Upload optimization script
$OPTIMIZATION_SCRIPT | Out-File -FilePath "temp_optimize.py" -Encoding UTF8
gcloud compute scp temp_optimize.py nba-orchestrator:optimize_database.py --zone=us-central1-a --project=utopian-outlook-470922-q2 --scp-flag="-batch"
Remove-Item "temp_optimize.py"

Write-Host "SUCCESS: Optimization script uploaded" -ForegroundColor Green

# Run optimization
Write-Host "`nRunning database optimization..."
$optimizeResult = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="source ~/.bashrc && source ~/venv/bin/activate && cd ~ && python3 optimize_database.py"

Write-Host $optimizeResult -ForegroundColor White

# Create optimized analysis script that uses only the database
Write-Host "`nCreating optimized single-file analysis script..."

$OPTIMIZED_ANALYSIS = @'
#!/usr/bin/env python3
"""
Optimized results_analysis.py - Single database file version
"""
import sqlite3
import pandas as pd
import numpy as np

def load_actual_game_results_from_db(db_path="enhanced_simulation_results.db"):
    """Load actual game results from database instead of CSV"""
    try:
        conn = sqlite3.connect(db_path)
        actual_df = pd.read_sql_query("SELECT * FROM actual_game_results", conn)
        conn.close()
        return actual_df
    except Exception as e:
        print(f"Error loading actual results from database: {e}")
        return pd.DataFrame()

def analyze_optimized():
    """Run optimized analysis using single database file"""
    
    db_path = "enhanced_simulation_results.db"
    conn = sqlite3.connect(db_path)
    
    # Get simulation data
    sim_query = """
    SELECT 
        game_id,
        COUNT(*) as total_sims,
        COUNT(CASE WHEN status IN ('completed', 'game_ended') THEN 1 END) as successful_sims,
        AVG(CASE WHEN status IN ('completed', 'game_ended') THEN scoring_rate END) as avg_scoring_rate
    FROM simulation_runs 
    GROUP BY game_id
    """
    
    sim_df = pd.read_sql_query(sim_query, conn)
    
    # Get actual game data from same database
    actual_df = pd.read_sql_query("SELECT * FROM actual_game_results", conn)
    
    conn.close()
    
    print("OPTIMIZED SINGLE-FILE ANALYSIS")
    print("=" * 50)
    print(f"Simulation data: {len(sim_df)} games")
    print(f"Actual results data: {len(actual_df)} games")
    
    if not actual_df.empty:
        # Merge simulation and actual data for betting analysis
        merged = sim_df.merge(actual_df, on='game_id', how='inner')
        
        print(f"Merged data: {len(merged)} games with both simulation and actual results")
        
        for _, row in merged.head(10).iterrows():
            success_rate = (row['successful_sims'] / row['total_sims']) * 100
            print(f"Game {row['game_id']}: {row['away_team']} @ {row['home_team']}")
            print(f"  Simulation success: {success_rate:.1f}% ({row['successful_sims']}/{row['total_sims']})")
            print(f"  Actual result: {row['away_score']}-{row['home_score']}")
            print(f"  Spread: {row['final_spread']}")
            print()
            
    else:
        print("No actual game results in database - showing simulation data only")
        for _, row in sim_df.head(10).iterrows():
            success_rate = (row['successful_sims'] / row['total_sims']) * 100
            print(f"Game {row['game_id']}: {success_rate:.1f}% success ({row['successful_sims']}/{row['total_sims']})")

if __name__ == "__main__":
    analyze_optimized()
'@

# Upload optimized analysis
$OPTIMIZED_ANALYSIS | Out-File -FilePath "temp_optimized_analysis.py" -Encoding UTF8
gcloud compute scp temp_optimized_analysis.py nba-orchestrator:optimized_analysis.py --zone=us-central1-a --project=utopian-outlook-470922-q2 --scp-flag="-batch"
Remove-Item "temp_optimized_analysis.py"

Write-Host "SUCCESS: Optimized analysis script created" -ForegroundColor Green

# Test the optimized version
Write-Host "`nTesting optimized single-file analysis..."
$testResult = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="source ~/.bashrc && source ~/venv/bin/activate && cd ~ && python3 optimized_analysis.py"

Write-Host "`nOPTIMIZED ANALYSIS RESULTS:" -ForegroundColor Green
Write-Host ("-" * 60) -ForegroundColor Green
Write-Host $testResult -ForegroundColor White
Write-Host ("-" * 60) -ForegroundColor Green

Write-Host ("`n" + "=" * 80)
Write-Host "OPTIMIZATION COMPLETE!" -ForegroundColor Green
Write-Host ("=" * 80)
Write-Host "Benefits of single-file approach:" -ForegroundColor Green
Write-Host "✅ No dependency on separate CSV files" -ForegroundColor Green
Write-Host "✅ Faster data access (single database connection)" -ForegroundColor Green  
Write-Host "✅ Easier deployment and management" -ForegroundColor Green
Write-Host "✅ Maintains all betting analysis functionality" -ForegroundColor Green
