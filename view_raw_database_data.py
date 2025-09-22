#!/usr/bin/env python3
"""
Simple script to view raw database data as a DataFrame for visual inspection.

This script connects to the simulation results database and displays all raw data
in a clean, readable format without any filtering or analysis.
"""

import pandas as pd
import sqlite3
from pathlib import Path

# Database configuration - change this path if needed
DATABASE_PATH = "enhanced_simulation_results_multithreaded_current.db"

def view_raw_data(db_path=DATABASE_PATH):
    """
    Load and display all raw data from the database as a DataFrame.
    
    Args:
        db_path: Path to the SQLite database file
    """
    
    # Check if database exists
    if not Path(db_path).exists():
        print(f"❌ Database file not found: {db_path}")
        return None
    
    try:
        # Connect to database
        print(f"📂 Reading data from: {db_path}")
        conn = sqlite3.connect(db_path)
        
        # Get all raw simulation data
        query = """
        SELECT 
            game_id,
            season_year,
            status,
            successful_predictions,
            duration_seconds,
            final_score,
            created_at
        FROM simulation_runs
        ORDER BY game_id, created_at
        """
        
        # Load data into DataFrame
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            print("❌ No data found in the database")
            return None
        
        # Display basic info
        print(f"\n📊 Database Overview:")
        print(f"   Total records: {len(df)}")
        print(f"   Date range: {df['created_at'].min()} to {df['created_at'].max()}")
        print(f"   Unique games: {df['game_id'].nunique()}")
        print(f"   Status breakdown:")
        print(f"      {df['status'].value_counts().to_dict()}")
        
        # Convert created_at to readable format
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Round duration to 2 decimal places for readability
        if 'duration_seconds' in df.columns:
            df['duration_minutes'] = (df['duration_seconds'] / 60).round(2)
        
        # Add some helpful derived columns
        df['predictions_range'] = pd.cut(df['successful_predictions'], 
                                       bins=[0, 100, 200, 300, 400, 500, 600, float('inf')],
                                       labels=['0-100', '101-200', '201-300', '301-400', '401-500', '501-600', '600+'])
        
        print(f"\n   Prediction count distribution:")
        print(f"      {df['predictions_range'].value_counts().to_dict()}")
        
        # Set pandas display options for better viewing
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
        
        print(f"\n" + "="*100)
        print("🔍 RAW DATABASE DATA:")
        print("="*100)
        
        # Display the full DataFrame
        print(df)
        
        # Additional summary statistics
        print(f"\n" + "="*100)
        print("📈 SUMMARY STATISTICS:")
        print("="*100)
        
        print(f"\nSuccessful Predictions Statistics:")
        print(f"   Mean: {df['successful_predictions'].mean():.1f}")
        print(f"   Median: {df['successful_predictions'].median():.1f}")
        print(f"   Min: {df['successful_predictions'].min()}")
        print(f"   Max: {df['successful_predictions'].max()}")
        print(f"   Std Dev: {df['successful_predictions'].std():.1f}")
        
        print(f"\nDuration Statistics (minutes):")
        print(f"   Mean: {df['duration_minutes'].mean():.1f}")
        print(f"   Median: {df['duration_minutes'].median():.1f}")
        print(f"   Min: {df['duration_minutes'].min():.1f}")
        print(f"   Max: {df['duration_minutes'].max():.1f}")
        
        print(f"\nGames by Season:")
        season_counts = df['season_year'].value_counts().sort_index()
        for season, count in season_counts.items():
            print(f"   {season}: {count} simulations")
        
        return df
        
    except Exception as e:
        print(f"❌ Error reading database: {e}")
        return None

def view_table_schema(db_path=DATABASE_PATH):
    """
    Display the database table structure.
    """
    if not Path(db_path).exists():
        print(f"❌ Database file not found: {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Get table names
        tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
        tables_df = pd.read_sql_query(tables_query, conn)
        
        print(f"\n📋 Database Tables:")
        for table_name in tables_df['name']:
            print(f"   - {table_name}")
            
            # Get schema for each table
            schema_query = f"PRAGMA table_info({table_name})"
            schema_df = pd.read_sql_query(schema_query, conn)
            
            print(f"     Columns:")
            for _, row in schema_df.iterrows():
                nullable = "NULL" if row['notnull'] == 0 else "NOT NULL"
                pk = "PRIMARY KEY" if row['pk'] == 1 else ""
                print(f"       {row['name']} ({row['type']}) {nullable} {pk}".strip())
            print()
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error reading database schema: {e}")

if __name__ == "__main__":
    print("🏀 NBA Simulation Results - Raw Data Viewer")
    print("="*60)
    
    # First show the database structure
    print("\n1. Database Schema:")
    view_table_schema()
    
    # Then show the raw data
    print("\n2. Raw Data:")
    raw_data = view_raw_data()
    
    if raw_data is not None:
        print(f"\n✅ Successfully loaded {len(raw_data)} records")
        print("💡 Tip: You can modify DATABASE_PATH at the top of this script to view other database files")
    
    # Ask if user wants to save to CSV
    try:
        save_csv = input("\n💾 Would you like to save this data to CSV? (y/n): ").lower().strip()
        if save_csv in ['y', 'yes'] and raw_data is not None:
            csv_filename = f"raw_simulation_data_{DATABASE_PATH.replace('.db', '')}.csv"
            raw_data.to_csv(csv_filename, index=False)
            print(f"✅ Data saved to: {csv_filename}")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
