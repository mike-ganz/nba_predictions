#!/usr/bin/env python3
"""
Generate 20-line samples from NBA data files
"""

import pandas as pd
import csv
import os

def sample_csv_file(filepath, num_lines=40):
    """Read first num_lines from a CSV file and save to sample file"""
    filename = os.path.basename(filepath)
    sample_filename = f"sample_{filename}"
    
    print(f"\n{'='*80}")
    print(f"CREATING SAMPLE: {sample_filename}")
    print(f"{'='*80}")
    
    try:
        lines_collected = []
        with open(filepath, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if i >= num_lines:
                    break
                lines_collected.append(line.rstrip())
                print(f"{i+1:2d}: {line.rstrip()}")
        
        # Save to sample file
        with open(sample_filename, 'w', encoding='utf-8') as sample_file:
            sample_file.write('\n'.join(lines_collected))
        
        print(f"\nSample saved to: {sample_filename}")
        
    except Exception as e:
        print(f"Error reading CSV file: {e}")

def sample_excel_file(filepath, num_rows=40):
    """Read first num_rows from an Excel file and save to sample Excel file"""
    filename = os.path.basename(filepath)
    sample_filename = f"sample_{filename}"  # Keep .xlsx extension
    
    print(f"\n{'='*80}")
    print(f"CREATING SAMPLE: {sample_filename}")
    print(f"{'='*80}")
    
    try:
        # Try to read the Excel file
        df = pd.read_excel(filepath, nrows=num_rows)
        
        # Print column headers
        print("Columns:", list(df.columns))
        print("\nData:")
        
        # Print each row with line numbers
        for i, (index, row) in enumerate(df.iterrows(), 1):
            if i > num_rows:
                break
            print(f"{i:2d}: {row.to_dict()}")
        
        # Save to sample Excel file (preserves formatting)
        df.to_excel(sample_filename, index=False, engine='openpyxl')
        print(f"\nSample saved to: {sample_filename}")
            
    except Exception as e:
        print(f"Error reading Excel file: {e}")

def main():
    """Generate samples from all three files"""
    
    # File paths
    csv_file = r"C:\Users\micha\nba_predictions\data\play_by_play\historical\[10-22-2024]-[06-22-2025]-combined-stats.csv"
    player_boxscore_file = r"C:\Users\micha\nba_predictions\data\player_boxscores\historical\NBA-2024-2025-Player-BoxScore-Dataset.xlsx"
    team_boxscore_file = r"C:\Users\micha\nba_predictions\data\team_boxscores\historical\2024-2025_NBA_Box_Score_Team-Stats.xlsx"
    
    print("NBA Data Files Sample Generator")
    print("Generating and saving 40-line samples from each file...")
    
    created_files = []
    
    # Sample CSV file
    if os.path.exists(csv_file):
        sample_csv_file(csv_file, 40)
        created_files.append("sample_[10-22-2024]-[06-22-2025]-combined-stats.csv")
    else:
        print(f"CSV file not found: {csv_file}")
    
    # Sample Excel files
    if os.path.exists(player_boxscore_file):
        sample_excel_file(player_boxscore_file, 40)
        created_files.append("sample_NBA-2024-2025-Player-BoxScore-Dataset.xlsx")
    else:
        print(f"Player boxscore file not found: {player_boxscore_file}")
    
    if os.path.exists(team_boxscore_file):
        sample_excel_file(team_boxscore_file, 40)
        created_files.append("sample_2024-2025_NBA_Box_Score_Team-Stats.xlsx")
    else:
        print(f"Team boxscore file not found: {team_boxscore_file}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY - Sample Files Created:")
    print(f"{'='*80}")
    for file in created_files:
        print(f"✓ {file}")
    print(f"\nTotal files created: {len(created_files)}")

if __name__ == "__main__":
    main()
