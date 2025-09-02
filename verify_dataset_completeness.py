#!/usr/bin/env python3
"""
Dataset Completeness Verification
=================================
Verifies that the generated JSONL training data contains the complete 2023-24 NBA season.

Checks:
- Row count comparison with original play-by-play data
- Date range coverage
- Game coverage
- Data quality metrics
"""

import json
import pandas as pd
import os
from datetime import datetime, date
from typing import Dict, Any, Tuple, List
import glob

def load_original_playbyplay_data() -> pd.DataFrame:
    """Load the original play-by-play data for comparison."""
    
    print("📊 Loading original play-by-play data...")
    
    # Try common file locations for play-by-play data
    possible_paths = [
        "data/play_by_play/historical/[10-24-2023]-[06-17-2024]-combined-stats.csv",  # Found 2023-24 season file
        "data/raw/NBA_2023-2024_season.csv",
        "data/raw/nba_2023_2024_season.csv", 
        "data/raw/NBA-2023-2024-season.csv",
        "data/raw/play_by_play_2023_2024.csv"
    ]
    
    # Also check for any CSV files in data/raw/
    raw_dir = "data/raw/"
    if os.path.exists(raw_dir):
        csv_files = glob.glob(f"{raw_dir}*.csv")
        print(f"🔍 Found CSV files in {raw_dir}: {[os.path.basename(f) for f in csv_files]}")
        possible_paths.extend(csv_files)
    
    # Try to load from any available path
    for path in possible_paths:
        if os.path.exists(path):
            print(f"📂 Loading from: {path}")
            try:
                df = pd.read_csv(path)
                print(f"✅ Loaded {len(df):,} rows from original data")
                return df
            except Exception as e:
                print(f"⚠️ Error loading {path}: {e}")
                continue
    
    # If no raw CSV found, try to reconstruct from config
    print("🔄 No raw CSV found, attempting to load from data loaders...")
    try:
        from data.loaders import data_loader
        df = data_loader.load_season_data("2023-2024")
        print(f"✅ Loaded {len(df):,} rows from data loader")
        return df
    except Exception as e:
        print(f"⚠️ Could not load via data loader: {e}")
        return None

def load_generated_csv() -> pd.DataFrame:
    """Load the generated training CSV for comparison."""
    
    print("\n📊 Loading generated training CSV...")
    
    csv_path = "data/training/SAMPLE_optimized_season.csv"
    if os.path.exists(csv_path):
        print(f"📂 Loading from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df):,} rows from generated CSV")
        return df
    else:
        print(f"❌ Generated CSV not found: {csv_path}")
        return None

def load_generated_jsonl() -> List[Dict]:
    """Load the generated JSONL training data."""
    
    print("\n📊 Loading generated JSONL data...")
    
    # Check for corrected file first, then fallback to incremental
    corrected_path = "data/training/CORRECTED_full_season_2023_2024.jsonl"
    incremental_path = "data/training/INCREMENTAL_full_season_2023_2024.jsonl"
    
    if os.path.exists(corrected_path):
        jsonl_path = corrected_path
        print(f"🎯 Using CORRECTED JSONL file")
    elif os.path.exists(incremental_path):
        jsonl_path = incremental_path 
        print(f"📂 Using incremental JSONL file")
    else:
        print(f"❌ No JSONL files found")
        return []
    
    if not os.path.exists(jsonl_path):
        print(f"❌ JSONL file not found: {jsonl_path}")
        return []
    
    print(f"📂 Loading from: {jsonl_path}")
    
    training_examples = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        example = json.loads(line.strip())
                        training_examples.append(example)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Invalid JSON on line {line_num}: {e}")
        
        print(f"✅ Loaded {len(training_examples):,} training examples from JSONL")
        return training_examples
        
    except Exception as e:
        print(f"❌ Error loading JSONL: {e}")
        return []

def analyze_date_coverage(original_df: pd.DataFrame, csv_df: pd.DataFrame, jsonl_data: List[Dict]) -> Dict[str, Any]:
    """Analyze date range coverage."""
    
    print("\n📅 Analyzing date coverage...")
    
    results = {
        'original': {},
        'csv': {},
        'jsonl': {}
    }
    
    # Original data dates
    if original_df is not None and 'game_date' in original_df.columns:
        original_dates = pd.to_datetime(original_df['game_date'])
        results['original'] = {
            'start_date': original_dates.min().strftime('%Y-%m-%d'),
            'end_date': original_dates.max().strftime('%Y-%m-%d'),
            'unique_dates': original_dates.dt.date.nunique(),
            'date_range_days': (original_dates.max() - original_dates.min()).days
        }
        print(f"📊 Original data: {results['original']['start_date']} to {results['original']['end_date']}")
    
    # CSV data dates  
    if csv_df is not None and 'game_date' in csv_df.columns:
        csv_dates = pd.to_datetime(csv_df['game_date'])
        results['csv'] = {
            'start_date': csv_dates.min().strftime('%Y-%m-%d'),
            'end_date': csv_dates.max().strftime('%Y-%m-%d'), 
            'unique_dates': csv_dates.dt.date.nunique(),
            'date_range_days': (csv_dates.max() - csv_dates.min()).days
        }
        print(f"📊 CSV data: {results['csv']['start_date']} to {results['csv']['end_date']}")
    
    # Extract dates from JSONL data (from user messages)
    if jsonl_data:
        jsonl_dates = []
        for example in jsonl_data[:1000]:  # Sample first 1000 to avoid processing all
            try:
                user_content = json.loads(example['messages'][0]['content'])
                # Try to extract date from recent plays or other context
                if 'recent_plays' in user_content and user_content['recent_plays']:
                    # Would need to extract from game context - this is approximate
                    pass
            except:
                continue
        
        if jsonl_dates:
            results['jsonl'] = {
                'sample_size': len(jsonl_dates),
                'date_coverage': 'analyzed'
            }
        else:
            results['jsonl'] = {'note': 'Date extraction from JSONL requires more complex parsing'}
    
    return results

def calculate_expected_training_examples(original_count: int, n_total: int = None) -> Tuple[int, int]:
    """Calculate expected number of training examples based on original data."""
    
    # Use default if not specified
    if n_total is None:
        from config.settings import DEFAULT_N_TOTAL_PLAYS
        n_total = DEFAULT_N_TOTAL_PLAYS
    
    # Each game needs at least n_total plays to start generating training examples
    # Assume average game has ~400-500 plays, so we lose ~n_total plays per game for context
    # Estimate ~1200 games in a season (30 teams * 82 games / 2)
    
    estimated_games = 1200  # Conservative estimate
    context_loss = estimated_games * n_total  # Plays lost to context requirements
    
    # Additional losses from filtering (NaN descriptions, etc.) - estimate 5%
    filter_loss = int(original_count * 0.05)
    
    # Expected range
    min_expected = original_count - context_loss - filter_loss
    max_expected = original_count - context_loss
    
    return min_expected, max_expected

def generate_verification_report(original_df: pd.DataFrame, csv_df: pd.DataFrame, 
                               jsonl_data: List[Dict]) -> Dict[str, Any]:
    """Generate comprehensive verification report."""
    
    print("\n📋 Generating verification report...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'counts': {},
        'coverage': {},
        'quality': {},
        'verdict': 'unknown'
    }
    
    # Count analysis
    original_count = len(original_df) if original_df is not None else 0
    csv_count = len(csv_df) if csv_df is not None else 0
    jsonl_count = len(jsonl_data)
    
    report['counts'] = {
        'original_plays': original_count,
        'csv_training_examples': csv_count,
        'jsonl_training_examples': jsonl_count
    }
    
    if original_count > 0:
        min_expected, max_expected = calculate_expected_training_examples(original_count)
        report['counts']['expected_range'] = {
            'min': min_expected,
            'max': max_expected
        }
        
        # Check if counts are in expected range
        csv_in_range = min_expected <= csv_count <= max_expected if csv_count > 0 else False
        jsonl_in_range = min_expected <= jsonl_count <= max_expected if jsonl_count > 0 else False
        
        report['counts']['csv_in_expected_range'] = csv_in_range
        report['counts']['jsonl_in_expected_range'] = jsonl_in_range
    
    # Date coverage analysis
    report['coverage'] = analyze_date_coverage(original_df, csv_df, jsonl_data)
    
    # Quality checks
    if jsonl_data:
        # Sample quality check on first 100 examples
        valid_examples = 0
        sample_size = min(100, len(jsonl_data))
        
        for i, example in enumerate(jsonl_data[:sample_size]):
            try:
                if ('messages' in example and 
                    len(example['messages']) == 2 and
                    example['messages'][0]['role'] == 'user' and
                    example['messages'][1]['role'] == 'assistant'):
                    
                    # Try to parse content
                    user_content = json.loads(example['messages'][0]['content'])
                    assistant_content = json.loads(example['messages'][1]['content'])
                    
                    if ('away_team' in user_content and 'home_team' in user_content and
                        'players' in user_content and 'recent_plays' in user_content and
                        'next_play' in assistant_content):
                        valid_examples += 1
            except:
                continue
        
        report['quality'] = {
            'sample_size': sample_size,
            'valid_examples': valid_examples,
            'quality_rate': valid_examples / sample_size if sample_size > 0 else 0
        }
    
    # Overall verdict
    if jsonl_count > 0 and report['counts'].get('jsonl_in_expected_range', False):
        if report['quality'].get('quality_rate', 0) > 0.9:
            report['verdict'] = 'complete_and_high_quality'
        else:
            report['verdict'] = 'complete_but_quality_issues'
    elif jsonl_count > 0:
        report['verdict'] = 'generated_but_unexpected_count'
    else:
        report['verdict'] = 'incomplete_or_missing'
    
    return report

def print_verification_summary(report: Dict[str, Any]):
    """Print a human-readable verification summary."""
    
    print("\n" + "="*60)
    print("🔍 DATASET COMPLETENESS VERIFICATION REPORT")
    print("="*60)
    
    counts = report.get('counts', {})
    print(f"\n📊 ROW COUNT ANALYSIS:")
    print(f"   • Original play-by-play: {counts.get('original_plays', 0):,} plays")
    print(f"   • Generated CSV training: {counts.get('csv_training_examples', 0):,} examples")  
    print(f"   • Generated JSONL training: {counts.get('jsonl_training_examples', 0):,} examples")
    
    if 'expected_range' in counts:
        exp_range = counts['expected_range']
        print(f"   • Expected range: {exp_range['min']:,} - {exp_range['max']:,} examples")
        print(f"   • CSV in range: {'✅' if counts.get('csv_in_expected_range') else '❌'}")
        print(f"   • JSONL in range: {'✅' if counts.get('jsonl_in_expected_range') else '❌'}")
    
    quality = report.get('quality', {})
    if quality:
        print(f"\n🎯 QUALITY ANALYSIS:")
        print(f"   • Sample checked: {quality.get('sample_size', 0)} examples")
        print(f"   • Valid examples: {quality.get('valid_examples', 0)}")
        print(f"   • Quality rate: {quality.get('quality_rate', 0):.1%}")
    
    coverage = report.get('coverage', {})
    if coverage.get('original') and coverage.get('csv'):
        orig = coverage['original']
        csv = coverage['csv']
        print(f"\n📅 DATE COVERAGE:")
        print(f"   • Original: {orig['start_date']} to {orig['end_date']} ({orig['unique_dates']} days)")
        print(f"   • Generated: {csv['start_date']} to {csv['end_date']} ({csv['unique_dates']} days)")
    
    verdict = report.get('verdict', 'unknown')
    print(f"\n🎯 OVERALL VERDICT: ", end="")
    
    verdict_messages = {
        'complete_and_high_quality': "✅ COMPLETE & HIGH QUALITY - Dataset ready for training!",
        'complete_but_quality_issues': "⚠️ COMPLETE but some quality issues detected",
        'generated_but_unexpected_count': "⚠️ Data generated but counts outside expected range", 
        'incomplete_or_missing': "❌ INCOMPLETE or missing data files"
    }
    
    print(verdict_messages.get(verdict, "❓ UNKNOWN"))
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if verdict == 'complete_and_high_quality':
        print("   • Your dataset is ready for OpenAI fine-tuning!")
        print("   • Consider running a small test training run to validate")
    elif verdict == 'complete_but_quality_issues':
        print("   • Review the quality issues in your JSONL data")
        print("   • Consider regenerating if quality rate is very low")
    else:
        print("   • Review the count discrepancies above")
        print("   • Check if training data generation completed successfully")

def main():
    """Main verification function."""
    
    print("🔍 NBA Dataset Completeness Verification")
    print("="*50)
    
    # Load all data sources
    original_df = load_original_playbyplay_data()
    csv_df = load_generated_csv()
    jsonl_data = load_generated_jsonl()
    
    # Generate report
    report = generate_verification_report(original_df, csv_df, jsonl_data)
    
    # Print summary
    print_verification_summary(report)
    
    # Save detailed report
    report_path = "data/training/verification_report.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n💾 Detailed report saved: {report_path}")

if __name__ == "__main__":
    main()
