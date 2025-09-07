#!/usr/bin/env python3
"""
Quick Error Check Tool

Simple script to quickly check for recent simulation errors.
Perfect for a fast check after running simulations.
"""

from error_analysis import SimulationErrorAnalyzer
from datetime import datetime, timedelta
import argparse


def quick_check(hours_back: int = 24, db_path: str = "enhanced_simulation_results.db"):
    """Perform a quick error check for recent hours."""
    days_back = max(1, hours_back // 24 + 1)  # Convert hours to days (minimum 1 day)
    
    analyzer = SimulationErrorAnalyzer(db_path)
    
    print(f"⚡ QUICK ERROR CHECK - Last {hours_back} hours")
    print("=" * 60)
    print(f"🕒 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get recent data
    error_logs = analyzer.get_database_error_logs(days_back)
    validation_df = analyzer.get_validation_terminations(days_back)
    
    # Filter by hours if needed
    if hours_back < 24:
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        # Filter error logs by timestamp
        recent_errors = []
        for error in error_logs:
            if error.get('timestamp'):
                try:
                    error_time = datetime.strptime(error['timestamp'], '%Y-%m-%d %H:%M:%S,%f')
                    if error_time >= cutoff_time:
                        recent_errors.append(error)
                except:
                    recent_errors.append(error)  # Include if can't parse timestamp
            else:
                recent_errors.append(error)
        error_logs = recent_errors
        
        # Filter validation terminations
        if not validation_df.empty:
            validation_df = validation_df[validation_df['validation_termination_timestamp'] >= cutoff_time]
    
    # Quick summary
    total_errors = len(error_logs)
    total_terminations = len(validation_df) if not validation_df.empty else 0
    
    if total_errors == 0 and total_terminations == 0:
        print("✅ All Clear! No errors or unusual terminations found.")
        return
    
    # Show issues
    if total_errors > 0:
        print(f"🔥 {total_errors} Database Save Errors Found!")
        
        # Group by error type
        error_types = {}
        games_affected = set()
        
        for error in error_logs:
            error_type = error.get('error_type', 'unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
            if error.get('game_id'):
                games_affected.add(error['game_id'])
        
        print(f"   📊 Error Types:")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"      • {error_type}: {count}")
        
        print(f"   🎮 Games Affected: {len(games_affected)}")
        if games_affected:
            games_list = sorted(list(games_affected))[:5]  # Show first 5
            print(f"      {', '.join(games_list)}")
            if len(games_affected) > 5:
                print(f"      ... and {len(games_affected) - 5} more")
        
        # Show most recent errors
        print(f"\n   🕒 Most Recent Errors:")
        for i, error in enumerate(error_logs[:3]):
            timestamp = error.get('timestamp', 'N/A')
            message = error.get('message', '')[:80] + "..." if len(error.get('message', '')) > 80 else error.get('message', '')
            print(f"      {i+1}. [{timestamp}] {message}")
    
    if total_terminations > 0:
        print(f"\n🛑 {total_terminations} Validation Terminations Found!")
        
        # Group by termination type
        termination_types = validation_df['validation_termination_type'].value_counts()
        print(f"   📊 Termination Types:")
        for term_type, count in termination_types.items():
            print(f"      • {term_type}: {count}")
        
        # Show recent terminations
        print(f"\n   🕒 Most Recent Terminations:")
        for i, (_, row) in enumerate(validation_df.head(3).iterrows()):
            game_id = row['game_id']
            term_type = row['validation_termination_type']
            trigger = row['validation_trigger_condition']
            print(f"      {i+1}. Game {game_id}: {term_type} ({trigger})")
    
    print(f"\n💡 For detailed analysis, run:")
    print(f"   python error_analysis.py --days {days_back}")
    if total_errors > 0:
        print(f"   python error_analysis.py --database-errors-only --days {days_back}")


def main():
    parser = argparse.ArgumentParser(description='Quick Error Check Tool')
    parser.add_argument('--hours', type=int, default=24,
                       help='Number of hours back to check (default: 24)')
    parser.add_argument('--db', type=str, default='enhanced_simulation_results.db',
                       help='Path to simulation results database')
    
    args = parser.parse_args()
    
    quick_check(args.hours, args.db)


if __name__ == "__main__":
    main()
