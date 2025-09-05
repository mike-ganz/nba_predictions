#!/usr/bin/env python3
"""
NBA Simulation Results Analyzer

Advanced analysis and visualization tools for orchestration results.
Provides detailed statistics, comparisons, and insights from simulation data.
"""

import sqlite3
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from pathlib import Path
import argparse


class SimulationAnalyzer:
    """Advanced analysis tools for simulation results."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.df = self._load_data()
    
    def _load_data(self) -> pd.DataFrame:
        """Load all simulation data into pandas DataFrame."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query('''
                SELECT * FROM simulation_runs 
                ORDER BY game_id, start_time
            ''', conn)
        
        # Convert datetime columns
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])
        
        return df
    
    def game_summary(self) -> pd.DataFrame:
        """Generate summary statistics by game."""
        summary = self.df.groupby('game_id').agg({
            'run_id': 'count',
            'status': lambda x: (x == 'completed').sum(),
            'duration_seconds': ['mean', 'std'],
            'total_predictions': ['mean', 'std'],
            'successful_predictions': ['mean', 'std'],
            'scoring_rate': ['mean', 'std'],
        }).round(2)
        
        # Flatten column names
        summary.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in summary.columns]
        
        # Rename for clarity
        summary = summary.rename(columns={
            'run_id_count': 'total_runs',
            'status_<lambda>': 'completed_runs',
            'duration_seconds_mean': 'avg_duration_sec',
            'duration_seconds_std': 'std_duration_sec',
            'total_predictions_mean': 'avg_total_predictions',
            'total_predictions_std': 'std_total_predictions',
            'successful_predictions_mean': 'avg_successful_predictions',
            'successful_predictions_std': 'std_successful_predictions',
            'scoring_rate_mean': 'avg_scoring_rate',
            'scoring_rate_std': 'std_scoring_rate'
        })
        
        # Calculate success rate
        summary['success_rate'] = (summary['completed_runs'] / summary['total_runs'] * 100).round(1)
        
        return summary
    
    def platform_comparison(self) -> pd.DataFrame:
        """Compare performance across different platforms."""
        if 'platform' not in self.df.columns or self.df['platform'].nunique() < 2:
            return pd.DataFrame({'message': ['Only one platform found in data']})
        
        comparison = self.df.groupby('platform').agg({
            'run_id': 'count',
            'status': lambda x: (x == 'completed').sum(),
            'duration_seconds': 'mean',
            'scoring_rate': 'mean',
            'successful_predictions': 'mean'
        }).round(2)
        
        comparison.columns = ['total_runs', 'completed_runs', 'avg_duration', 'avg_scoring_rate', 'avg_predictions']
        comparison['success_rate'] = (comparison['completed_runs'] / comparison['total_runs'] * 100).round(1)
        
        return comparison
    
    def scoring_patterns_analysis(self) -> Dict[str, Any]:
        """Analyze scoring patterns across simulations."""
        completed = self.df[self.df['status'] == 'completed']
        
        if completed.empty:
            return {"error": "No completed simulations found"}
        
        patterns = {
            "overall_scoring_rate": {
                "mean": completed['scoring_rate'].mean(),
                "median": completed['scoring_rate'].median(),
                "std": completed['scoring_rate'].std(),
                "min": completed['scoring_rate'].min(),
                "max": completed['scoring_rate'].max()
            },
            "by_game": completed.groupby('game_id')['scoring_rate'].agg(['mean', 'std']).round(2).to_dict(),
            "high_scoring_simulations": len(completed[completed['scoring_rate'] > 25]),
            "low_scoring_simulations": len(completed[completed['scoring_rate'] < 15]),
            "total_completed": len(completed)
        }
        
        return patterns
    
    def duration_analysis(self) -> Dict[str, Any]:
        """Analyze simulation durations and performance."""
        analysis = {
            "overall_duration": {
                "mean_minutes": self.df['duration_seconds'].mean() / 60,
                "median_minutes": self.df['duration_seconds'].median() / 60,
                "std_minutes": self.df['duration_seconds'].std() / 60
            },
            "by_status": self.df.groupby('status')['duration_seconds'].agg(['count', 'mean']).round(2).to_dict(),
            "by_game": self.df.groupby('game_id')['duration_seconds'].mean().round(2).to_dict()
        }
        
        return analysis
    
    def failure_analysis(self) -> Dict[str, Any]:
        """Analyze failed simulations and error patterns."""
        failed = self.df[self.df['status'] != 'completed']
        
        if failed.empty:
            return {"message": "No failed simulations found"}
        
        analysis = {
            "total_failures": len(failed),
            "failure_rate": len(failed) / len(self.df) * 100,
            "failure_types": failed['status'].value_counts().to_dict(),
            "by_game": failed.groupby('game_id')['status'].count().to_dict()
        }
        
        # Error messages analysis
        if 'error_message' in failed.columns:
            error_messages = failed[failed['error_message'].notna()]['error_message'].value_counts()
            analysis['common_errors'] = error_messages.head(5).to_dict()
        
        return analysis
    
    def termination_analysis(self) -> Dict[str, Any]:
        """Analyze game termination reasons."""
        completed = self.df[self.df['status'].isin(['completed', 'game_ended'])]
        
        if completed.empty:
            return {"error": "No completed/ended simulations found"}
        
        # Count termination reasons
        termination_counts = {}
        for _, row in completed.iterrows():
            reason = row.get('termination_reason') or 'normal_completion'
            termination_counts[reason] = termination_counts.get(reason, 0) + 1
        
        analysis = {
            "termination_reasons": termination_counts,
            "natural_game_endings": termination_counts.get('Game ended', 0),
            "total_analyzed": len(completed),
            "natural_ending_rate": (termination_counts.get('Game ended', 0) / len(completed) * 100) if completed.any() else 0
        }
        
        return analysis
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate comprehensive analysis report."""
        report_lines = []
        
        report_lines.append("🏀 NBA SIMULATION ANALYSIS REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Database: {self.db_path}")
        report_lines.append(f"Total Simulations: {len(self.df)}")
        report_lines.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Game Summary
        report_lines.append("📊 GAME SUMMARY")
        report_lines.append("-" * 20)
        game_summary = self.game_summary()
        for game_id, row in game_summary.iterrows():
            report_lines.append(f"Game {game_id}:")
            report_lines.append(f"  • Total Runs: {row['total_runs']}")
            report_lines.append(f"  • Success Rate: {row['success_rate']}%")
            report_lines.append(f"  • Avg Duration: {row['avg_duration_sec']:.1f}s")
            report_lines.append(f"  • Avg Scoring Rate: {row['avg_scoring_rate']:.1f}%")
        report_lines.append("")
        
        # Scoring Patterns
        report_lines.append("🎯 SCORING PATTERNS")
        report_lines.append("-" * 20)
        scoring = self.scoring_patterns_analysis()
        if "error" not in scoring:
            overall = scoring['overall_scoring_rate']
            report_lines.append(f"Overall Scoring Rate: {overall['mean']:.1f}% ± {overall['std']:.1f}%")
            report_lines.append(f"Range: {overall['min']:.1f}% - {overall['max']:.1f}%")
            report_lines.append(f"High Scoring Games (>25%): {scoring['high_scoring_simulations']}")
            report_lines.append(f"Low Scoring Games (<15%): {scoring['low_scoring_simulations']}")
        report_lines.append("")
        
        # Failure Analysis
        report_lines.append("⚠️ FAILURE ANALYSIS")
        report_lines.append("-" * 20)
        failures = self.failure_analysis()
        if "message" not in failures:
            report_lines.append(f"Total Failures: {failures['total_failures']} ({failures['failure_rate']:.1f}%)")
            report_lines.append("Failure Types:")
            for status, count in failures['failure_types'].items():
                report_lines.append(f"  • {status}: {count}")
        else:
            report_lines.append(failures["message"])
        report_lines.append("")
        
        # Termination Analysis
        report_lines.append("🏁 TERMINATION ANALYSIS")
        report_lines.append("-" * 20)
        terminations = self.termination_analysis()
        if "error" not in terminations:
            report_lines.append(f"Natural Game Endings: {terminations['natural_game_endings']}")
            report_lines.append(f"Natural Ending Rate: {terminations['natural_ending_rate']:.1f}%")
            report_lines.append("All Termination Reasons:")
            for reason, count in terminations['termination_reasons'].items():
                report_lines.append(f"  • {reason}: {count}")
        report_lines.append("")
        
        # Duration Analysis
        report_lines.append("⏱️ DURATION ANALYSIS")
        report_lines.append("-" * 20)
        durations = self.duration_analysis()
        overall_dur = durations['overall_duration']
        report_lines.append(f"Average Duration: {overall_dur['mean_minutes']:.1f} minutes")
        report_lines.append(f"Median Duration: {overall_dur['median_minutes']:.1f} minutes")
        report_lines.append(f"Duration Std: {overall_dur['std_minutes']:.1f} minutes")
        report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def create_visualizations(self, output_dir: str = "analysis_plots"):
        """Create visualization plots for the analysis."""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # 1. Success Rate by Game
        if not self.df.empty:
            game_summary = self.game_summary()
            
            plt.figure(figsize=(10, 6))
            game_summary['success_rate'].plot(kind='bar')
            plt.title('Success Rate by Game')
            plt.ylabel('Success Rate (%)')
            plt.xlabel('Game ID')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/success_rate_by_game.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Scoring Rate Distribution
            completed = self.df[self.df['status'] == 'completed']
            if not completed.empty:
                plt.figure(figsize=(10, 6))
                plt.hist(completed['scoring_rate'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                plt.title('Distribution of Scoring Rates')
                plt.xlabel('Scoring Rate (%)')
                plt.ylabel('Frequency')
                plt.axvline(completed['scoring_rate'].mean(), color='red', linestyle='--', 
                           label=f'Mean: {completed["scoring_rate"].mean():.1f}%')
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"{output_dir}/scoring_rate_distribution.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                # 3. Duration vs Scoring Rate
                plt.figure(figsize=(10, 6))
                plt.scatter(completed['duration_seconds'] / 60, completed['scoring_rate'], alpha=0.6)
                plt.xlabel('Duration (minutes)')
                plt.ylabel('Scoring Rate (%)')
                plt.title('Duration vs Scoring Rate')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/duration_vs_scoring.png", dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"📈 Visualizations saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Analyze NBA simulation results')
    parser.add_argument('--db', type=str, default='simulation_results.db', 
                       help='Path to simulation results database')
    parser.add_argument('--report', type=str, help='Output file for analysis report')
    parser.add_argument('--plots', action='store_true', help='Generate visualization plots')
    parser.add_argument('--plot-dir', type=str, default='analysis_plots', 
                       help='Directory for plots')
    
    args = parser.parse_args()
    
    if not Path(args.db).exists():
        print(f"❌ Database file not found: {args.db}")
        return
    
    analyzer = SimulationAnalyzer(args.db)
    
    # Generate report
    report = analyzer.generate_report(args.report)
    print(report)
    
    if args.report:
        print(f"\n📄 Report saved to: {args.report}")
    
    # Generate plots
    if args.plots:
        try:
            analyzer.create_visualizations(args.plot_dir)
        except ImportError:
            print("⚠️ Matplotlib/Seaborn not available for plotting")


if __name__ == "__main__":
    main()
