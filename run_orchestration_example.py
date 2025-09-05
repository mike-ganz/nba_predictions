#!/usr/bin/env python3
"""
NBA Orchestration Usage Examples

Demonstrates different ways to use the orchestration system for NBA predictions.
"""

import json
import os
from pathlib import Path
from orchestrator import NBA_Orchestrator, SimulationConfig
from game_context_builder import GameContextBuilder


def example_basic_orchestration():
    """Example 1: Basic orchestration with a few games."""
    print("🚀 Example 1: Basic Orchestration")
    print("=" * 50)
    
    # Configure the orchestration
    config = SimulationConfig(
        season_year="2023-2024",
        games=["22200001", "22200002"],  # 2 games
        runs_per_game=3,                 # 3 simulations each
        max_iterations_per_run=100,      # Shorter runs for testing
        skip_stage1=True,
        platform="gemini",
        output_db="basic_example.db",
        log_level="INFO"
    )
    
    # Run orchestration
    orchestrator = NBA_Orchestrator(config)
    summary = orchestrator.run_all_simulations()
    
    # Show results
    print(f"\n📊 Results Summary:")
    print(f"  Total simulations: {summary['total_simulations']}")
    print(f"  Completed: {summary['completed']}")
    print(f"  Success rate: {summary['success_rate']:.1f}%")
    print(f"  Total time: {summary['total_duration_minutes']:.1f} minutes")
    
    # Generate analysis
    analysis = orchestrator.analyze_results()
    print(f"\n📈 Analysis:")
    print(f"  Average scoring rate: {analysis.get('avg_scoring_rate', 'N/A')}")
    
    return config.output_db


def example_config_file_orchestration():
    """Example 2: Using configuration file."""
    print("\n🚀 Example 2: Configuration File Orchestration")
    print("=" * 50)
    
    # Create a custom config file
    custom_config = {
        "season_year": "2023-2024",
        "games": ["22200001", "22200002", "22200003"],
        "runs_per_game": 5,
        "max_iterations_per_run": 200,
        "skip_stage1": True,
        "platform": "gemini",
        "output_db": "config_example.db",
        "log_level": "INFO",
        "resume_on_error": True,
        "timeout_minutes": 15
    }
    
    # Save config to file
    config_file = "custom_orchestration_config.json"
    with open(config_file, 'w') as f:
        json.dump(custom_config, f, indent=2)
    
    print(f"💾 Saved configuration to: {config_file}")
    
    # Load and run
    from orchestrator import load_config_from_file
    config = load_config_from_file(config_file)
    orchestrator = NBA_Orchestrator(config)
    
    # Run just one game for demonstration
    config.games = config.games[:1]  # Just first game
    summary = orchestrator.run_all_simulations()
    
    print(f"📊 Configuration-based run completed: {summary['completed']} simulations")
    
    # Clean up
    Path(config_file).unlink(missing_ok=True)
    
    return config.output_db


def example_game_context_building():
    """Example 3: Building custom game contexts."""
    print("\n🚀 Example 3: Custom Game Context Building")
    print("=" * 50)
    
    # Build custom game contexts
    builder = GameContextBuilder("2023-2024")
    
    # Get available games
    available_games = builder.get_available_games()
    print(f"📋 Found {len(available_games)} available games")
    
    # Build contexts for first 3 games
    game_ids = available_games[:3]
    contexts = builder.batch_build_contexts(game_ids, "example_contexts")
    
    # Use the built contexts in orchestration
    config = SimulationConfig(
        season_year="2023-2024",
        games=game_ids,
        runs_per_game=2,
        max_iterations_per_run=50,
        output_db="context_example.db"
    )
    
    orchestrator = NBA_Orchestrator(config)
    
    # Modify the context loader to use our built contexts
    context_dir = Path("example_contexts")
    
    def custom_context_loader(game_id, season_year):
        context_file = context_dir / f"game_{game_id}_context.json"
        if context_file.exists():
            with open(context_file) as f:
                return json.load(f)
        else:
            return orchestrator.context_loader.load_game_context(game_id, season_year)
    
    # Replace the loader
    orchestrator.context_loader.load_game_context = custom_context_loader
    
    # Run orchestration
    summary = orchestrator.run_all_simulations()
    print(f"📊 Custom context run: {summary['completed']} completed simulations")
    
    return config.output_db


def example_analysis_and_reporting():
    """Example 4: Analysis and reporting on results."""
    print("\n🚀 Example 4: Analysis and Reporting")
    print("=" * 50)
    
    # First, run a larger orchestration to get meaningful data
    config = SimulationConfig(
        season_year="2023-2024",
        games=["22200001", "22200002"],
        runs_per_game=5,
        max_iterations_per_run=150,
        output_db="analysis_example.db"
    )
    
    orchestrator = NBA_Orchestrator(config)
    summary = orchestrator.run_all_simulations()
    
    print(f"📊 Generated {summary['total_simulations']} simulations for analysis")
    
    # Now analyze the results
    from results_analyzer import SimulationAnalyzer
    
    analyzer = SimulationAnalyzer(config.output_db)
    
    # Generate comprehensive report
    report = analyzer.generate_report("analysis_report.txt")
    print(f"\n📄 Analysis Report:")
    print("=" * 30)
    print(report[:500])  # Show first 500 characters
    print("... (truncated)")
    
    # Game summary
    game_summary = analyzer.game_summary()
    print(f"\n📈 Game Summary:")
    print(game_summary)
    
    # Scoring patterns
    scoring_patterns = analyzer.scoring_patterns_analysis()
    if "error" not in scoring_patterns:
        overall = scoring_patterns['overall_scoring_rate']
        print(f"\n🎯 Scoring Analysis:")
        print(f"  Average scoring rate: {overall['mean']:.1f}% ± {overall['std']:.1f}%")
        print(f"  Range: {overall['min']:.1f}% - {overall['max']:.1f}%")
    
    return config.output_db


def example_error_handling_and_recovery():
    """Example 5: Error handling and recovery scenarios."""
    print("\n🚀 Example 5: Error Handling and Recovery")
    print("=" * 50)
    
    # Configure with potentially problematic settings
    config = SimulationConfig(
        season_year="2023-2024",
        games=["22200001", "INVALID_GAME", "22200002"],  # Include invalid game
        runs_per_game=2,
        max_iterations_per_run=50,
        output_db="error_handling_example.db",
        resume_on_error=True,
        timeout_minutes=5
    )
    
    orchestrator = NBA_Orchestrator(config)
    
    # Run with error handling
    summary = orchestrator.run_all_simulations()
    
    print(f"📊 Error handling run results:")
    print(f"  Total attempted: {summary['total_simulations']}")
    print(f"  Completed: {summary['completed']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Success rate: {summary['success_rate']:.1f}%")
    
    # Analyze failures
    from results_analyzer import SimulationAnalyzer
    analyzer = SimulationAnalyzer(config.output_db)
    failure_analysis = analyzer.failure_analysis()
    
    if "message" not in failure_analysis:
        print(f"\n⚠️ Failure Analysis:")
        print(f"  Total failures: {failure_analysis['total_failures']}")
        print(f"  Failure rate: {failure_analysis['failure_rate']:.1f}%")
        print(f"  Failure types: {failure_analysis['failure_types']}")
    
    return config.output_db


def cleanup_example_files():
    """Clean up example files."""
    print("\n🧹 Cleaning up example files...")
    
    example_files = [
        "basic_example.db",
        "config_example.db", 
        "context_example.db",
        "analysis_example.db",
        "error_handling_example.db",
        "analysis_report.txt",
        "custom_orchestration_config.json"
    ]
    
    for file in example_files:
        Path(file).unlink(missing_ok=True)
    
    # Remove example directories
    import shutil
    for dir_name in ["example_contexts", "analysis_plots"]:
        if Path(dir_name).exists():
            shutil.rmtree(dir_name)
    
    print("✅ Cleanup completed")


def main():
    """Run all examples."""
    print("🏀 NBA Orchestration System Examples")
    print("=" * 60)
    
    try:
        # Run examples
        db1 = example_basic_orchestration()
        db2 = example_config_file_orchestration()
        db3 = example_game_context_building()
        db4 = example_analysis_and_reporting()
        db5 = example_error_handling_and_recovery()
        
        print("\n🎉 All examples completed successfully!")
        print("\nGenerated databases:")
        for db in [db1, db2, db3, db4, db5]:
            if Path(db).exists():
                size = Path(db).stat().st_size / 1024  # KB
                print(f"  • {db} ({size:.1f} KB)")
        
        # Ask if user wants to keep files
        keep_files = input("\nKeep example files? (y/n): ").lower().startswith('y')
        if not keep_files:
            cleanup_example_files()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Example failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
