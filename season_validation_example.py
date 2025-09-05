#!/usr/bin/env python3
"""
Season Validation Example

Demonstrates how the enhanced orchestration system handles 
game IDs from different seasons automatically.
"""

from season_game_validator import SeasonGameValidator
from enhanced_orchestrator import EnhancedSimulationConfig, Enhanced_NBA_Orchestrator


def example_season_validation():
    """Example showing season validation in action."""
    print("🏀 Season Validation Examples")
    print("=" * 60)
    
    validator = SeasonGameValidator()
    
    # Example 1: Mixed game IDs from different seasons
    print("\n📋 Example 1: Mixed Season Game IDs")
    mixed_games = [
        "22200001",  # 2022-2023 season
        "22200145",  # 2022-2023 season  
        "22300001",  # 2023-2024 season
        "22300078",  # 2023-2024 season
    ]
    
    print(f"Game IDs: {mixed_games}")
    
    # Show what each game ID maps to
    for game_id in mixed_games:
        info = validator.parse_game_id(game_id)
        print(f"  {game_id} → {info.detected_season}")
    
    # Auto-detect seasons
    season_groups = validator.auto_detect_seasons(mixed_games)
    print(f"\n🔍 Auto-detected season groups:")
    for season, games in season_groups.items():
        print(f"  {season}: {games}")
    
    # Example 2: Configuration with wrong season
    print(f"\n📋 Example 2: Mismatched Configuration")
    print(f"Scenario: Configure 2023-2024 season but provide 2022-2023 game IDs")
    
    # This would typically cause issues in the original system
    games_22_23 = ["22200001", "22200002", "22200003"]
    configured_season = "2023-2024"
    
    print(f"Games: {games_22_23}")
    print(f"Configured Season: {configured_season}")
    
    # Validate each game
    print(f"\n🔍 Validation Results:")
    for game_id in games_22_23:
        is_valid, message = validator.validate_game_season_match(game_id, configured_season)
        status = "✅" if is_valid else "❌"
        print(f"  {status} {game_id}: {message}")
    
    # Show what the enhanced system would do
    print(f"\n💡 Enhanced System Behavior:")
    print(f"  1. Detects season mismatch")
    print(f"  2. Auto-detects correct season: 2022-2023")
    print(f"  3. Either warns user or auto-corrects configuration")
    

def example_enhanced_configuration():
    """Example showing enhanced configuration in action."""
    print(f"\n\n🚀 Enhanced Configuration Examples") 
    print("=" * 60)
    
    # Example 1: Auto-detection
    print(f"\n📋 Example 1: Automatic Season Detection")
    games_no_season = ["22200001", "22200002", "22200003"]
    
    print(f"Games: {games_no_season}")
    print(f"Season: Not specified - will auto-detect")
    
    # Create configuration without specifying season
    config = EnhancedSimulationConfig(
        games=games_no_season,
        runs_per_game=2,
        max_iterations_per_run=50,
        output_db="example_autodetect.db"
    )
    
    print(f"Result: Auto-detected season as {config.season_year}")
    
    # Example 2: Validation warnings
    print(f"\n📋 Example 2: Configuration Validation")
    mixed_games = ["22200001", "22300001"]  # Mix of seasons
    
    print(f"Games: {mixed_games} (mixed seasons)")
    print(f"Season: 2023-2024 (explicit)")
    
    try:
        config = EnhancedSimulationConfig(
            season_year="2023-2024",
            games=mixed_games,
            runs_per_game=2,
            output_db="example_validation.db"
        )
        print(f"Configuration created with warnings (see above)")
    except Exception as e:
        print(f"Configuration failed: {e}")


def example_practical_usage():
    """Show practical usage scenarios."""
    print(f"\n\n🎯 Practical Usage Scenarios")
    print("=" * 60)
    
    # Scenario 1: Research across multiple seasons
    print(f"\n📊 Scenario 1: Multi-Season Research")
    print(f"Goal: Compare model performance across seasons")
    
    games_22_23 = ["22200001", "22200002"]
    games_23_24 = ["22300001", "22300002"] 
    
    print(f"2022-2023 games: {games_22_23}")
    print(f"2023-2024 games: {games_23_24}")
    print(f"Approach: Run separate orchestrations for each season")
    
    # Scenario 2: Data validation before large runs
    print(f"\n🔍 Scenario 2: Pre-Run Validation")
    print(f"Goal: Validate 50 game IDs before running 1000 simulations each")
    
    large_game_list = [f"22200{i:03d}" for i in range(1, 51)]  # 50 games from 2022-2023
    
    validator = SeasonGameValidator()
    report = validator.format_validation_report(large_game_list[:5], "2022-2023")  # Show sample
    
    print(f"Sample validation report:")
    print(report)


def example_command_line_usage():
    """Show command line examples."""
    print(f"\n\n💻 Command Line Usage Examples")
    print("=" * 60)
    
    print(f"\n🔍 Validation and Detection:")
    print(f"# Auto-detect seasons for game IDs")
    print(f"python enhanced_orchestrator.py --auto-detect '22200001,22300001,22200002'")
    print(f"")
    print(f"# Validate game IDs against season")
    print(f"python enhanced_orchestrator.py --games '22200001,22200002' --season 2022-2023 --validate-only")
    print(f"")
    print(f"# Show supported seasons")
    print(f"python enhanced_orchestrator.py --show-seasons")
    
    print(f"\n🚀 Enhanced Orchestration:")
    print(f"# Auto-detect season and run (recommended)")
    print(f"python enhanced_orchestrator.py --games '22200001,22200002,22200003' --runs-per-game 10")
    print(f"")
    print(f"# Explicit season with validation")
    print(f"python enhanced_orchestrator.py --games '22300001,22300002' --season 2023-2024 --runs-per-game 5")
    print(f"")
    print(f"# Mixed seasons (system will handle gracefully)")
    print(f"python enhanced_orchestrator.py --games '22200001,22300001' --runs-per-game 3")


def main():
    """Run all examples."""
    try:
        example_season_validation()
        example_enhanced_configuration()
        example_practical_usage() 
        example_command_line_usage()
        
        print(f"\n\n✅ All examples completed!")
        print(f"\n🎯 Key Improvements:")
        print(f"  • Automatic season detection from game IDs")
        print(f"  • Validation warnings for mismatched configurations")
        print(f"  • Graceful handling of mixed-season game lists")
        print(f"  • Clear error messages and suggestions")
        print(f"  • Command-line validation tools")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
