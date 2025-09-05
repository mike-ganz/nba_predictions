#!/usr/bin/env python3
"""
Enhanced NBA Prediction Orchestration System

Improved version that properly handles season/game ID relationships
and provides better validation and auto-detection capabilities.
"""

import json
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import the original orchestrator components
from orchestrator import (
    SimulationConfig, SimulationResult, SimulationDatabase, 
    NBA_Orchestrator as BaseOrchestrator, load_config_from_file
)

# Import the new validator
from season_game_validator import SeasonGameValidator


class EnhancedSimulationConfig(SimulationConfig):
    """Enhanced configuration with season validation."""
    
    def __init__(self, **kwargs):
        # Auto-detect seasons if not specified
        if 'games' in kwargs and 'season_year' not in kwargs:
            validator = SeasonGameValidator()
            suggested_season = validator.suggest_season_for_games(kwargs['games'])
            if suggested_season:
                kwargs['season_year'] = suggested_season
                print(f"🔍 Auto-detected season: {suggested_season}")
        
        # Set Stage 1 enabled by default (matching first_n_plays training mode)
        if 'skip_stage1' not in kwargs:
            kwargs['skip_stage1'] = False
        
        super().__init__(**kwargs)
        
        # Validate configuration after initialization
        self._validate_config()
    
    def _validate_config(self):
        """Validate that games match the configured season."""
        validator = SeasonGameValidator()
        
        # Check each game
        invalid_games = []
        season_mismatches = []
        
        for game_id in self.games:
            is_valid, message = validator.validate_game_season_match(game_id, self.season_year)
            if not is_valid:
                game_info = validator.parse_game_id(game_id)
                if game_info.is_valid and game_info.detected_season != self.season_year:
                    season_mismatches.append((game_id, game_info.detected_season))
                else:
                    invalid_games.append((game_id, message))
        
        if invalid_games or season_mismatches:
            print(f"\n⚠️ Configuration Validation Issues:")
            print(f"   Configured Season: {self.season_year}")
            
            if invalid_games:
                print(f"   Invalid Game IDs:")
                for game_id, error in invalid_games:
                    print(f"     • {game_id}: {error}")
            
            if season_mismatches:
                print(f"   Season Mismatches:")
                for game_id, detected_season in season_mismatches:
                    print(f"     • {game_id} belongs to {detected_season}")
                
                # Group mismatches by season
                season_groups = {}
                for game_id, detected_season in season_mismatches:
                    if detected_season not in season_groups:
                        season_groups[detected_season] = []
                    season_groups[detected_season].append(game_id)
                
                print(f"\n💡 Suggestion: Consider running separate orchestrations:")
                print(f"   Current season ({self.season_year}): {[g for g in self.games if g not in [gid for gid, _ in season_mismatches]]}")
                for season, games in season_groups.items():
                    print(f"   {season}: {games}")
    
    def get_validation_report(self) -> str:
        """Get detailed validation report."""
        validator = SeasonGameValidator()
        return validator.format_validation_report(self.games, self.season_year)


class Enhanced_NBA_Orchestrator(BaseOrchestrator):
    """Enhanced orchestrator with better season handling."""
    
    def __init__(self, config: EnhancedSimulationConfig):
        # Validate and potentially fix configuration
        self.validator = SeasonGameValidator()
        self.config = config
        
        # Initialize with validated configuration
        super().__init__(config)
    
    def run_all_simulations(self) -> Dict[str, Any]:
        """Run simulations with enhanced season handling."""
        
        # Check if we have games from multiple seasons
        season_groups = self.validator.auto_detect_seasons(self.config.games)
        
        if len(season_groups) > 1:
            print(f"\n🔄 Multiple seasons detected: {list(season_groups.keys())}")
            print(f"   Current configuration targets: {self.config.season_year}")
            
            # Filter to only games that match the configured season
            valid_games = season_groups.get(self.config.season_year, [])
            
            if valid_games:
                print(f"   Running simulations for {len(valid_games)} games in {self.config.season_year}")
                self.config.games = valid_games
            else:
                print(f"   ❌ No games found for configured season {self.config.season_year}")
                print(f"   Available seasons: {list(season_groups.keys())}")
                
                # Ask user what to do
                print(f"\n🤔 What would you like to do?")
                print(f"   1. Run with auto-detected season")
                print(f"   2. Cancel and reconfigure")
                
                # For automated runs, use the most common season
                most_common_season = max(season_groups.keys(), key=lambda k: len(season_groups[k]))
                print(f"\n⚡ Auto-selecting most common season: {most_common_season}")
                
                self.config.season_year = most_common_season
                self.config.games = season_groups[most_common_season]
                
                # Reinitialize with correct season
                from config.settings import set_season_year
                set_season_year(most_common_season)
        
        # Run the original simulation logic
        return super().run_all_simulations()


def create_enhanced_config_from_args(args) -> EnhancedSimulationConfig:
    """Create enhanced configuration from command line arguments."""
    
    if args.config:
        # Load base config and enhance it
        base_data = {}
        with open(args.config, 'r') as f:
            base_data = json.load(f)
        return EnhancedSimulationConfig(**base_data)
    else:
        if not args.games:
            raise ValueError("Either --config or --games must be specified")
        
        games = [g.strip() for g in args.games.split(',')]
        
        config_data = {
            'games': games,
            'runs_per_game': args.runs_per_game,
            'max_iterations_per_run': args.max_iterations,
            'platform': args.platform,
            'output_db': args.output_db,
            'log_level': args.log_level
        }
        
        # Only set season if explicitly provided, otherwise let auto-detection work
        if args.season:
            config_data['season_year'] = args.season
        
        return EnhancedSimulationConfig(**config_data)


def main():
    parser = argparse.ArgumentParser(description='Enhanced NBA Prediction Orchestration System')
    
    # Configuration options
    parser.add_argument('--config', type=str, help='Path to JSON configuration file')
    parser.add_argument('--games', type=str, help='Comma-separated list of game IDs')
    parser.add_argument('--runs-per-game', type=int, default=5, help='Number of runs per game')
    parser.add_argument('--season', type=str, help='Season year (YYYY-YYYY) - auto-detected if not provided')
    parser.add_argument('--platform', type=str, default='gemini', choices=['openai', 'gemini'], help='AI platform')
    parser.add_argument('--max-iterations', type=int, default=2000, help='Max iterations per run')
    parser.add_argument('--output-db', type=str, default='enhanced_simulation_results.db', help='Output database file')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    # Validation options
    parser.add_argument('--validate-only', action='store_true', help='Only validate game IDs, don\'t run simulations')
    parser.add_argument('--show-seasons', action='store_true', help='Show supported seasons')
    parser.add_argument('--auto-detect', type=str, help='Auto-detect seasons for comma-separated game IDs')
    
    # Action options  
    parser.add_argument('--analyze', action='store_true', help='Analyze existing results')
    parser.add_argument('--analyze-game', type=str, help='Analyze results for specific game ID')
    
    args = parser.parse_args()
    
    validator = SeasonGameValidator()
    
    if args.show_seasons:
        seasons = validator.get_supported_seasons()
        print("🏀 Supported NBA Seasons:")
        for season in seasons:
            print(f"  • {season}")
        return
    
    if args.auto_detect:
        games = [g.strip() for g in args.auto_detect.split(',')]
        print(f"🔍 Auto-detecting seasons for {len(games)} games...")
        
        season_groups = validator.auto_detect_seasons(games)
        
        print(f"\n📊 Results:")
        for season, game_list in season_groups.items():
            print(f"  {season}: {len(game_list)} games")
            for game_id in game_list[:5]:  # Show first 5
                print(f"    • {game_id}")
            if len(game_list) > 5:
                print(f"    ... and {len(game_list) - 5} more")
        
        suggested = validator.suggest_season_for_games(games)
        if suggested:
            print(f"\n💡 Suggested season for mixed batch: {suggested}")
        
        return
    
    # Create enhanced configuration
    try:
        config = create_enhanced_config_from_args(args)
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return
    
    if args.validate_only:
        print("🔍 Validation Report:")
        print(config.get_validation_report())
        return
    
    # Create enhanced orchestrator
    orchestrator = Enhanced_NBA_Orchestrator(config)
    
    if args.analyze:
        # Analyze all results
        analysis = orchestrator.analyze_results()
        print(f"\n📊 Analysis Results - {analysis.get('title', 'All Games')}")
        print(f"Total simulations: {analysis.get('total_simulations', 0)}")
        print(f"Completed: {analysis.get('completed_simulations', 0)} ({analysis.get('success_rate', 0):.1f}%)")
        if 'avg_scoring_rate' in analysis:
            print(f"Average scoring rate: {analysis['avg_scoring_rate']:.1f}%")
    
    elif args.analyze_game:
        # Analyze specific game
        analysis = orchestrator.analyze_results(args.analyze_game)
        if 'error' not in analysis:
            print(f"\n📊 Analysis Results - {analysis.get('title', args.analyze_game)}")
            print(f"Total simulations: {analysis.get('total_simulations', 0)}")
            print(f"Completed: {analysis.get('completed_simulations', 0)} ({analysis.get('success_rate', 0):.1f}%)")
    
    else:
        # Run simulations
        print("\n🏀 Enhanced NBA Prediction Orchestration")
        print("=" * 60)
        print(f"Configuration validation: {'✅ Passed' if len(config.games) > 0 else '❌ Failed'}")
        print(f"Season: {config.season_year}")
        print(f"Games: {len(config.games)} games")
        print(f"Total simulations: {len(config.games) * config.runs_per_game}")
        
        summary = orchestrator.run_all_simulations()
        
        # Show final analysis
        print(f"\n🎯 Final Results:")
        print(f"  Success rate: {summary.get('success_rate', 0):.1f}%")
        print(f"  Total duration: {summary.get('total_duration_minutes', 0):.1f} minutes")
        print(f"  Database: {config.output_db}")


if __name__ == "__main__":
    main()
