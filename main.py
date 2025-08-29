"""
NBA Predictions Training Data Generator - Main Entry Point

This is the main orchestration script for generating NBA training data.
It demonstrates the modular API and serves as the primary interface
for data generation operations.

Usage Examples:
    python main.py --help
    python main.py test
    python main.py generate-game 12345
    python main.py generate-dataset --season 2023-2024 --sample-size 1000
    python main.py generate-openai --game-id 12345
"""

import argparse
import sys
from typing import Optional, List
import pandas as pd

# Import our modular components
from config.settings import config, set_season_year, get_current_season_year
from data.loaders import data_loader, test_data_loading
from data.file_utils import file_manager, data_preview
from training.data_generator import training_data_generator
from training.openai_formatter import openai_dataset_generator
from analysis.player_stats import player_analyzer


class TrainingDataCLI:
    """Command-line interface for NBA training data generation."""
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser with all commands and options."""
        parser = argparse.ArgumentParser(
            description="NBA Predictions Training Data Generator",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s test                                    # Run basic functionality tests
  %(prog)s generate-game 22200001                  # Generate data for specific game
  %(prog)s generate-dataset --sample-size 500     # Generate sample dataset
  %(prog)s generate-openai --game-id 22200001     # Generate OpenAI format data
  %(prog)s list-datasets                           # Show available datasets
  %(prog)s preview dataset.csv                     # Preview training data
            """
        )
        
        # Global options
        parser.add_argument('--season', type=str, default=None,
                          help='Season year (e.g., 2023-2024)')
        parser.add_argument('--verbose', '-v', action='store_true',
                          help='Verbose output')
        
        # Subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Test command
        test_parser = subparsers.add_parser('test', help='Run functionality tests')
        
        # Generate single game
        game_parser = subparsers.add_parser('generate-game', help='Generate data for specific game')
        game_parser.add_argument('game_id', type=int, help='Game ID to process')
        game_parser.add_argument('--n-total', type=int, default=5, 
                               help='Number of recent plays per sequence (default: 5)')
        game_parser.add_argument('--max-plays', type=int, default=None,
                               help='Maximum plays to process (default: all)')
        game_parser.add_argument('--output', type=str, default=None,
                               help='Output filename (default: auto-generated)')
        
        # Generate full dataset
        dataset_parser = subparsers.add_parser('generate-dataset', help='Generate complete dataset')
        dataset_parser.add_argument('--n-total', type=int, default=5,
                                  help='Number of recent plays per sequence (default: 5)')
        dataset_parser.add_argument('--sample-size', type=int, default=None,
                                  help='Random sample size (default: all data)')
        dataset_parser.add_argument('--games', type=str, default=None,
                                  help='Comma-separated list of game IDs to include')
        dataset_parser.add_argument('--output', type=str, default=None,
                                  help='Output filename (default: auto-generated)')
        
        # Generate OpenAI format
        openai_parser = subparsers.add_parser('generate-openai', help='Generate OpenAI training data')
        openai_parser.add_argument('--game-id', type=int, default=None,
                                 help='Specific game ID (for single game)')
        openai_parser.add_argument('--n-total', type=int, default=5,
                                 help='Number of recent plays per sequence (default: 5)')
        openai_parser.add_argument('--sample-size', type=int, default=None,
                                 help='Random sample size (for full dataset)')
        openai_parser.add_argument('--games', type=str, default=None,
                                 help='Comma-separated list of game IDs to include')
        openai_parser.add_argument('--output', type=str, default=None,
                                 help='Output filename (default: auto-generated)')
        
        # List available datasets
        list_parser = subparsers.add_parser('list-datasets', help='List available datasets')
        
        # Preview dataset
        preview_parser = subparsers.add_parser('preview', help='Preview training data')
        preview_parser.add_argument('filename', type=str, help='Dataset filename to preview')
        preview_parser.add_argument('--samples', type=int, default=3,
                                  help='Number of samples to show (default: 3)')
        
        # Configuration management
        config_parser = subparsers.add_parser('config', help='Configuration management')
        config_parser.add_argument('--set-season', type=str, default=None,
                                 help='Set default season year')
        config_parser.add_argument('--show', action='store_true',
                                 help='Show current configuration')
        
        return parser
    
    def run(self, args: Optional[List[str]] = None) -> int:
        """
        Run the CLI with the given arguments.
        
        Args:
            args: Command line arguments (None to use sys.argv)
            
        Returns:
            int: Exit code (0 for success)
        """
        try:
            parsed_args = self.parser.parse_args(args)
            
            # Set global season if provided
            if parsed_args.season:
                set_season_year(parsed_args.season)
            
            # Dispatch to appropriate handler
            if parsed_args.command is None:
                self.parser.print_help()
                return 1
            
            handler_name = f'_handle_{parsed_args.command.replace("-", "_")}'
            handler = getattr(self, handler_name, None)
            
            if handler:
                return handler(parsed_args)
            else:
                print(f"Unknown command: {parsed_args.command}")
                return 1
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return 1
        except Exception as e:
            print(f"Error: {e}")
            if parsed_args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def _handle_test(self, args) -> int:
        """Handle the test command."""
        print("🧪 Running NBA Training Data Generator Tests")
        print("=" * 50)
        
        # Test data loading
        test_data_loading()
        
        # Test single game generation
        print("\n" + "=" * 50)
        print("TESTING SINGLE GAME TRAINING DATA GENERATION")
        print("=" * 50)
        
        try:
            # Load a sample game
            df = data_loader.load_play_by_play_data()
            if len(df) == 0:
                print("❌ No data available for testing")
                return 1
            
            sample_game_id = df['game_id'].iloc[0]
            print(f"Testing with game_id: {sample_game_id}")
            
            # Generate training data (small sample for speed)
            result_df = training_data_generator.generate_for_game(
                sample_game_id, max_plays=10, n_total=3
            )
            
            print(f"✅ Generated {len(result_df)} training records")
            
            # Test OpenAI format generation
            if len(result_df) > 0:
                from training.openai_formatter import openai_formatter
                openai_examples = openai_formatter.create_training_data(result_df)
                print(f"✅ Generated {len(openai_examples)} OpenAI training examples")
            
            print("\n✅ All tests passed!")
            return 0
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return 1
    
    def _handle_generate_game(self, args) -> int:
        """Handle the generate-game command."""
        print(f"Generating training data for game {args.game_id}")
        
        try:
            result_df = training_data_generator.generate_for_game(
                args.game_id, 
                season_year=get_current_season_year(),
                n_total=args.n_total, 
                max_plays=args.max_plays
            )
            
            # Save the dataset
            filename = args.output or f"game_{args.game_id}_training_data.csv"
            filepath = file_manager.save_llm_dataset(result_df, filename)
            
            # Show preview
            if len(result_df) > 0:
                data_preview.preview_llm_data(result_df, n_samples=2)
            
            print(f"\n✅ Training data saved to: {filepath}")
            return 0
            
        except Exception as e:
            print(f"❌ Failed to generate training data: {e}")
            return 1
    
    def _handle_generate_dataset(self, args) -> int:
        """Handle the generate-dataset command."""
        print(f"Generating complete training dataset for season {get_current_season_year()}")
        
        try:
            # Parse game filter if provided
            game_filter = None
            if args.games:
                game_filter = [int(gid.strip()) for gid in args.games.split(',')]
                print(f"Filtering to {len(game_filter)} specific games")
            
            result_df = training_data_generator.generate_dataset(
                season_year=get_current_season_year(),
                n_total=args.n_total,
                sample_size=args.sample_size,
                game_id_filter=game_filter
            )
            
            # Save the dataset
            filename = args.output
            filepath = file_manager.save_llm_dataset(result_df, filename)
            
            # Show dataset statistics
            stats = data_preview.analyze_dataset_stats(result_df)
            print(f"\n📊 Dataset Statistics:")
            print(f"  Total rows: {stats['total_rows']:,}")
            print(f"  Total games: {stats['total_games']:,}")
            print(f"  Memory usage: {stats['memory_usage_mb']} MB")
            if stats['date_range']:
                print(f"  Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
            
            print(f"\n✅ Training dataset saved to: {filepath}")
            return 0
            
        except Exception as e:
            print(f"❌ Failed to generate dataset: {e}")
            return 1
    
    def _handle_generate_openai(self, args) -> int:
        """Handle the generate-openai command."""
        try:
            if args.game_id:
                # Generate for specific game
                print(f"Generating OpenAI training data for game {args.game_id}")
                training_examples, filepath = openai_dataset_generator.generate_for_game(
                    args.game_id, 
                    season_year=get_current_season_year(),
                    n_total=args.n_total
                )
            else:
                # Generate for full dataset
                print(f"Generating OpenAI training dataset for season {get_current_season_year()}")
                
                # Parse game filter if provided
                game_filter = None
                if args.games:
                    game_filter = [int(gid.strip()) for gid in args.games.split(',')]
                
                training_examples, filepath = openai_dataset_generator.generate_dataset(
                    season_year=get_current_season_year(),
                    n_total=args.n_total,
                    sample_size=args.sample_size,
                    game_id_filter=game_filter
                )
            
            # Show preview
            if training_examples:
                data_preview.preview_openai_training_data(training_examples, n_samples=2)
            
            print(f"\n✅ OpenAI training data saved to: {filepath}")
            print(f"Generated {len(training_examples)} training examples")
            return 0
            
        except Exception as e:
            print(f"❌ Failed to generate OpenAI data: {e}")
            return 1
    
    def _handle_list_datasets(self, args) -> int:
        """Handle the list-datasets command."""
        print("📁 Available Training Datasets")
        print("=" * 40)
        
        datasets = file_manager.get_available_datasets()
        
        for format_type, files in datasets.items():
            if files:
                print(f"\n{format_type.upper()} files:")
                for filename in sorted(files):
                    try:
                        info = file_manager.get_file_info(filename)
                        print(f"  📄 {filename} ({info['size_mb']} MB)")
                    except:
                        print(f"  📄 {filename}")
            else:
                print(f"\n{format_type.upper()} files: None")
        
        return 0
    
    def _handle_preview(self, args) -> int:
        """Handle the preview command."""
        try:
            # Determine file type and load appropriately
            filename = args.filename
            
            if filename.endswith('.csv'):
                # Load CSV and preview
                filepath = file_manager.training_dir + "/" + filename
                df = pd.read_csv(filepath)
                
                if 'json_training_data' in df.columns:
                    data_preview.preview_llm_data(df, n_samples=args.samples)
                else:
                    print("CSV file structure:")
                    print(df.info())
                    print("\nFirst few rows:")
                    print(df.head())
            
            elif filename.endswith('.jsonl'):
                # Load JSONL and preview
                import json
                filepath = file_manager.training_dir + "/" + filename
                training_examples = []
                
                with open(filepath, 'r') as f:
                    for line in f:
                        training_examples.append(json.loads(line))
                
                data_preview.preview_openai_training_data(training_examples, n_samples=args.samples)
            
            else:
                print(f"Unsupported file format: {filename}")
                return 1
            
            return 0
            
        except Exception as e:
            print(f"❌ Failed to preview {args.filename}: {e}")
            return 1
    
    def _handle_config(self, args) -> int:
        """Handle the config command."""
        try:
            if args.set_season:
                set_season_year(args.set_season)
                print(f"✅ Season year set to: {args.set_season}")
            
            if args.show or not args.set_season:
                print("📋 Current Configuration:")
                print(f"  Season Year: {get_current_season_year()}")
                print(f"  Data Directory: {config.data_dir}")
                print(f"  Training Directory: {config.training_dir}")
                
                # Show available seasons
                available_seasons = data_loader.get_available_seasons()
                print(f"  Available Seasons: {', '.join(available_seasons)}")
            
            return 0
            
        except Exception as e:
            print(f"❌ Configuration error: {e}")
            return 1


def main() -> int:
    """Main entry point."""
    cli = TrainingDataCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
