"""
Core training data generation module for NBA predictions.

This module contains the main business logic for generating LLM training data
from NBA play-by-play data, including team stats integration, player analysis,
and JSON formatting.
"""

import json
import pandas as pd
import re
from typing import Dict, List, Optional, Any
from config.settings import config, DEFAULT_N_TOTAL_PLAYS, DEFAULT_MIN_GAMES_THRESHOLD
from data.loaders import data_loader
from game.time_utils import convert_to_quarter_time
from game.team_utils import team_manager, get_team_stats_for_game, determine_home_away_teams
from game.scoring_utils import determine_scoring_info
from analysis.player_stats import player_analyzer, lineup_manager


def remove_parentheses_content(text):
    """
    Remove content within parentheses (including the parentheses) from text.
    
    Args:
        text (str): Input text that may contain parentheses
        
    Returns:
        str: Text with parentheses content removed and extra spaces cleaned up
    
    Example:
        "Lebron James 3-pt make (17 pts)" -> "Lebron James 3-pt make"
    """
    if not text or pd.isna(text):
        return text
    
    # Remove content within parentheses using regex
    # \([^)]*\) matches opening paren, any chars except closing paren, closing paren
    cleaned_text = re.sub(r'\([^)]*\)', '', str(text))
    
    # Clean up extra whitespace that may result from removal
    cleaned_text = ' '.join(cleaned_text.split())
    
    return cleaned_text


class TrainingDataGenerator:
    """Main class for generating NBA training data."""
    
    def __init__(self):
        self._team_stats_cache = {}  # Cache team stats for performance
        self._lineup_cache = {}      # Cache lineup data for performance
    
    def generate_for_game(self, game_id: int, season_year: Optional[str] = None,
                         n_total: int = DEFAULT_N_TOTAL_PLAYS, 
                         max_plays: Optional[int] = None) -> pd.DataFrame:
        """
        Generate LLM training data for a specific game_id.
        
        Args:
            game_id: The specific game ID to generate data for
            season_year: Season year (e.g., "2023-2024")
            n_total: Total number of recent plays to include in each sequence
            max_plays: Maximum number of plays from the game to process (None for all)
        
        Returns:
            pd.DataFrame: DataFrame with LLM training data for the specified game
        """
        print(f"Generating training data for game_id: {game_id}")
        
        # Load the full dataset
        df = data_loader.load_play_by_play_data(season_year)
        
        # Filter to the specific game
        game_df = df[df['game_id'] == game_id]
        
        if len(game_df) == 0:
            raise ValueError(f"Game ID {game_id} not found in {season_year} season data")
        
        # Get game info
        game_date = game_df.iloc[0]['date']
        total_plays = len(game_df)
        
        print(f"Found game on {game_date} with {total_plays} total plays")
        
        # Limit plays if requested
        if max_plays and max_plays < total_plays:
            game_df = game_df.head(max_plays)
            print(f"Limited to first {max_plays} plays")
        
        # Sort by play order
        game_df = game_df.sort_values(['game_id', 'play_id']).reset_index(drop=True)
        
        # Generate training data
        print(f"Generating training sequences with n_total={n_total}...")
        result_df = self.create_llm_training_data(game_df, n_total=n_total, filter_nan=True)
        
        print(f"✅ Generated {len(result_df)} training records for game {game_id}")
        
        return result_df
    
    def generate_dataset(self, season_year: Optional[str] = None, 
                        n_total: int = DEFAULT_N_TOTAL_PLAYS, 
                        sample_size: Optional[int] = None, 
                        game_id_filter: Optional[List[int]] = None,
                        force_real_pca: bool = False) -> pd.DataFrame:
        """
        Generate a complete LLM training dataset from play-by-play data.
        
        Args:
            season_year: Season year to use. If None, uses current config
            n_total: Total number of descriptions to concatenate together per row
            sample_size: If provided, randomly sample this many rows
            game_id_filter: If provided, only include these game IDs
            force_real_pca: If True, always use real PCA calculations (for cache building)
        
        Returns:
            pd.DataFrame: DataFrame ready for LLM training with concatenated descriptions
        """
        if season_year is None:
            season_year = config.season_year
        
        print(f"Generating LLM dataset for season {season_year}...")
        
        # Load the play-by-play data
        df = data_loader.load_play_by_play_data(season_year)
        
        # Filter by game IDs if specified
        if game_id_filter:
            df = df[df['game_id'].isin(game_id_filter)]
            print(f"Filtered to {len(df)} rows from {len(game_id_filter)} games")
        
        # Sort by game_id and play sequence to ensure proper chronological order
        df = df.sort_values(['game_id', 'play_id']).reset_index(drop=True)
        
        # Create concatenated descriptions
        result_df = self.create_llm_training_data(df, n_total=n_total, force_real_pca=force_real_pca)
        
        # Sample if requested
        if sample_size and sample_size < len(result_df):
            result_df = result_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            print(f"Sampled {sample_size} rows from dataset")
        
        print(f"Generated dataset with {len(result_df)} rows")
        print(f"Each row contains up to {n_total} descriptions concatenated together")
        
        return result_df
    
    def create_llm_training_data(self, df: pd.DataFrame, n_total: int = DEFAULT_N_TOTAL_PLAYS,
                               filter_nan: bool = True, force_real_pca: bool = False) -> pd.DataFrame:
        """
        Create LLM training data in structured JSON format with team stats and recent plays.
        Only includes plays within the same game (respects game_id boundaries).
        
        Args:
            df: Play-by-play DataFrame with required columns
            n_total: Total number of recent plays to include (default: 5)
            filter_nan: Whether to filter out rows with NaN descriptions (default: True)
            force_real_pca: If True, always use real PCA calculations (for cache building)
        
        Returns:
            pd.DataFrame: DataFrame with new 'json_training_data' column containing JSON strings
        """
        # Work with a copy to avoid modifying original DataFrame
        result_df = df.copy()
        
        # Filter out NaN descriptions if requested
        if filter_nan:
            result_df = result_df.dropna(subset=['description']).reset_index(drop=True)
        
        # Determine home/away team mapping for all games
        print("Determining home/away team mappings...")
        game_team_mapping = determine_home_away_teams(result_df)
        
        # Load and cache team stats and lineups
        print("Loading team stats and lineups...")
        self._load_game_context(result_df, game_team_mapping, force_real_pca=force_real_pca)
        
        json_training_data = []
        
        for i in range(len(result_df)):
            current_game_id = result_df.iloc[i]['game_id']
            current_desc = result_df.iloc[i]['description']
            
            # Get cached team stats and lineups for current game
            game_context = self._team_stats_cache.get(current_game_id, {})
            
            # Create the JSON training data for this row
            json_obj = self._create_training_json(
                result_df, i, game_context, game_team_mapping, n_total, force_real_pca=force_real_pca
            )
            
            # Convert to JSON string
            json_string = json.dumps(json_obj, separators=(',', ':'))
            json_training_data.append(json_string)
        
        # Add the JSON training data as a new column
        result_df['json_training_data'] = json_training_data
        
        return result_df
    
    def _load_game_context(self, df: pd.DataFrame, game_team_mapping: Dict[int, Dict[str, str]], 
                          force_real_pca: bool = False) -> None:
        """
        Load and cache team stats and lineups for all games in the dataset.
        
        Args:
            df: Play-by-play DataFrame
            game_team_mapping: Mapping of game IDs to home/away teams
        """
        unique_games = df['game_id'].unique()
        print(f"Loading team stats for {len(unique_games)} unique games...")
        
        # Find the latest game date for date filtering optimization
        max_game_date = None
        if 'date' in df.columns:
            try:
                # Handle multiple date formats commonly found in NBA datasets (M/D/YYYY, YYYY-MM-DD, etc.)
                max_game_date = pd.to_datetime(df['date'], format='mixed', dayfirst=False).max().strftime('%Y-%m-%d')
                print(f"📅 Using max game date for filtering: {max_game_date}")
            except Exception as e:
                try:
                    # Fallback: Try inferring the format
                    max_game_date = pd.to_datetime(df['date'], infer_datetime_format=True).max().strftime('%Y-%m-%d')
                    print(f"📅 Using max game date for filtering (inferred format): {max_game_date}")
                except Exception as e2:
                    print(f"Warning: Could not determine max game date: {e}")
                    print(f"Fallback also failed: {e2}")
                    print("Continuing without date filtering optimization...")
        
        # Load player boxscore data for lineups (with date filtering optimization)
        boxscore_data = self._load_boxscore_data(unique_games, max_game_date)
        
        for game_id in unique_games:
            game_df = df[df['game_id'] == game_id]
            game_date = game_df.iloc[0].get('date', None)
            
            # Get team stats
            stats = get_team_stats_for_game(
                game_df, game_team_mapping, 
                target_date=game_date, 
                min_games_threshold=DEFAULT_MIN_GAMES_THRESHOLD
            )
            
            # Get lineups
            lineups = {}
            if boxscore_data is not None:
                try:
                    lineups = lineup_manager.get_lineup_by_game_id(
                        game_id, 
                        boxscore_data,
                        max_date=game_date,  # Pass game date for additional filtering
                        current_season=config.season_year
                    )
                except Exception as e:
                    print(f"Warning: Could not get lineups for game {game_id}: {e}")
            
            # Cache the complete context
            self._team_stats_cache[game_id] = {
                **stats,
                'lineups': lineups,
                'game_date': game_date
            }
    
    def _load_boxscore_data(self, unique_games: List[int], max_game_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load player boxscore data with date filtering optimization.
        
        Args:
            unique_games: List of unique game IDs
            max_game_date: Latest game date to load data for (YYYY-MM-DD format)
            
        Returns:
            pd.DataFrame or None: Player boxscore data if available
        """
        # Check if we already have boxscore data cached (MAJOR PERFORMANCE IMPROVEMENT)
        cache_key = f"boxscore_{max_game_date}_{config.season_year}"
        if hasattr(self, '_boxscore_cache') and cache_key in self._boxscore_cache:
            cached_data = self._boxscore_cache[cache_key]
            print(f"🚀 Using cached boxscore data ({len(cached_data):,} records) - HUGE speedup!")
            return cached_data
        
        print("📊 Loading player boxscore data for lineups (first time only)...")
        try:
            # Use date filtering to only load relevant data
            boxscore_data = data_loader.load_all_player_boxscores(
                max_date=max_game_date,
                current_season=config.season_year
            )
            if boxscore_data is None:
                return None
                
            print(f"✅ Loaded {len(boxscore_data):,} player records (date-filtered)")
            
            # ADDITIONAL OPTIMIZATION: Filter to only games we need if testing with small dataset
            if len(unique_games) <= 5:  # Testing mode - filter data
                original_size = len(boxscore_data)
                game_id_column = 'GAME-ID' if 'GAME-ID' in boxscore_data.columns else 'game_id'
                boxscore_data = boxscore_data[boxscore_data[game_id_column].isin(unique_games)]
                filtered_size = len(boxscore_data)
                print(f"🚀 FURTHER OPTIMIZED: Filtered from {original_size} to {filtered_size} records for target games")
            
            # Cache the loaded data to avoid repeated loading (HUGE PERFORMANCE GAIN)
            if not hasattr(self, '_boxscore_cache'):
                self._boxscore_cache = {}
            self._boxscore_cache[cache_key] = boxscore_data
            print(f"💾 Cached boxscore data for future batches - subsequent batches will be lightning fast!")
            
            return boxscore_data
            
        except Exception as e:
            print(f"Warning: Could not load boxscore data for lineups: {e}")
            return None
    
    def _create_training_json(self, df: pd.DataFrame, row_index: int, 
                            game_context: Dict[str, Any], 
                            game_team_mapping: Dict[int, Dict[str, str]],
                            n_total: int, force_real_pca: bool = False) -> Dict[str, Any]:
        """
        Create the JSON training data object for a single row.
        
        Args:
            df: Full DataFrame
            row_index: Index of current row
            game_context: Cached team stats and lineup data for the game
            game_team_mapping: Mapping of game IDs to home/away teams
            n_total: Number of recent plays to include
            force_real_pca: If True, always use real PCA calculations (for cache building)
            
        Returns:
            dict: Complete JSON training data object
        """
        current_game_id = df.iloc[row_index]['game_id']
        
        # Extract team information
        away_stats = game_context.get('away_team_stats', {})
        home_stats = game_context.get('home_team_stats', {})
        away_abbrev = game_context.get('away_abbrev', 'Unknown')
        home_abbrev = game_context.get('home_abbrev', 'Unknown')
        lineups = game_context.get('lineups', {})
        game_date = game_context.get('game_date')
        
        # Collect recent plays from the same game only
        recent_plays = self._collect_recent_plays(df, row_index, n_total, away_abbrev, home_abbrev)
        
        # Create players array from lineups
        players = self._create_player_objects(
            lineups, away_abbrev, home_abbrev, game_date, force_real_pca=force_real_pca
        )
        
        # Handle None REST_DAYS by converting to 0
        away_rest_days = away_stats.get('REST_DAYS') if away_stats.get('REST_DAYS') is not None else 0
        home_rest_days = home_stats.get('REST_DAYS') if home_stats.get('REST_DAYS') is not None else 0
        
        # Create JSON structure with proper type conversion
        json_obj = {
            "away_team": {
                "name": str(away_abbrev) if away_abbrev else "Unknown",
                "stats": {
                    "OEFF": float(away_stats.get('OEFF')) if away_stats.get('OEFF') is not None else None,
                    "DEFF": float(away_stats.get('DEFF')) if away_stats.get('DEFF') is not None else None,
                    "PACE": float(away_stats.get('PACE')) if away_stats.get('PACE') is not None else None,
                    "REST_DAYS": int(away_rest_days) if away_rest_days is not None else 0
                }
            },
            "home_team": {
                "name": str(home_abbrev) if home_abbrev else "Unknown",
                "stats": {
                    "OEFF": float(home_stats.get('OEFF')) if home_stats.get('OEFF') is not None else None,
                    "DEFF": float(home_stats.get('DEFF')) if home_stats.get('DEFF') is not None else None,
                    "PACE": float(home_stats.get('PACE')) if home_stats.get('PACE') is not None else None,
                    "REST_DAYS": int(home_rest_days) if home_rest_days is not None else 0
                }
            },
            "players": players,
            "recent_plays": recent_plays
        }
        
        return json_obj
    
    def _collect_recent_plays(self, df: pd.DataFrame, current_index: int, n_total: int,
                            away_abbrev: str, home_abbrev: str) -> List[Dict[str, Any]]:
        """
        Collect recent plays from the same game for context.
        
        Args:
            df: Full DataFrame
            current_index: Index of current row
            n_total: Number of recent plays to collect
            away_abbrev: Away team abbreviation
            home_abbrev: Home team abbreviation
            
        Returns:
            list: Recent plays data
        """
        current_game_id = df.iloc[current_index]['game_id']
        recent_plays = []
        collected_count = 0
        
        # Go backwards from current position to collect recent plays
        for j in range(current_index, -1, -1):  # Start from current row and go backwards
            row_game_id = df.iloc[j]['game_id']
            row_desc = df.iloc[j]['description']
            row_away_score = df.iloc[j].get('away_score', 0) or 0
            row_home_score = df.iloc[j].get('home_score', 0) or 0
            
            # Stop if we've moved to a different game
            if row_game_id != current_game_id:
                break
            
            # Add valid descriptions as recent plays
            if pd.notna(row_desc):
                # Get quarter and time remaining in quarter
                row_period = df.iloc[j].get('period', 4)
                row_remaining_time = df.iloc[j].get('remaining_time', '0:00:00')
                quarter, time_in_quarter = convert_to_quarter_time(row_period, row_remaining_time)
                
                # Determine scoring info by comparing with previous play
                scoring_team = None
                points_scored = 0
                
                # To determine scoring, we need to compare this play with the chronologically previous play
                # Since we're iterating backwards (j decreasing), the previous play chronologically is at j-1
                if j > 0:  # Make sure we have a previous play
                    prev_j = j - 1
                    if (prev_j >= 0 and df.iloc[prev_j]['game_id'] == current_game_id):
                        prev_away_score = df.iloc[prev_j].get('away_score', 0) or 0
                        prev_home_score = df.iloc[prev_j].get('home_score', 0) or 0
                        
                        scoring_team, points_scored = determine_scoring_info(
                            prev_away_score, prev_home_score, 
                            row_away_score, row_home_score, 
                            away_abbrev, home_abbrev
                        )
                
                # Create play object with proper type conversion
                play_obj = {
                    "quarter": int(quarter),
                    "time_remaining": str(time_in_quarter),
                    "description": remove_parentheses_content(row_desc),
                    "score": f"{away_abbrev} {int(row_away_score)} - {home_abbrev} {int(row_home_score)}",
                    "scoring_team": str(scoring_team) if scoring_team else None,
                    "points_scored": int(points_scored)
                }
                
                recent_plays.insert(0, play_obj)  # Insert at beginning to maintain chronological order
                collected_count += 1
                
                # Stop if we've collected the desired total number of plays
                if collected_count >= n_total:
                    break
        
        return recent_plays
    
    def _create_player_objects(self, lineups: Dict[str, List[str]], away_abbrev: str, 
                             home_abbrev: str, game_date: Optional[str],
                             force_real_pca: bool = False) -> List[Dict[str, Any]]:
        """
        Create player objects with PCA stats from lineup data.
        
        Args:
            lineups: Dictionary mapping team names to player lists
            away_abbrev: Away team abbreviation
            home_abbrev: Home team abbreviation
            game_date: Date of the game
            force_real_pca: If True, always use real PCA calculations (for cache building)
            
        Returns:
            list: List of player objects with stats
        """
        # Get abbreviation to full name mapping for lineup matching
        away_full_name = team_manager.get_team_full_name(away_abbrev)
        home_full_name = team_manager.get_team_full_name(home_abbrev)
        
        # Extract season from config
        season = None
        if config.season_year and '-' in config.season_year:
            season = config.season_year.split('-')[1]  # "2023-2024" -> "2024"
        
        # Always use real PCA since user has pre-built cache
        # The old fast mode detection was incorrectly triggering dummy values
        if force_real_pca:
            use_fast_mode = False  # Cache building mode
            print(f"🔥 CACHE BUILDING MODE: Using REAL PCA values for {sum(len(players) for players in lineups.values()) if lineups else 0} players")
        else:
            use_fast_mode = False  # Always use real cached PCA values
            # Removed buggy player count detection that was causing dummy values
        
        return lineup_manager.process_lineups_for_training_data(
            lineups, away_abbrev, home_abbrev, away_full_name, home_full_name, 
            game_date, season, fast_mode=use_fast_mode
        )
    
    def clear_caches(self) -> Dict[str, int]:
        """
        Clear all internal caches.
        
        Returns:
            dict: Cache statistics before clearing
        """
        stats = {
            'team_stats_entries': len(self._team_stats_cache),
            'lineup_entries': len(self._lineup_cache),
            'boxscore_entries': len(getattr(self, '_boxscore_cache', {}))
        }
        
        self._team_stats_cache.clear()
        self._lineup_cache.clear()
        if hasattr(self, '_boxscore_cache'):
            self._boxscore_cache.clear()
        
        # Also clear player analyzer cache
        player_analyzer.clear_pca_cache()
        
        print(f"Cleared caches: {stats}")
        return stats


# Global instance
training_data_generator = TrainingDataGenerator()

# Convenience functions for backward compatibility
def create_llm_training_data(df: pd.DataFrame, n_total: int = DEFAULT_N_TOTAL_PLAYS,
                           filter_nan: bool = True) -> pd.DataFrame:
    """Create LLM training data in structured JSON format."""
    return training_data_generator.create_llm_training_data(df, n_total, filter_nan)


def generate_training_data_for_game(game_id: int, season_year: str = "2023-2024", 
                                  n_total: int = DEFAULT_N_TOTAL_PLAYS, 
                                  max_plays: Optional[int] = None) -> pd.DataFrame:
    """Generate LLM training data for a specific game_id."""
    return training_data_generator.generate_for_game(game_id, season_year, n_total, max_plays)


def generate_llm_dataset(season_year: Optional[str] = None, 
                        n_total: int = DEFAULT_N_TOTAL_PLAYS, 
                        sample_size: Optional[int] = None, 
                        game_id_filter: Optional[List[int]] = None) -> pd.DataFrame:
    """Generate a complete LLM training dataset from play-by-play data."""
    return training_data_generator.generate_dataset(season_year, n_total, sample_size, game_id_filter)
