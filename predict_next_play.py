#!/usr/bin/env python3
"""
NBA Play Prediction Script

Multi-platform script to test predictions using fine-tuned models.
Supports OpenAI and Gemini platforms.
Takes team and player data and predicts the next play using the trained model.

CONFIGURATION:
=============

Platform Selection:
- Set PREDICTION_PLATFORM environment variable to "openai" or "gemini"
- Defaults to "openai" if not specified

OpenAI Setup:
1. Set OPENAI_API_KEY environment variable
2. Default model IDs are preconfigured for the provided fine-tuned models

Gemini Setup:
1. Install: pip install google-cloud-aiplatform
2. Authenticate: gcloud auth login
3. Set project: gcloud config set project YOUR_PROJECT_ID
4. Set environment variables:
   - GEMINI_MODEL_1_ENDPOINT=your_model_1_endpoint_id
   - GEMINI_MODEL_2_ENDPOINT=your_model_2_endpoint_id
   - GOOGLE_CLOUD_PROJECT=your_project_id (optional)
   - GOOGLE_CLOUD_LOCATION=us-central1 (optional)

USAGE:
======
python predict_next_play.py

The script will automatically detect your platform configuration and run
the appropriate prediction client.
"""

import json
import os
from typing import Dict, Any
from dotenv import load_dotenv
from game.prediction_client import PredictionClientFactory, BasePredictionClient

def get_prediction_platform() -> str:
    """Get the prediction platform from environment variable or default to OpenAI."""
    load_dotenv()
    platform = os.getenv("PREDICTION_PLATFORM", "gemini").lower()
    
    # Validate platform
    if not PredictionClientFactory.is_platform_supported(platform):
        available = PredictionClientFactory.get_available_platforms()
        print(f"⚠️  Warning: Unsupported platform '{platform}'. Using 'openai' instead.")
        print(f"   Available platforms: {available}")
        platform = "openai"
    
    return platform

def init_prediction_client() -> tuple[BasePredictionClient, Dict[str, str]]:
    """Initialize prediction client based on platform configuration."""
    platform = get_prediction_platform()
    print(f"🤖 Initializing {platform.upper()} client...")
    
    try:
        client = PredictionClientFactory.create_client(platform)
        model_config = client.get_model_config()
        
        print(f"✅ {platform.upper()} client initialized successfully")
        print(f"📋 Model 1: {model_config['model_1_id']}")
        print(f"📋 Model 2: {model_config['model_2_id']}")
        
        return client, model_config
        
    except Exception as e:
        print(f"❌ Failed to initialize {platform.upper()} client: {e}")
        raise

def predict_rolling_sequence(game_context: Dict[str, Any], n_iterations: int = 5, skip_stage1: bool = False) -> Dict[str, Any]:
    """
    Rolling prediction pipeline:
    1. [Optional] Send game_context to MODEL_1 to get next_plays (initial sequence)
    2. [Optional] Add next_plays to game_context as recent_plays
    3. Send to MODEL_2 to get next_play
    4. Add next_play to END of recent_plays, remove FIRST play (sliding window)
    5. Repeat step 3-4 N times
    
    Args:
        game_context: Dictionary containing team and player data
        n_iterations: Number of times to repeat the rolling prediction
        skip_stage1: If True, assumes game_context already has recent_plays and skips Stage 1
        
    Returns:
        Dict containing all stages and iterations
    """
    client, model_config = init_prediction_client()
    
    # Results storage
    results = {
        "stage1_response": None,
        "stage2_responses": [],
        "iterations": []
    }
    
    if skip_stage1:
        print("\n⚡ SKIPPING STAGE 1: Using pre-loaded recent_plays...")
        
        # Validate that game_context has recent_plays
        if "recent_plays" not in game_context:
            return {"error": "skip_stage1=True but game_context has no 'recent_plays'"}
        
        working_context = game_context.copy()
        print(f"✅ Using {len(working_context['recent_plays'])} existing recent_plays")
        results["stage1_response"] = "SKIPPED - Stage 1 bypassed for testing"
        
    else:
        # === STAGE 1: Get initial predictions ===
        print("\n🎯 STAGE 1: Getting initial next_plays from first model...")
        context_json = json.dumps(game_context, separators=(',', ':'))
        print(f"📡 Sending to model: {model_config['model_1_id']}")
        print(f"📊 Context size: {len(context_json)} characters")
        
        try:
            # Stage 1 API call with validation
            stage1_content, stage1_usage, stage1_game_ended = client.predict_with_validation(
                context=game_context,
                model_id=model_config['model_1_id'],
                max_tokens=1500,
                temperature=1.01,
                max_retries=6
            )
            
            if stage1_game_ended:
                print("🏁 Game ended during Stage 1 - terminating prediction sequence")
                results["termination_reason"] = "Game ended during Stage 1"
                return results
            
            print(f"📊 Stage 1 tokens: {stage1_usage.get('completion_tokens', 'N/A')} / 1500")
            print("✅ Stage 1 completed!")
            
            results["stage1_response"] = stage1_content
            
            # Parse Stage 1 response
            try:
                stage1_json = json.loads(stage1_content)
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse Stage 1 response as JSON: {e}")
                return {"error": f"Stage 1 JSON parse error: {e}"}
            
            # Initialize working context with recent_plays
            working_context = game_context.copy()
            if "next_plays" in stage1_json:
                working_context["recent_plays"] = stage1_json["next_plays"]
                print(f"✅ Added {len(stage1_json['next_plays'])} recent_plays to context")
            else:
                print("⚠️ Warning: No 'next_plays' found in Stage 1 response")
                working_context["recent_plays"] = []
                
        except Exception as e:
            print(f"❌ Error in Stage 1: {e}")
            raise
    
    # === ROLLING ITERATIONS ===
    print(f"\n🔄 Starting {n_iterations} rolling iterations...")
    
    try:
        for iteration in range(n_iterations):
            print(f"\n--- ITERATION {iteration + 1}/{n_iterations} ---")
            
            # Convert current context to JSON
            iteration_json = json.dumps(working_context, separators=(',', ':'))
            print(f"📡 Sending to model: {model_config['model_2_id']}")
            print(f"📊 Context size: {len(iteration_json)} characters")
            print(f"📋 Current recent_plays count: {len(working_context.get('recent_plays', []))}")
            
            # 🔍 LOG: Show the input context being sent to the model
            print("\n" + "="*60)
            print(f"📤 INPUT TO MODEL (Iteration {iteration + 1}):")
            print("="*60)
            print("📋 RECENT_PLAYS being sent:")
            if "recent_plays" in working_context:
                for i, play in enumerate(working_context["recent_plays"]):
                    play_desc = play.get('description', 'No description')
                    play_time = play.get('time_remaining', 'No time')
                    print(f"   {i+1:2d}. [{play_time}] {play_desc}")
            else:
                print("   ❌ NO recent_plays in context!")
            print("="*60)
            print("📤 FULL INPUT JSON:")
            print(iteration_json)
            print("="*60 + "\n")
            
            # Stage 2 API call with validation
            stage2_content, stage2_usage, stage2_game_ended = client.predict_with_validation(
                context=working_context,
                model_id=model_config['model_2_id'],
                max_tokens=1500,
                temperature=1.01,
                max_retries=6
            )
            
            if stage2_game_ended:
                print(f"🏁 Game ended during iteration {iteration + 1} - terminating prediction sequence")
                results["termination_reason"] = f"Game ended at iteration {iteration + 1}"
                # Still process this final response before breaking
                # Continue to process the response below, then break after processing
                should_break_after_processing = True
            else:
                should_break_after_processing = False
            
            print(f"📊 Stage 2 tokens: {stage2_usage.get('completion_tokens', 'N/A')} / 1500")
            
            # 📝 LOG: Print full model response for debugging
            print("\n" + "="*60)
            print(f"🔍 FULL MODEL RESPONSE (Iteration {iteration + 1}):")
            print("="*60)
            print(stage2_content)
            print("="*60 + "\n")
            
            # Store this iteration's response
            results["stage2_responses"].append(stage2_content)
            
            # Parse the response to get the next_play
            try:
                stage2_json = json.loads(stage2_content)
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse iteration {iteration + 1} response as JSON: {e}")
                results["iterations"].append({
                    "iteration": iteration + 1,
                    "error": f"JSON parse error: {e}",
                    "raw_response": stage2_content
                })
                continue
            
            # Extract the next_play
            if "next_play" in stage2_json:
                next_play = stage2_json["next_play"]
                print(f"✅ Got next_play: {next_play.get('description', 'No description')}")
                
                # 🔍 AUDIT: Check for scoring information
                if "shot_details" in next_play:
                    shot_details = next_play["shot_details"]
                    # Handle None values properly - JSON null becomes Python None
                    points = shot_details.get("points") if shot_details else None
                    if shot_details and points is not None and points > 0:
                        print(f"🏀 🎯 SCORING PLAY DETECTED: {shot_details.get('team', 'Unknown')} +{points} points!")
                    else:
                        points_display = points if points is not None else 'N/A'
                        print(f"📋 Non-scoring play (points: {points_display})")
                else:
                    print("⚠️ WARNING: No 'shot_details' field found in next_play")
                
                # Check if score field exists in the play
                if "score" in next_play:
                    print(f"📊 Current game score: {next_play['score']}")
                else:
                    print("⚠️ WARNING: No 'score' field found in next_play")
                
                # Update the sliding window: add to end, remove from beginning
                if "recent_plays" in working_context:
                    # Add new play to the END
                    working_context["recent_plays"].append(next_play)
                    # Remove first play (maintain window size)
                    from config.settings import DEFAULT_N_TOTAL_PLAYS
                    if len(working_context["recent_plays"]) > DEFAULT_N_TOTAL_PLAYS:
                        removed_play = working_context["recent_plays"].pop(0)
                        print(f"🔄 Sliding window: Added new play, removed: {removed_play.get('description', 'No description')}")
                    else:
                        print(f"📈 Window growing: Now {len(working_context['recent_plays'])} plays")
                
                results["iterations"].append({
                    "iteration": iteration + 1,
                    "next_play": next_play,
                    "raw_response": stage2_content,
                    "recent_plays_count": len(working_context.get("recent_plays", []))
                })
                
                # Break if game ended
                if should_break_after_processing:
                    print(f"🏁 Breaking out of prediction loop - game ended")
                    break
                
            else:
                print(f"⚠️ Warning: No 'next_play' found in iteration {iteration + 1} response")
                results["iterations"].append({
                    "iteration": iteration + 1,
                    "error": "No 'next_play' in response",
                    "raw_response": stage2_content
                })
                
                # Break if game ended (even with invalid response)
                if should_break_after_processing:
                    print(f"🏁 Breaking out of prediction loop - game ended")
                    break
        
        print(f"\n✅ Rolling sequence completed! {n_iterations} iterations done.")
        
        # 📊 SCORING ANALYSIS SUMMARY
        scoring_plays = 0
        non_scoring_plays = 0
        for iteration_result in results["iterations"]:
            if "next_play" in iteration_result:
                next_play = iteration_result["next_play"]
                # Check shot_details for scoring information, handling None values
                if "shot_details" in next_play and next_play["shot_details"]:
                    points = next_play["shot_details"].get("points")
                    if points is not None and points > 0:
                        scoring_plays += 1
                    else:
                        non_scoring_plays += 1
                else:
                    non_scoring_plays += 1
        
        print(f"\n📈 SCORING SUMMARY:")
        print(f"   🏀 Scoring plays: {scoring_plays}/{scoring_plays + non_scoring_plays}")
        print(f"   📋 Non-scoring plays: {non_scoring_plays}/{scoring_plays + non_scoring_plays}")
        if scoring_plays + non_scoring_plays > 0:
            scoring_rate = (scoring_plays / (scoring_plays + non_scoring_plays)) * 100
            print(f"   📊 Scoring rate: {scoring_rate:.1f}%")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in rolling iterations: {e}")
        raise

def main():
    """Main function to test the prediction system."""
    
    print("🏀 NBA Multi-Platform Play Prediction Test")
    print("=" * 60)
    
    # Display platform information
    platform = get_prediction_platform()
    available_platforms = PredictionClientFactory.get_available_platforms()
    print(f"📱 Platform: {platform.upper()}")
    print(f"🔧 Available platforms: {', '.join(available_platforms)}")
    
    if platform == "gemini":
        print("\n💡 Gemini Configuration Notes:")
        print("   • Ensure you've run: gcloud auth login")
        print("   • Set project: gcloud config set project YOUR_PROJECT_ID")
        print("   • Update model endpoint IDs in GEMINI_MODEL_1_ENDPOINT and GEMINI_MODEL_2_ENDPOINT")
        print("   • Or modify the _get_default_models() method in GeminiPredictionClient")
    elif platform == "openai":
        print("\n💡 OpenAI Configuration Notes:")
        print("   • Ensure OPENAI_API_KEY environment variable is set")
        print("   • Default model IDs are configured for the provided fine-tuned models")
    
    print("\n" + "=" * 60)
    
    # ==================================================================================
    # 🎯 TESTING MODE SELECTION - Change this to switch between modes
    # ==================================================================================
    TEST_MODE = "skip_stage1"  # Options: "full_pipeline" or "skip_stage1"
    # ==================================================================================
    
    if TEST_MODE == "full_pipeline":
        print("🔄 Mode: FULL PIPELINE (Stage 1 + Rolling Iterations)")
        
        # Example team and player data (original context without recent_plays)
        game_context = {"away_team":{"name":"ORL","stats":{"OEFF":113.7,"DEFF":112.7,"PACE":96.5,"REST_DAYS":2},"players":[{"name":"Franz Wagner","profile":{"offense":2.34,"defense":1.42,"shot_selection":0.69,"efficiency":-0.67,"MPG":33,"usage":26}},{"name":"Paolo Banchero","profile":{"offense":4.14,"defense":1.02,"shot_selection":1.28,"efficiency":-1.04,"MPG":35,"usage":30}},{"name":"Jonathan Isaac","profile":{"offense":-0.71,"defense":1.02,"shot_selection":0.3,"efficiency":-0.5,"MPG":16,"usage":17}},{"name":"Gary Harris","profile":{"offense":-0.41,"defense":0.52,"shot_selection":-1.29,"efficiency":-0.33,"MPG":24,"usage":12}},{"name":"Jalen Suggs","profile":{"offense":1.44,"defense":2.5,"shot_selection":-0.34,"efficiency":-1.05,"MPG":27,"usage":20}},{"name":"Wendell Carter Jr.","profile":{"offense":0.42,"defense":0.76,"shot_selection":0.46,"efficiency":-0.96,"MPG":25,"usage":18}},{"name":"Joe Ingles","profile":{"offense":0.18,"defense":-0.52,"shot_selection":-1.26,"efficiency":-0.82,"MPG":17,"usage":11}},{"name":"Markelle Fultz","profile":{"offense":-0.09,"defense":0.42,"shot_selection":0.52,"efficiency":0.67,"MPG":21,"usage":19}},{"name":"Cole Anthony","profile":{"offense":1.16,"defense":0.87,"shot_selection":0.48,"efficiency":-0.2,"MPG":22,"usage":24}},{"name":"Moritz Wagner","profile":{"offense":0.68,"defense":0.16,"shot_selection":1.33,"efficiency":-1.71,"MPG":18,"usage":23}},{"name":"Caleb Houstan","profile":{"offense":-1.09,"defense":-1.29,"shot_selection":-2.17,"efficiency":-0.26,"MPG":14,"usage":12}}]},"home_team":{"name":"CLE","stats":{"OEFF":114.9,"DEFF":112.5,"PACE":97.2,"REST_DAYS":2},"players":[{"name":"Max Strus","profile":{"offense":1.48,"defense":1.24,"shot_selection":-1.23,"efficiency":-0.44,"MPG":32,"usage":17}},{"name":"Evan Mobley","profile":{"offense":1.49,"defense":2.77,"shot_selection":1.39,"efficiency":-1.13,"MPG":31,"usage":21}},{"name":"Jarrett Allen","profile":{"offense":1.82,"defense":1.41,"shot_selection":2.04,"efficiency":-1.77,"MPG":32,"usage":20}},{"name":"Donovan Mitchell","profile":{"offense":3.79,"defense":2.55,"shot_selection":0.2,"efficiency":-1.04,"MPG":35,"usage":31}},{"name":"Darius Garland","profile":{"offense":3.38,"defense":0.87,"shot_selection":-0.12,"efficiency":-0.88,"MPG":33,"usage":25}},{"name":"Caris LeVert","profile":{"offense":1.86,"defense":1.08,"shot_selection":0.15,"efficiency":-0.01,"MPG":29,"usage":23}},{"name":"Georges Niang","profile":{"offense":0.07,"defense":0.16,"shot_selection":-1.34,"efficiency":-0.52,"MPG":22,"usage":18}},{"name":"Isaac Okoro","profile":{"offense":0.17,"defense":0.95,"shot_selection":0.19,"efficiency":-0.68,"MPG":27,"usage":14}}]}}
        
        print("📋 Game Context Summary:")
        print(f"   Away Team: {game_context['away_team']['name']} ({len(game_context['away_team']['players'])} players)")
        print(f"   Home Team: {game_context['home_team']['name']} ({len(game_context['home_team']['players'])} players)")
        print()
        
    elif TEST_MODE == "skip_stage1":
        print("⚡ Mode: SKIP STAGE 1 (Direct to Rolling Iterations)")
        
        # Pre-built context with recent_plays already included (MIN vs POR game)
        # Load your JSON data and convert null to None
        import json
        json_data = """{\"away_team\":{\"name\":\"BOS\",\"stats\":{\"OEFF\":122.7,\"DEFF\":113.2,\"PACE\":98.0,\"REST_DAYS\":2},\"players\":[{\"name\":\"Jaylen Brown\",\"profile\":{\"offense\":2.9,\"defense\":1.75,\"shot_selection\":0.34,\"efficiency\":-1.16,\"MPG\":33,\"usage\":28}},{\"name\":\"Jayson Tatum\",\"profile\":{\"offense\":3.81,\"defense\":1.28,\"shot_selection\":0.48,\"efficiency\":-1.39,\"MPG\":36,\"usage\":30}},{\"name\":\"Kristaps Porzingis\",\"profile\":{\"offense\":1.65,\"defense\":2.91,\"shot_selection\":0.91,\"efficiency\":-1.37,\"MPG\":30,\"usage\":25}},{\"name\":\"Derrick White\",\"profile\":{\"offense\":2.23,\"defense\":2.16,\"shot_selection\":-0.64,\"efficiency\":-1.21,\"MPG\":32,\"usage\":19}},{\"name\":\"Jrue Holiday\",\"profile\":{\"offense\":2.12,\"defense\":0.99,\"shot_selection\":-0.65,\"efficiency\":-1.16,\"MPG\":33,\"usage\":17}},{\"name\":\"Al Horford\",\"profile\":{\"offense\":0.3,\"defense\":0.74,\"shot_selection\":-1.52,\"efficiency\":-1.03,\"MPG\":27,\"usage\":12}},{\"name\":\"Sam Hauser\",\"profile\":{\"offense\":-0.29,\"defense\":-0.54,\"shot_selection\":-2.38,\"efficiency\":-1.04,\"MPG\":21,\"usage\":14}},{\"name\":\"Payton Pritchard\",\"profile\":{\"offense\":0.27,\"defense\":-0.72,\"shot_selection\":-1.49,\"efficiency\":-0.53,\"MPG\":20,\"usage\":16}},{\"name\":\"Luke Kornet\",\"profile\":{\"offense\":-0.49,\"defense\":0.38,\"shot_selection\":1.49,\"efficiency\":-1.83,\"MPG\":15,\"usage\":10}}]},\"home_team\":{\"name\":\"CLE\",\"stats\":{\"OEFF\":115.2,\"DEFF\":111.2,\"PACE\":97.7,\"REST_DAYS\":2},\"players\":[{\"name\":\"Isaac Okoro\",\"profile\":{\"offense\":0.26,\"defense\":0.91,\"shot_selection\":0.26,\"efficiency\":-0.79,\"MPG\":26,\"usage\":14}},{\"name\":\"Evan Mobley\",\"profile\":{\"offense\":1.48,\"defense\":2.99,\"shot_selection\":1.35,\"efficiency\":-1.08,\"MPG\":31,\"usage\":21}},{\"name\":\"Jarrett Allen\",\"profile\":{\"offense\":1.77,\"defense\":1.71,\"shot_selection\":1.91,\"efficiency\":-1.71,\"MPG\":31,\"usage\":19}},{\"name\":\"Sam Merrill\",\"profile\":{\"offense\":-0.31,\"defense\":-1.09,\"shot_selection\":-2.43,\"efficiency\":-0.95,\"MPG\":16,\"usage\":16}},{\"name\":\"Darius Garland\",\"profile\":{\"offense\":3.05,\"defense\":0.78,\"shot_selection\":0.15,\"efficiency\":-0.78,\"MPG\":33,\"usage\":26}},{\"name\":\"Caris LeVert\",\"profile\":{\"offense\":1.59,\"defense\":0.84,\"shot_selection\":0.05,\"efficiency\":0.06,\"MPG\":28,\"usage\":23}},{\"name\":\"Dean Wade\",\"profile\":{\"offense\":-0.7,\"defense\":0.56,\"shot_selection\":-2.06,\"efficiency\":-0.65,\"MPG\":20,\"usage\":11}},{\"name\":\"Georges Niang\",\"profile\":{\"offense\":-0.11,\"defense\":-0.13,\"shot_selection\":-1.31,\"efficiency\":-0.27,\"MPG\":21,\"usage\":18}},{\"name\":\"Craig Porter Jr.\",\"profile\":{\"offense\":-0.44,\"defense\":-0.89,\"shot_selection\":0.91,\"efficiency\":0.59,\"MPG\":13,\"usage\":17}}]},\"recent_plays\":[{\"quarter\":1,\"time_remaining\":\"00:42\",\"score\":\"BOS 30 - CLE 23\",\"players_on_court\":[{\"team\":\"BOS\",\"players\":[\"Payton Pritchard\",\"Jayson Tatum\",\"Jrue Holiday\",\"Sam Hauser\",\"Al Horford\"]},{\"team\":\"CLE\",\"players\":[\"Evan Mobley\",\"Georges Niang\",\"Craig Porter Jr.\",\"Dean Wade\",\"Caris LeVert\"]}],\"player\":\"Sam Hauser\",\"description\":\"MISS Hauser 32' 3PT Jump Shot\",\"shot_details\":{\"team\":\"BOS\",\"points\":0,\"x_coord\":5.3,\"y_coord\":30.8}},{\"quarter\":1,\"time_remaining\":\"00:39\",\"score\":\"BOS 30 - CLE 23\",\"players_on_court\":[{\"team\":\"BOS\",\"players\":[\"Payton Pritchard\",\"Jayson Tatum\",\"Jrue Holiday\",\"Sam Hauser\",\"Al Horford\"]},{\"team\":\"CLE\",\"players\":[\"Evan Mobley\",\"Georges Niang\",\"Craig Porter Jr.\",\"Dean Wade\",\"Caris LeVert\"]}],\"player\":\"Jrue Holiday\",\"description\":\"HOLIDAY OFF.REBOUND\",\"shot_details\":{\"team\":null,\"points\":null,\"x_coord\":null,\"y_coord\":null}},{\"quarter\":1,\"time_remaining\":\"00:33\",\"score\":\"BOS 30 - CLE 23\",\"players_on_court\":[{\"team\":\"BOS\",\"players\":[\"Payton Pritchard\",\"Jayson Tatum\",\"Jrue Holiday\",\"Sam Hauser\",\"Al Horford\"]},{\"team\":\"CLE\",\"players\":[\"Evan Mobley\",\"Georges Niang\",\"Craig Porter Jr.\",\"Dean Wade\",\"Caris LeVert\"]}],\"player\":\"Payton Pritchard\",\"description\":\"Wade STEAL : Pritchard Bad Pass Turnover\",\"shot_details\":{\"team\":null,\"points\":null,\"x_coord\":null,\"y_coord\":null}},{\"quarter\":1,\"time_remaining\":\"00:26\",\"score\":\"BOS 30 - CLE 23\",\"players_on_court\":[{\"team\":\"BOS\",\"players\":[\"Payton Pritchard\",\"Jayson Tatum\",\"Jrue Holiday\",\"Sam Hauser\",\"Al Horford\"]},{\"team\":\"CLE\",\"players\":[\"Evan Mobley\",\"Georges Niang\",\"Craig Porter Jr.\",\"Dean Wade\",\"Caris LeVert\"]}],\"player\":\"Caris LeVert\",\"description\":\"MISS LeVert 1' Running Finger Roll Layup\",\"shot_details\":{\"team\":\"CLE\",\"points\":0,\"x_coord\":24.8,\"y_coord\":42.9}},{\"quarter\":1,\"time_remaining\":\"00:24\",\"score\":\"BOS 30 - CLE 23\",\"players_on_court\":[{\"team\":\"BOS\",\"players\":[\"Payton Pritchard\",\"Jayson Tatum\",\"Jrue Holiday\",\"Sam Hauser\",\"Al Horford\"]},{\"team\":\"CLE\",\"players\":[\"Evan Mobley\",\"Georges Niang\",\"Craig Porter Jr.\",\"Dean Wade\",\"Caris LeVert\"]}],\"player\":\"Dean Wade\",\"description\":\"WADE OFF.REBOUND\",\"shot_details\":{\"team\":null,\"points\":null,\"x_coord\":null,\"y_coord\":null}},{\"quarter\":1,\"time_remaining\":\"00:19\",\"score\":\"BOS 30 - CLE 25\",\"players_on_court\":[{\"team\":\"BOS\",\"players\":[\"Payton Pritchard\",\"Jayson Tatum\",\"Jrue Holiday\",\"Sam Hauser\",\"Al Horford\"]},{\"team\":\"CLE\",\"players\":[\"Evan Mobley\",\"Georges Niang\",\"Craig Porter Jr.\",\"Dean Wade\",\"Caris LeVert\"]}],\"player\":\"Caris LeVert\",\"description\":\"LeVert 5' Turnaround Jump Shot\",\"shot_details\":{\"team\":\"CLE\",\"points\":2,\"x_coord\":24.6,\"y_coord\":37.2}},{\"quarter\":1,\"time_remaining\":\"00:00\",\"score\":\"BOS 33 - CLE 25\",\"players_on_court\":[{\"team\":\"BOS\",\"players\":[\"Payton Pritchard\",\"Jayson Tatum\",\"Jrue Holiday\",\"Sam Hauser\",\"Al Horford\"]},{\"team\":\"CLE\",\"players\":[\"Evan Mobley\",\"Georges Niang\",\"Craig Porter Jr.\",\"Dean Wade\",\"Caris LeVert\"]}],\"player\":\"Jayson Tatum\",\"description\":\"Tatum 26' 3PT Pullup Jump Shot\",\"shot_details\":{\"team\":\"BOS\",\"points\":3,\"x_coord\":38.1,\"y_coord\":27.1}},{\"quarter\":2,\"time_remaining\":\"11:44\",\"score\":\"BOS 33 - CLE 25\",\"players_on_court\":[{\"team\":\"BOS\",\"players\":[\"Payton Pritchard\",\"Jrue Holiday\",\"Jayson Tatum\",\"Al Horford\",\"Luke Kornet\"]},{\"team\":\"CLE\",\"players\":[\"Craig Porter Jr.\",\"Evan Mobley\",\"Caris LeVert\",\"Sam Merrill\",\"Georges Niang\"]}],\"player\":\"Craig Porter Jr.\",\"description\":\"MISS Porter Jr. 26' 3PT Pullup Jump Shot\",\"shot_details\":{\"team\":\"CLE\",\"points\":0,\"x_coord\":47.5,\"y_coord\":29.4}},{\"quarter\":2,\"time_remaining\":\"11:41\",\"score\":\"BOS 33 - CLE 25\",\"players_on_court\":[{\"team\":\"BOS\",\"players\":[\"Payton Pritchard\",\"Jrue Holiday\",\"Jayson Tatum\",\"Al Horford\",\"Luke Kornet\"]},{\"team\":\"CLE\",\"players\":[\"Craig Porter Jr.\",\"Evan Mobley\",\"Caris LeVert\",\"Sam Merrill\",\"Georges Niang\"]}],\"player\":\"Payton Pritchard\",\"description\":\"PRITCHARD DEF.REBOUND\",\"shot_details\":{\"team\":null,\"points\":null,\"x_coord\":null,\"y_coord\":null}},{\"quarter\":2,\"time_remaining\":\"11:18\",\"score\":\"BOS 33 - CLE 25\",\"players_on_court\":[{\"team\":\"BOS\",\"players\":[\"Payton Pritchard\",\"Jrue Holiday\",\"Jayson Tatum\",\"Al Horford\",\"Luke Kornet\"]},{\"team\":\"CLE\",\"players\":[\"Craig Porter Jr.\",\"Evan Mobley\",\"Caris LeVert\",\"Sam Merrill\",\"Georges Niang\"]}],\"player\":\"Payton Pritchard\",\"description\":\"MISS Pritchard 12' Turnaround Fadeaway Shot\",\"shot_details\":{\"team\":\"BOS\",\"points\":0,\"x_coord\":13.7,\"y_coord\":7.7}}]}"""
        game_context = json.loads(json_data)  # This properly converts null to None
        
        print("📋 Pre-built Context Summary:")
        print(f"   Away Team: {game_context['away_team']['name']} ({len(game_context['away_team']['players'])} players)")
        print(f"   Home Team: {game_context['home_team']['name']} ({len(game_context['home_team']['players'])} players)")
        print(f"   Recent Plays: {len(game_context['recent_plays'])} plays already loaded")
        print(f"   Current Score: {game_context['recent_plays'][-1]['score']}")
        print(f"   Last Play: {game_context['recent_plays'][-1]['description']}")
        print()
        
    else:
        print(f"❌ Invalid TEST_MODE: {TEST_MODE}")
        print("   Valid options: 'full_pipeline' or 'skip_stage1'")
        return
    
    try:
        # Configure rolling sequence parameters
        n_iterations = 400  # Change this to control how many rolling predictions
        
        print(f"\n🚀 Starting rolling prediction sequence (N={n_iterations})")
        
        # Run the rolling sequence based on the selected mode
        skip_stage1 = (TEST_MODE == "skip_stage1")
        results = predict_rolling_sequence(game_context, n_iterations=n_iterations, skip_stage1=skip_stage1)
        
        print("\n" + "=" * 80)
        print("🎯 ROLLING SEQUENCE RESULTS")
        print("=" * 80)
        
        # Show Stage 1 results
        if results.get("stage1_response"):
            print("\n📋 STAGE 1 (Initial next_plays):")
            try:
                stage1_json = json.loads(results["stage1_response"])
                print(f"Generated {len(stage1_json.get('next_plays', []))} initial plays")
                # Show first few plays
                for i, play in enumerate(stage1_json.get('next_plays', [])[:3]):
                    print(f"  {i+1}. {play.get('time_remaining', 'N/A')} - {play.get('description', 'No description')}")
                if len(stage1_json.get('next_plays', [])) > 3:
                    print(f"  ... and {len(stage1_json.get('next_plays', [])) - 3} more plays")
            except json.JSONDecodeError:
                print("❌ Could not parse Stage 1 response")
        
        # Show rolling iterations
        print(f"\n🔄 ROLLING ITERATIONS ({len(results.get('iterations', []))} completed):")
        for iteration_data in results.get("iterations", []):
            iteration_num = iteration_data.get("iteration", "?")
            print(f"\n--- Iteration {iteration_num} ---")
            
            if "error" in iteration_data:
                print(f"❌ Error: {iteration_data['error']}")
            elif "next_play" in iteration_data:
                next_play = iteration_data["next_play"]
                time_remaining = next_play.get("time_remaining", "N/A")
                description = next_play.get("description", "No description")
                score = next_play.get("score", "N/A")
                recent_plays_count = iteration_data.get("recent_plays_count", "?")
                
                print(f"⏰ Time: {time_remaining}")
                print(f"🏀 Play: {description}")
                print(f"📊 Score: {score}")
                print(f"📋 Recent plays window: {recent_plays_count} plays")
        
        print("\n" + "=" * 80)
        print("✅ ROLLING SEQUENCE COMPLETE!")
        print("=" * 80)
        
        # Summary
        total_predictions = len(results.get("iterations", []))
        successful_predictions = len([i for i in results.get("iterations", []) if "next_play" in i])
        print(f"📊 Summary: {successful_predictions}/{total_predictions} successful predictions")
        
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        platform = get_prediction_platform()
        print(f"\n💡 Troubleshooting tips for {platform.upper()}:")
        
        if platform == "openai":
            print("   • Make sure OPENAI_API_KEY environment variable is set")
            print("   • Verify the fine-tuned model ID is correct")
            print("   • Check your OpenAI account has access to the model")
        elif platform == "gemini":
            print("   • Ensure you've authenticated: gcloud auth login")
            print("   • Set correct project: gcloud config set project YOUR_PROJECT_ID")
            print("   • Set GEMINI_MODEL_1_ENDPOINT and GEMINI_MODEL_2_ENDPOINT environment variables")
            print("   • Verify your fine-tuned model endpoints are deployed and accessible")
            print("   • Check that the google-cloud-aiplatform library is installed")
        
        print("   • Check your internet connection")
        print("   • Verify the input data format is correct")

if __name__ == "__main__":
    main()
