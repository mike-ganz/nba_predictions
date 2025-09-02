#!/usr/bin/env python3
"""
NBA Play Prediction Script

Simple script to test predictions using a fine-tuned OpenAI model.
Takes team and player data and predicts the next play using the trained model.
"""

import json
from dotenv import load_dotenv
import os
from typing import Dict, Any, Tuple
from openai import OpenAI

# Fine-tuned model IDs
MODEL_1_ID = "ft:gpt-4.1-nano-2025-04-14:personal:first-n-plays:CAieHHyW"  # First model
MODEL_2_ID = "ft:gpt-4.1-nano-2025-04-14:personal:part-1:CAoc9Uw6"  # Second model

def init_openai_client() -> OpenAI:
    """Initialize OpenAI client with API key."""
    load_dotenv()  # this loads the .env file
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not found. "
            "Please set your OpenAI API key as an environment variable."
        )
    
    return OpenAI(api_key=api_key)

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
    print("🤖 Initializing OpenAI client...")
    client = init_openai_client()
    
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
        print(f"📡 Sending to model: {MODEL_1_ID}")
        print(f"📊 Context size: {len(context_json)} characters")
        
        try:
            # Stage 1 API call
            stage1_response = client.chat.completions.create(
                model=MODEL_1_ID,
                messages=[
                    {
                        "role": "user",
                        "content": context_json
                    }
                ],
                max_tokens=1500,
                temperature=0.1
            )
            
            stage1_content = stage1_response.choices[0].message.content
            print(f"📊 Stage 1 tokens: {stage1_response.usage.completion_tokens} / 1500")
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
            print(f"📡 Sending to model: {MODEL_2_ID}")
            print(f"📊 Context size: {len(iteration_json)} characters")
            print(f"📋 Current recent_plays count: {len(working_context.get('recent_plays', []))}")
            
            # 🔍 LOG: Show the input context being sent to OpenAI
            print("\n" + "="*60)
            print(f"📤 INPUT TO OPENAI (Iteration {iteration + 1}):")
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
            
            # Stage 2 API call
            stage2_response = client.chat.completions.create(
                model=MODEL_2_ID,
                messages=[
                    {
                        "role": "user",
                        "content": iteration_json
                    }
                ],
                max_tokens=1500,
                temperature=0.1
            )
            
            stage2_content = stage2_response.choices[0].message.content
            print(f"📊 Stage 2 tokens: {stage2_response.usage.completion_tokens} / 1500")
            
            # 📝 LOG: Print full OpenAI response for debugging
            print("\n" + "="*60)
            print(f"🔍 FULL OPENAI RESPONSE (Iteration {iteration + 1}):")
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
                if "scoring" in next_play:
                    scoring = next_play["scoring"]
                    if scoring and scoring.get("points", 0) > 0:
                        print(f"🏀 🎯 SCORING PLAY DETECTED: {scoring.get('team', 'Unknown')} +{scoring['points']} points!")
                    else:
                        print(f"📋 Non-scoring play (points: {scoring.get('points', 0) if scoring else 'N/A'})")
                else:
                    print("⚠️ WARNING: No 'scoring' field found in next_play")
                
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
                    if len(working_context["recent_plays"]) > 10:
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
                
            else:
                print(f"⚠️ Warning: No 'next_play' found in iteration {iteration + 1} response")
                results["iterations"].append({
                    "iteration": iteration + 1,
                    "error": "No 'next_play' in response",
                    "raw_response": stage2_content
                })
        
        print(f"\n✅ Rolling sequence completed! {n_iterations} iterations done.")
        
        # 📊 SCORING ANALYSIS SUMMARY
        scoring_plays = 0
        non_scoring_plays = 0
        for iteration_result in results["iterations"]:
            if "next_play" in iteration_result:
                next_play = iteration_result["next_play"]
                if "scoring" in next_play and next_play["scoring"] and next_play["scoring"].get("points", 0) > 0:
                    scoring_plays += 1
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
    
    print("🏀 NBA Play Prediction Test")
    print("=" * 50)
    
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
        
        # Pre-built context with recent_plays already included (from your example)
        game_context = {"away_team":{"name":"GSW","stats":{"OEFF":118.6,"DEFF":115.5,"PACE":99.2,"REST_DAYS":1},"players":[{"name":"Andrew Wiggins","profile":{"offense":0.54,"defense":0.72,"shot_selection":0.42,"efficiency":-0.02,"MPG":27,"usage":20}},{"name":"Jonathan Kuminga","profile":{"offense":1.39,"defense":0.69,"shot_selection":1.23,"efficiency":-1.03,"MPG":26,"usage":24}},{"name":"Draymond Green","profile":{"offense":1.91,"defense":2.34,"shot_selection":0.15,"efficiency":-1.04,"MPG":27,"usage":16}},{"name":"Brandin Podziemski","profile":{"offense":1.01,"defense":0.42,"shot_selection":-0.3,"efficiency":-0.34,"MPG":26,"usage":19}},{"name":"Stephen Curry","profile":{"offense":4.34,"defense":0.58,"shot_selection":-0.44,"efficiency":-1.89,"MPG":33,"usage":31}},{"name":"Klay Thompson","profile":{"offense":1.39,"defense":0.41,"shot_selection":-1.02,"efficiency":-0.5,"MPG":31,"usage":23}},{"name":"Gary Payton II","profile":{"offense":-0.74,"defense":0.79,"shot_selection":-0.41,"efficiency":-0.36,"MPG":16,"usage":15}},{"name":"Kevon Looney","profile":{"offense":0.01,"defense":0.38,"shot_selection":1.79,"efficiency":-1.03,"MPG":18,"usage":12}},{"name":"Lester Quinones","profile":{"offense":-0.85,"defense":-0.77,"shot_selection":-0.6,"efficiency":0.27,"MPG":13,"usage":18}},{"name":"Trayce Jackson-Davis","profile":{"offense":-0.28,"defense":0.18,"shot_selection":2.22,"efficiency":-1.64,"MPG":13,"usage":17}},{"name":"Dario Saric","profile":{"offense":0.73,"defense":0.2,"shot_selection":-0.17,"efficiency":-0.95,"MPG":19,"usage":19}}]},"home_team":{"name":"UTA","stats":{"OEFF":118.1,"DEFF":120.1,"PACE":99.0,"REST_DAYS":1},"players":[{"name":"Lauri Markkanen","profile":{"offense":1.77,"defense":0.81,"shot_selection":0.07,"efficiency":-1.16,"MPG":33,"usage":25}},{"name":"John Collins","profile":{"offense":0.9,"defense":1.55,"shot_selection":0.28,"efficiency":-0.96,"MPG":28,"usage":20}},{"name":"Walker Kessler","profile":{"offense":0.14,"defense":3.41,"shot_selection":1.31,"efficiency":-1.52,"MPG":23,"usage":13}},{"name":"Collin Sexton","profile":{"offense":2.69,"defense":0.3,"shot_selection":0.92,"efficiency":-1.19,"MPG":25,"usage":28}},{"name":"Keyonte George","profile":{"offense":1.8,"defense":-0.75,"shot_selection":-0.13,"efficiency":-0.36,"MPG":25,"usage":21}},{"name":"Kris Dunn","profile":{"offense":0.75,"defense":1.04,"shot_selection":-0.41,"efficiency":-0.48,"MPG":18,"usage":15}},{"name":"Taylor Hendricks","profile":{"offense":-1.14,"defense":-0.0,"shot_selection":-1.22,"efficiency":0.57,"MPG":15,"usage":19}},{"name":"Jordan Clarkson","profile":{"offense":2.76,"defense":-0.43,"shot_selection":0.24,"efficiency":-0.41,"MPG":30,"usage":26}},{"name":"Talen Horton-Tucker","profile":{"offense":0.89,"defense":0.78,"shot_selection":0.18,"efficiency":0.27,"MPG":20,"usage":24}}]},"recent_plays":[{"quarter":3,"time_remaining":"10:31","score":"GSW 90 - UTA 71","player":"Keyonte George","description":"George OFF.Foul","scoring":{"team":None,"points":0}},{"quarter":3,"time_remaining":"10:31","score":"GSW 90 - UTA 71","player":"Keyonte George","description":"George Offensive Foul Turnover","scoring":{"team":None,"points":0}},{"quarter":3,"time_remaining":"10:22","score":"GSW 90 - UTA 71","player":"Stephen Curry","description":"MISS Curry 24' 3PT Pullup Jump Shot","scoring":{"team":None,"points":0}},{"quarter":3,"time_remaining":"10:19","score":"GSW 90 - UTA 71","player":"Brandin Podziemski","description":"Podziemski REBOUND","scoring":{"team":None,"points":0}},{"quarter":3,"time_remaining":"10:17","score":"GSW 90 - UTA 71","player":"Andrew Wiggins","description":"George STEAL : Wiggins Bad Pass Turnover","scoring":{"team":None,"points":0}},{"quarter":3,"time_remaining":"10:11","score":"GSW 90 - UTA 74","player":"Collin Sexton","description":"Sexton 24' 3PT Running Jump Shot","scoring":{"team":"UTA","points":3}},{"quarter":3,"time_remaining":"09:57","score":"GSW 90 - UTA 74","player":"Stephen Curry","description":"Curry Traveling Turnover","scoring":{"team":None,"points":0}},{"quarter":3,"time_remaining":"09:45","score":"GSW 90 - UTA 74","player":"Collin Sexton","description":"Sexton Bad Pass Turnover : Green STEAL","scoring":{"team":None,"points":0}},{"quarter":3,"time_remaining":"09:41","score":"GSW 92 - UTA 74","player":"Jonathan Kuminga","description":"Kuminga 3' Running Alley Oop Dunk Shot","scoring":{"team":"GSW","points":2}},{"quarter":3,"time_remaining":"09:15","score":"GSW 92 - UTA 76","player":"Collin Sexton","description":"Sexton 11' Pullup Jump Shot","scoring":{"team":"UTA","points":2}}]}
        
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
        n_iterations = 25  # Change this to control how many rolling predictions
        
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
        print("\n💡 Troubleshooting tips:")
        print("   • Make sure OPENAI_API_KEY environment variable is set")
        print("   • Verify the fine-tuned model ID is correct")
        print("   • Check your OpenAI account has access to the model")

if __name__ == "__main__":
    main()
