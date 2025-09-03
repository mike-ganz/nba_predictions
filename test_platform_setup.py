#!/usr/bin/env python3
"""
Test script to verify multi-platform prediction setup.
This script helps debug configuration issues for both OpenAI and Gemini platforms.
"""

import os
import sys
from dotenv import load_dotenv

def test_platform_setup():
    """Test the setup for both prediction platforms."""
    load_dotenv()
    
    print("🔧 NBA Multi-Platform Prediction Setup Test")
    print("=" * 60)
    
    # Test platform selection
    platform = os.getenv("PREDICTION_PLATFORM", "openai").lower()
    print(f"📱 Selected Platform: {platform.upper()}")
    
    print("\n🔍 Testing platform imports and configurations...")
    
    # Test imports
    try:
        from game.prediction_client import PredictionClientFactory
        print("✅ Prediction client factory imported successfully")
        
        available_platforms = PredictionClientFactory.get_available_platforms()
        print(f"✅ Available platforms: {', '.join(available_platforms)}")
        
    except ImportError as e:
        print(f"❌ Failed to import prediction client: {e}")
        return False
    
    # Test OpenAI
    print(f"\n🤖 Testing OpenAI Configuration:")
    openai_configured = True
    
    try:
        import openai as openai_module
        print("✅ OpenAI library available")
    except ImportError:
        print("❌ OpenAI library not installed (pip install openai)")
        openai_configured = False
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("✅ OPENAI_API_KEY environment variable set")
    else:
        print("❌ OPENAI_API_KEY environment variable not set")
        openai_configured = False
    
    if openai_configured:
        try:
            client = PredictionClientFactory.create_client("openai")
            print("✅ OpenAI client created successfully")
            model_config = client.get_model_config()
            print(f"✅ Model 1: {model_config['model_1_id'][:50]}...")
            print(f"✅ Model 2: {model_config['model_2_id'][:50]}...")
        except Exception as e:
            print(f"❌ Failed to create OpenAI client: {e}")
            openai_configured = False
    
    # Test Gemini
    print(f"\n🔮 Testing Gemini Configuration:")
    gemini_configured = True
    
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel, GenerationConfig
        print("✅ Vertex AI library available")
    except ImportError:
        print("❌ Vertex AI library not installed (pip install google-cloud-aiplatform)")
        gemini_configured = False
    
    # Check gcloud authentication
    try:
        import subprocess
        result = subprocess.run(
            ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"], 
            capture_output=True, text=True, check=True
        )
        if result.stdout.strip():
            print(f"✅ gcloud authenticated as: {result.stdout.strip()}")
        else:
            print("❌ No active gcloud authentication (run: gcloud auth login)")
            gemini_configured = False
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ gcloud CLI not available or not authenticated")
        gemini_configured = False
    
    # Check project configuration
    try:
        import subprocess
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"], 
            capture_output=True, text=True, check=True
        )
        project_id = result.stdout.strip()
        if project_id:
            print(f"✅ gcloud project set: {project_id}")
        else:
            print("❌ No gcloud project set (run: gcloud config set project YOUR_PROJECT)")
            gemini_configured = False
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Could not get gcloud project configuration")
        gemini_configured = False
    
    # Check Gemini model endpoints
    model_1_endpoint = os.getenv("GEMINI_MODEL_1_ENDPOINT")
    model_2_endpoint = os.getenv("GEMINI_MODEL_2_ENDPOINT")
    
    if model_1_endpoint and model_1_endpoint != "YOUR_MODEL_1_ENDPOINT_ID":
        print(f"✅ GEMINI_MODEL_1_ENDPOINT set: {model_1_endpoint}")
    else:
        print("❌ GEMINI_MODEL_1_ENDPOINT not properly configured")
        gemini_configured = False
    
    if model_2_endpoint and model_2_endpoint != "YOUR_MODEL_2_ENDPOINT_ID":
        print(f"✅ GEMINI_MODEL_2_ENDPOINT set: {model_2_endpoint}")
    else:
        print("❌ GEMINI_MODEL_2_ENDPOINT not properly configured")
        gemini_configured = False
    
    if gemini_configured:
        try:
            client = PredictionClientFactory.create_client("gemini")
            print("✅ Gemini client created successfully")
            # Don't test model config as it requires actual endpoints
        except Exception as e:
            print(f"❌ Failed to create Gemini client: {e}")
            gemini_configured = False
    
    # Summary
    print(f"\n📊 Configuration Summary:")
    print(f"✅ OpenAI Ready: {'Yes' if openai_configured else 'No'}")
    print(f"✅ Gemini Ready: {'Yes' if gemini_configured else 'No'}")
    
    if platform == "openai" and not openai_configured:
        print(f"⚠️  WARNING: Selected platform (OpenAI) is not properly configured!")
        return False
    elif platform == "gemini" and not gemini_configured:
        print(f"⚠️  WARNING: Selected platform (Gemini) is not properly configured!")
        return False
    else:
        print(f"✅ Selected platform ({platform.upper()}) is properly configured!")
        return True

if __name__ == "__main__":
    success = test_platform_setup()
    sys.exit(0 if success else 1)
