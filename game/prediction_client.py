"""
Base prediction client interface for multi-platform NBA play prediction.

This module provides the abstract base class for prediction clients,
enabling support for multiple platforms (OpenAI, Gemini, etc.) with a
consistent interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import json
import os
from dotenv import load_dotenv
from .response_validator import NBAResponseValidator


class BasePredictionClient(ABC):
    """Abstract base class for prediction clients."""
    
    def __init__(self):
        """Initialize the prediction client."""
        self._initialize_client()
        self.validator = NBAResponseValidator()
    
    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return the name of the platform this client targets."""
        pass
    
    @abstractmethod
    def _initialize_client(self):
        """Initialize the platform-specific client."""
        pass
    
    @abstractmethod
    def predict(self, context: Dict[str, Any], model_id: str, 
                max_tokens: int = 1500, temperature: float = 0.1) -> Tuple[str, Dict[str, Any]]:
        """
        Make a prediction using the specified model.
        
        Args:
            context: Game context dictionary
            model_id: Model identifier for the platform
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            
        Returns:
            tuple: (response_content, usage_stats)
        """
        pass
    
    def get_model_config(self) -> Dict[str, str]:
        """
        Get platform-specific model configuration.
        
        Returns:
            dict: Dictionary with 'model_1_id' and 'model_2_id' keys
        """
        return self._get_default_models()
    
    def predict_with_validation(self, context: Dict[str, Any], model_id: str, 
                               max_tokens: int = 1500, temperature: float = 0.1,
                               max_retries: int = 3) -> Tuple[str, Dict[str, Any]]:
        """
        Make a prediction with validation and retry logic.
        
        Args:
            context: Game context dictionary
            model_id: Model identifier for the platform
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            max_retries: Maximum number of retries on validation failure
            
        Returns:
            tuple: (response_content, usage_stats)
            
        Raises:
            ValueError: If validation fails after all retries
        """
        validation_attempts = []
        
        for attempt in range(max_retries + 1):
            try:
                # Make prediction
                response_content, usage_stats = self.predict(context, model_id, max_tokens, temperature)
                
                # Validate response
                is_valid, validation_errors = self.validator.validate_response(response_content, context)
                
                if is_valid:
                    if attempt > 0:
                        print(f"✅ Validation successful on attempt {attempt + 1}")
                    return response_content, usage_stats
                
                # Log validation errors
                validation_attempts.append({
                    'attempt': attempt + 1,
                    'errors': validation_errors,
                    'response_preview': response_content[:200] + "..." if len(response_content) > 200 else response_content
                })
                
                print(f"❌ Validation failed on attempt {attempt + 1}/{max_retries + 1}")
                print(f"   Errors: {len(validation_errors)} validation issues")
                
                if attempt < max_retries:
                    print(f"🔄 Retrying prediction...")
                else:
                    print(f"💥 Max retries ({max_retries}) exceeded")
                
            except Exception as e:
                print(f"❌ Prediction attempt {attempt + 1} failed with error: {str(e)}")
                if attempt == max_retries:
                    raise e
        
        # All attempts failed - create detailed error message
        error_details = []
        for attempt_info in validation_attempts:
            error_details.append(f"\nAttempt {attempt_info['attempt']}:")
            error_details.append(f"  Response preview: {attempt_info['response_preview']}")
            error_details.append(f"  Validation errors ({len(attempt_info['errors'])}):")
            for error in attempt_info['errors'][:3]:  # Show first 3 errors
                error_details.append(f"    - {error.field_path}: {error.message}")
            if len(attempt_info['errors']) > 3:
                error_details.append(f"    - ... and {len(attempt_info['errors']) - 3} more errors")
        
        full_error_message = f"Response validation failed after {max_retries + 1} attempts:{''.join(error_details)}"
        raise ValueError(full_error_message)
    
    @abstractmethod
    def _get_default_models(self) -> Dict[str, str]:
        """Get default model IDs for this platform."""
        pass


class OpenAIPredictionClient(BasePredictionClient):
    """OpenAI prediction client implementation."""
    
    @property
    def platform_name(self) -> str:
        return "openai"
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI library not found. Install with: pip install openai")
        
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not found. "
                "Please set your OpenAI API key as an environment variable."
            )
        
        self.client = OpenAI(api_key=api_key)
    
    def predict(self, context: Dict[str, Any], model_id: str, 
                max_tokens: int = 1500, temperature: float = 0.1) -> Tuple[str, Dict[str, Any]]:
        """Make prediction using OpenAI API."""
        context_json = json.dumps(context, separators=(',', ':'))
        
        response = self.client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": context_json
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        content = response.choices[0].message.content
        usage_stats = {
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        return content, usage_stats
    
    def _get_default_models(self) -> Dict[str, str]:
        """Get default OpenAI model IDs."""
        return {
            "model_1_id": "ft:gpt-4.1-nano-2025-04-14:personal:first-n-plays:CAieHHyW",
            "model_2_id": "ft:gpt-4.1-nano-2025-04-14:personal:part-1:CAoc9Uw6"
        }


class GeminiPredictionClient(BasePredictionClient):
    """Gemini prediction client implementation."""
    
    @property
    def platform_name(self) -> str:
        return "gemini"
    
    def _initialize_client(self):
        """Initialize Gemini client."""
        try:
            from vertexai import init
            from vertexai.generative_models import GenerativeModel, GenerationConfig
        except ImportError:
            raise ImportError(
                "Vertex AI library not found. Install with: pip install google-cloud-aiplatform"
            )
        
        # Initialize Vertex AI with default project (assumes gcloud CLI is configured)
        try:
            # Try to get project from environment or gcloud config
            project_id = (os.getenv("GOOGLE_CLOUD_PROJECT") or 
                         os.getenv("GCP_PROJECT") or 
                         os.getenv("GOOGLE_PROJECT_ID"))
            if not project_id:
                # Try to get from gcloud config
                import subprocess
                try:
                    result = subprocess.run(
                        ["gcloud", "config", "get-value", "project"], 
                        capture_output=True, text=True, check=True
                    )
                    project_id = result.stdout.strip()
                except (subprocess.CalledProcessError, FileNotFoundError):
                    raise ValueError(
                        "Could not determine Google Cloud project. Please set GOOGLE_CLOUD_PROJECT "
                        "environment variable or configure gcloud CLI with: gcloud config set project PROJECT_ID"
                    )
            
            # Default to us-central1 location
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            
            init(project=project_id, location=location)
            self.project_id = project_id
            self.location = location
            
        except Exception as e:
            raise ValueError(f"Failed to initialize Vertex AI: {e}")
    
    def predict(self, context: Dict[str, Any], model_id: str, 
                max_tokens: int = 1500, temperature: float = 0.1) -> Tuple[str, Dict[str, Any]]:
        """Make prediction using Gemini API."""
        # Try the new Google GenAI SDK with proper thinking control first
        try:
            return self._predict_with_genai_sdk(context, model_id, max_tokens, temperature)
        except Exception as genai_error:
            print(f"⚠️ Google GenAI SDK failed: {genai_error}")
            print("🔄 Falling back to Vertex AI approach...")
            return self._predict_with_vertexai(context, model_id, max_tokens, temperature)
    
    def _predict_with_genai_sdk(self, context: Dict[str, Any], model_id: str, 
                               max_tokens: int = 1500, temperature: float = 0.1) -> Tuple[str, Dict[str, Any]]:
        """Make prediction using the new Google GenAI SDK with thinking control."""
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("Google GenAI library not found. Install with: pip install google-generativeai")
        
        # Add instruction to encourage direct responses
        enhanced_context = context.copy()
        enhanced_context["_instruction"] = "Provide direct JSON response immediately. No reasoning or explanation needed."
        context_json = json.dumps(enhanced_context, separators=(',', ':'))
        
        # Initialize the client with Vertex AI backend
        client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location,
        )
        
        # Prepare contents for the model
        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text=context_json)]
            )
        ]
        
        # Configure generation with thinking disabled
        generate_content_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(
                thinking_budget=0,  # This is the key - disables thinking!
            ),
        )
        
        print("🔧 Using Google GenAI SDK with thinking_budget=0 (thinking disabled)")
        
        # Generate content using streaming (but collect all chunks)
        response_text = ""
        token_count = 0
        
        for chunk in client.models.generate_content_stream(
            model=model_id,
            contents=contents,
            config=generate_content_config,
        ):
            if chunk.text:
                response_text += chunk.text
                token_count += len(chunk.text.split()) * 1.3  # Rough estimate
        
        # Prepare usage stats
        usage_stats = {
            "completion_tokens": token_count,
            "prompt_tokens": len(context_json.split()) * 1.3,  # Rough estimate
            "total_tokens": len(context_json.split()) * 1.3 + token_count
        }
        
        return response_text, usage_stats
    
    def _predict_with_vertexai(self, context: Dict[str, Any], model_id: str, 
                              max_tokens: int = 1500, temperature: float = 0.1) -> Tuple[str, Dict[str, Any]]:
        """Fallback prediction using Vertex AI SDK."""
        try:
            from vertexai.generative_models import GenerativeModel, GenerationConfig
            # Try to import ThinkingConfig if available in newer versions
            try:
                from vertexai.generative_models._generative_models import ThinkingConfig
                thinking_config_available = True
            except ImportError:
                thinking_config_available = False
        except ImportError:
            raise ImportError("Vertex AI library not found. Install with: pip install google-cloud-aiplatform")
        
        # Add instruction to encourage direct responses without extensive reasoning
        enhanced_context = context.copy()
        enhanced_context["_instruction"] = "Provide direct JSON response immediately. No reasoning or explanation needed."
        
        context_json = json.dumps(enhanced_context, separators=(',', ':'))
        
        # Create model instance - model_id should be the full endpoint path
        # e.g., "projects/PROJECT_ID/locations/us-central1/endpoints/ENDPOINT_ID"
        model = GenerativeModel(model_id)
        
        # Create generation config - accommodate thinking tokens until we can disable them
        if thinking_config_available:
            try:
                # Create ThinkingConfig to disable reasoning (Gemini 2.5+ models)
                thinking_config = ThinkingConfig(thinking_budget=0)
                generation_config = GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    thinking_config=thinking_config
                )
                print("🔧 Thinking budget disabled (set to 0) for faster, direct responses")
            except Exception as e:
                print(f"⚠️ Could not disable thinking budget: {e}. Using expanded token config.")
                # Increase tokens to accommodate thinking overhead
                generation_config = GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens + 2000  # Extra tokens for thinking
                )
        else:
            print("ℹ️ ThinkingConfig not available - increasing token limit to accommodate thinking overhead")
            # Since we can't disable thinking, give the model more tokens
            # The model used 1499 thinking tokens, so we need buffer space
            generation_config = GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens + 2000  # 3500 total: ~1500 thinking + 1500+ response
            )
        
        # Generate content
        response = model.generate_content(
            context_json,
            generation_config=generation_config
        )
        
        content = response.text
        
        # Gemini doesn't provide detailed token usage in the same way
        # We'll estimate based on content length
        usage_stats = {
            "completion_tokens": len(content.split()) * 1.3,  # Rough estimate
            "prompt_tokens": len(context_json.split()) * 1.3,  # Rough estimate
            "total_tokens": len(context_json.split()) * 1.3 + len(content.split()) * 1.3
        }
        
        return content, usage_stats
    
    def _get_default_models(self) -> Dict[str, str]:
        """Get default Gemini model endpoint IDs."""
        # Get endpoint IDs from environment variables or use placeholders
        # Support multiple variable name formats
        model_1_endpoint = (os.getenv("GEMINI_MODEL_1_ENDPOINT") or 
                          os.getenv("ROLLING_PREDICTIONS_ENDPOINT_ID") or 
                          "YOUR_MODEL_1_ENDPOINT_ID")
        model_2_endpoint = (os.getenv("GEMINI_MODEL_2_ENDPOINT") or 
                          os.getenv("ROLLING_PREDICTIONS_ENDPOINT_ID") or 
                          "YOUR_MODEL_2_ENDPOINT_ID")
        
        # Format: "projects/PROJECT_ID/locations/LOCATION/endpoints/ENDPOINT_ID"
        return {
            "model_1_id": f"projects/{self.project_id}/locations/{self.location}/endpoints/{model_1_endpoint}",
            "model_2_id": f"projects/{self.project_id}/locations/{self.location}/endpoints/{model_2_endpoint}"
        }


class PredictionClientFactory:
    """Factory for creating platform-specific prediction clients."""
    
    _clients = {
        'openai': OpenAIPredictionClient,
        'gemini': GeminiPredictionClient
    }
    
    @classmethod
    def create_client(cls, platform_name: str) -> BasePredictionClient:
        """
        Create a prediction client instance for the specified platform.
        
        Args:
            platform_name: Name of the platform ('openai' or 'gemini')
            
        Returns:
            BasePredictionClient: Client instance for the platform
            
        Raises:
            ValueError: If platform is not supported
        """
        platform_key = platform_name.lower()
        if platform_key not in cls._clients:
            available = list(cls._clients.keys())
            raise ValueError(f"Unsupported platform '{platform_name}'. Available platforms: {available}")
        
        return cls._clients[platform_key]()
    
    @classmethod
    def get_available_platforms(cls) -> list:
        """
        Get list of available platform names.
        
        Returns:
            list: List of supported platform names
        """
        return list(cls._clients.keys())
    
    @classmethod
    def is_platform_supported(cls, platform_name: str) -> bool:
        """
        Check if a platform is supported.
        
        Args:
            platform_name: Name of the platform
            
        Returns:
            bool: True if platform is supported, False otherwise
        """
        return platform_name.lower() in cls._clients
