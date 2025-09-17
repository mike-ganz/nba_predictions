"""
Base prediction client interface for multi-platform NBA play prediction.

This module provides the abstract base class for prediction clients,
enabling support for multiple platforms (OpenAI, Gemini, etc.) with a
consistent interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import json
import os
from dotenv import load_dotenv
from .response_validator import NBAResponseValidator, ValidationResult


class BasePredictionClient(ABC):
    """Abstract base class for prediction clients."""
    
    def __init__(self, validation_mode: str = "fast"):
        """Initialize the prediction client."""
        self._initialize_client()
        self.validator = NBAResponseValidator(validation_mode=validation_mode)
        self.last_validation_failure = None  # Store detailed validation failure info
    
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
    
    def predict_from_json(self, context_json: str, model_id: str, 
                         max_tokens: int = 1500, temperature: float = 0.1) -> Tuple[str, Dict[str, Any]]:
        """
        Make a prediction using pre-serialized JSON context (performance optimization).
        
        Args:
            context_json: Pre-serialized JSON context string
            model_id: Model identifier for the platform
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            
        Returns:
            tuple: (response_content, usage_stats)
        """
        # Default implementation: parse JSON and call regular predict
        # Subclasses can override this for better performance
        try:
            context = json.loads(context_json)
            return self.predict(context, model_id, max_tokens, temperature)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON context: {e}")
    
    def get_model_config(self) -> Dict[str, str]:
        """
        Get platform-specific model configuration.
        
        Returns:
            dict: Dictionary with 'model_1_id' and 'model_2_id' keys
        """
        return self._get_default_models()
    
    def predict_with_validation(self, context: Dict[str, Any], model_id: str, 
                               max_tokens: int = 1500, temperature: float = 0.1,
                               max_retries: int = 3, stage1_mode: bool = False) -> Tuple[str, Dict[str, Any], bool, bool, Optional[Dict[str, Any]]]:
        """
        Make a prediction with validation and retry logic.
        
        Args:
            context: Game context dictionary
            model_id: Model identifier for the platform
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            max_retries: Maximum number of retries on validation failure
            stage1_mode: If True, validates for Stage 1 (next_plays array) instead of Stage 2 (next_play object)
            
        Returns:
            tuple: (response_content, usage_stats, is_game_ended, needs_rollback, termination_info)
                - response_content: The validated response text
                - usage_stats: Token usage statistics  
                - is_game_ended: True if the game has ended naturally
                - needs_rollback: True if timestamp rollback is required
                - termination_info: Detailed termination info dict if game ended via validation, None otherwise
            
        Raises:
            ValueError: If validation fails after all retries
        """
        validation_attempts = []
        
        for attempt in range(max_retries + 1):
            try:
                # Make prediction
                response_content, usage_stats = self.predict(context, model_id, max_tokens, temperature)
                
                # Validate response (different logic for Stage 1 vs Stage 2)
                if stage1_mode:
                    validation_result, validation_errors, reason = self._validate_stage1_response(response_content, context)
                else:
                    validation_result, validation_errors, reason = self.validator.validate_response(response_content, context)
                
                if validation_result == ValidationResult.VALID:
                    if attempt > 0:
                        print(f"✅ Validation successful on attempt {attempt + 1}")
                    return response_content, usage_stats, False, False, None
                
                elif validation_result == ValidationResult.END_GAME:
                    print(f"🏁 Game ending condition detected: {reason}")
                    print(f"🎮 Returning final play and ending game loop")
                    
                    # Capture detailed termination information
                    termination_info = None
                    if not stage1_mode and self.validator.has_termination_record():
                        termination_info = self.validator.get_termination_for_database()
                        print(f"📊 Captured termination details: {termination_info.get('validation_termination_type', 'unknown')}")
                        
                        # Also print summary to console
                        if self.validator.get_termination_info():
                            print(f"🛑 TERMINATION SUMMARY:")
                            print(f"   Type: {termination_info.get('validation_termination_type', 'N/A')}")
                            print(f"   Trigger: {termination_info.get('validation_trigger_condition', 'N/A')}")
                            print(f"   Game State: Q{termination_info.get('validation_game_state_quarter', '?')} {termination_info.get('validation_game_state_time', 'N/A')}")
                    
                    return response_content, usage_stats, True, False, termination_info
                
                elif validation_result == ValidationResult.ROLLBACK_TIME:
                    print(f"🔄 Timestamp rollback required: {reason}")
                    print(f"🎮 Returning rollback signal to main loop")
                    
                    # Capture detailed rollback termination information
                    termination_info = None
                    if not stage1_mode and self.validator.has_termination_record():
                        termination_info = self.validator.get_termination_for_database()
                        print(f"📊 Captured rollback details: {termination_info.get('validation_termination_type', 'unknown')}")
                        
                        # Also print summary to console
                        if self.validator.get_termination_info():
                            print(f"🔄 ROLLBACK SUMMARY:")
                            print(f"   Trigger: {termination_info.get('validation_trigger_condition', 'N/A')}")
                            print(f"   Consecutive Count: {termination_info.get('validation_consecutive_count', 0)}")
                            print(f"   Total Attempts: {termination_info.get('validation_total_attempts', 0)}")
                    
                    # Note: response_content may be invalid, but needs_rollback=True signals the main loop to handle this
                    return response_content, usage_stats, False, True, termination_info
                
                elif validation_result == ValidationResult.RETRY:
                    # Log validation retry reason
                    validation_attempts.append({
                        'attempt': attempt + 1,
                        'reason': reason,
                        'errors': validation_errors,
                        'response_preview': response_content[:200] + "..." if len(response_content) > 200 else response_content
                    })
                    
                    print(f"🔄 Validation requires retry on attempt {attempt + 1}/{max_retries + 1}")
                    print(f"   Reason: {reason}")
                    if validation_errors:
                        print(f"   Additional errors: {len(validation_errors)} validation issues")
                    
                    if attempt < max_retries:
                        print(f"🔄 Retrying prediction...")
                    else:
                        print(f"💥 Max retries ({max_retries}) exceeded")
                
            except Exception as e:
                print(f"❌ Prediction attempt {attempt + 1} failed with error: {str(e)}")
                if attempt == max_retries:
                    raise e
        
        # All attempts failed - create detailed error message and validation failure info
        error_details = []
        for attempt_info in validation_attempts:
            error_details.append(f"\nAttempt {attempt_info['attempt']}:")
            error_details.append(f"  Response preview: {attempt_info['response_preview']}")
            error_details.append(f"  Retry reason: {attempt_info['reason']}")
            if attempt_info['errors']:
                error_details.append(f"  Additional validation errors ({len(attempt_info['errors'])}):")
                for error in attempt_info['errors'][:3]:  # Show first 3 errors
                    error_details.append(f"    - {error.field_path}: {error.message}")
                if len(attempt_info['errors']) > 3:
                    error_details.append(f"    - ... and {len(attempt_info['errors']) - 3} more errors")
        
        full_error_message = f"Response validation failed after {max_retries + 1} attempts:{''.join(error_details)}"
        
        # Create comprehensive validation failure info for database storage
        validation_failure_info = self._create_validation_failure_info(validation_attempts, max_retries + 1)
        
        # Store validation failure info for potential capture by orchestrator
        self.last_validation_failure = validation_failure_info
        
        raise ValueError(full_error_message)
    
    def _create_validation_failure_info(self, validation_attempts: List[Dict[str, Any]], total_attempts: int) -> Dict[str, Any]:
        """Create comprehensive validation failure information for database storage."""
        from datetime import datetime
        
        if not validation_attempts:
            return {}
        
        # Analyze failure patterns
        failure_reasons = {}
        error_types = {}
        error_fields = {}
        
        for attempt in validation_attempts:
            reason = attempt['reason']
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            
            # Analyze validation errors
            for error in attempt.get('errors', []):
                error_type = error.error_type
                error_field = error.field_path
                
                error_types[error_type] = error_types.get(error_type, 0) + 1
                error_fields[error_field] = error_fields.get(error_field, 0) + 1
        
        # Get most common issues
        most_common_reason = max(failure_reasons.items(), key=lambda x: x[1]) if failure_reasons else ("unknown", 0)
        most_common_error_type = max(error_types.items(), key=lambda x: x[1]) if error_types else ("none", 0)
        most_common_field = max(error_fields.items(), key=lambda x: x[1]) if error_fields else ("none", 0)
        
        # Get examples of failed responses (first and last)
        response_examples = []
        if len(validation_attempts) > 0:
            response_examples.append(validation_attempts[0]['response_preview'])
            if len(validation_attempts) > 1:
                response_examples.append(validation_attempts[-1]['response_preview'])
        
        return {
            'validation_failure_timestamp': datetime.now().isoformat(),
            'validation_total_failed_attempts': total_attempts,
            'validation_most_common_reason': most_common_reason[0],
            'validation_most_common_reason_count': most_common_reason[1],
            'validation_most_common_error_type': most_common_error_type[0],
            'validation_most_common_error_type_count': most_common_error_type[1],
            'validation_most_common_field': most_common_field[0],
            'validation_most_common_field_count': most_common_field[1],
            'validation_unique_reasons': len(failure_reasons),
            'validation_unique_error_types': len(error_types),
            'validation_unique_fields': len(error_fields),
            'validation_failure_summary': json.dumps({
                'reasons': failure_reasons,
                'error_types': error_types,
                'error_fields': error_fields
            }),
            'validation_response_examples': json.dumps(response_examples)
        }
    
    def get_last_validation_failure_info(self) -> Optional[Dict[str, Any]]:
        """Get the most recent validation failure information."""
        return self.last_validation_failure
    
    def _validate_stage1_response(self, response_text: str, context: Dict[str, Any]) -> Tuple[ValidationResult, list, str]:
        """
        Validate Stage 1 response which should return multiple plays.
        Expected formats: 
        - Verbose: {"next_plays": [play1, play2, ..., playN]}
        - Compact: {"p": [[play_tuple1], [play_tuple2], ..., [play_tupleN]]}
        """
        try:
            response_data = json.loads(response_text.strip())
        except json.JSONDecodeError as e:
            return ValidationResult.RETRY, [], f"JSON parse error: {str(e)}"
        
        # Check for plays array - handle both compact ("y") and verbose ("next_plays") formats
        next_plays = None
        
        if "y" in response_data:
            # Compact format - convert play tuples to minimal verbose format for validation
            play_tuples = response_data["y"]
            if not isinstance(play_tuples, list):
                return ValidationResult.RETRY, [], "'y' field must be an array"
            
            next_plays = []
            for i, play_tuple in enumerate(play_tuples):
                if not isinstance(play_tuple, list) or len(play_tuple) < 6:
                    return ValidationResult.RETRY, [], f"Play tuple {i+1} invalid format"
                
                # Convert tuple to minimal play object for validation
                quarter = play_tuple[0]
                time_seconds = play_tuple[1]
                score_array = play_tuple[2] if len(play_tuple[2]) >= 2 else [0, 0]
                actor = play_tuple[3]
                event_code = play_tuple[4]
                
                # Convert time back to MM:SS format
                minutes = time_seconds // 60
                seconds = time_seconds % 60
                time_remaining = f"{minutes:02d}:{seconds:02d}"
                
                # Create minimal play object
                play = {
                    "quarter": quarter,
                    "time_remaining": time_remaining,
                    "description": f"Predicted {event_code}",  # Satisfies validation requirement
                    "score": f"AWAY {score_array[0]} - HOME {score_array[1]}",
                    "_compact_format": True
                }
                next_plays.append(play)
                
        elif "next_plays" in response_data:
            # Verbose format - use as-is
            next_plays = response_data["next_plays"]
        else:
            return ValidationResult.RETRY, [], "Missing 'next_plays' field (verbose) or 'y' field (compact)"
        
        # Validate it's an array
        if not isinstance(next_plays, list):
            if "y" in response_data:
                return ValidationResult.RETRY, [], "'y' field must be an array"
            else:
                return ValidationResult.RETRY, [], "'next_plays' must be an array"
        
        # Check array length (should have reasonable number of plays)
        if len(next_plays) == 0:
            if "y" in response_data:
                return ValidationResult.RETRY, [], "'y' array cannot be empty"
            else:
                return ValidationResult.RETRY, [], "'next_plays' array cannot be empty"
        
        if len(next_plays) > 50:  # Reasonable upper limit
            field_name = "'y'" if "y" in response_data else "'next_plays'"
            return ValidationResult.RETRY, [], f"{field_name} array too large ({len(next_plays)} plays)"
        
        # Basic validation of each play (less strict than Stage 2)
        for i, play in enumerate(next_plays):
            if not isinstance(play, dict):
                return ValidationResult.RETRY, [], f"Play {i+1} must be a dictionary"
                
            # Check for basic required fields
            required_fields = ["description"]  # Minimal requirement for Stage 1
            for field in required_fields:
                if field not in play:
                    return ValidationResult.RETRY, [], f"Play {i+1} missing required field: {field}"
        
        print(f"Stage 1 validation passed: {len(next_plays)} plays received")
        return ValidationResult.VALID, [], "Valid Stage 1 response"
    
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
        return self.predict_from_json(context_json, model_id, max_tokens, temperature)
    
    def predict_from_json(self, context_json: str, model_id: str, 
                         max_tokens: int = 1500, temperature: float = 0.1) -> Tuple[str, Dict[str, Any]]:
        """Make prediction using pre-serialized JSON (optimized)."""
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
        context_json = json.dumps(context, separators=(',', ':'))
        return self.predict_from_json(context_json, model_id, max_tokens, temperature)
    
    def predict_from_json(self, context_json: str, model_id: str, 
                         max_tokens: int = 1500, temperature: float = 0.1) -> Tuple[str, Dict[str, Any]]:
        """Make prediction using pre-serialized JSON (optimized)."""
        # Try the new Google GenAI SDK with proper thinking control first
        try:
            return self._predict_with_genai_sdk_json(context_json, model_id, max_tokens, temperature)
        except Exception as genai_error:
            print(f"⚠️ Google GenAI SDK failed: {genai_error}")
            print("🔄 Falling back to Vertex AI approach...")
            return self._predict_with_vertexai_json(context_json, model_id, max_tokens, temperature)
    
    def _predict_with_genai_sdk(self, context: Dict[str, Any], model_id: str, 
                               max_tokens: int = 1500, temperature: float = 0.1) -> Tuple[str, Dict[str, Any]]:
        """Make prediction using the new Google GenAI SDK with thinking control."""
        # Add instruction to encourage direct responses
        enhanced_context = context.copy()
        enhanced_context["_instruction"] = "Provide direct JSON response immediately. No reasoning or explanation needed."
        context_json = json.dumps(enhanced_context, separators=(',', ':'))
        return self._predict_with_genai_sdk_json(context_json, model_id, max_tokens, temperature)
    
    def _predict_with_genai_sdk_json(self, context_json: str, model_id: str, 
                                    max_tokens: int = 1500, temperature: float = 0.1) -> Tuple[str, Dict[str, Any]]:
        """Make prediction using the new Google GenAI SDK with pre-serialized JSON (optimized)."""
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("Google GenAI library not found. Install with: pip install google-generativeai")
        
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
        # Add instruction to encourage direct responses without extensive reasoning
        enhanced_context = context.copy()
        enhanced_context["_instruction"] = "Provide direct JSON response immediately. No reasoning or explanation needed."
        context_json = json.dumps(enhanced_context, separators=(',', ':'))
        return self._predict_with_vertexai_json(context_json, model_id, max_tokens, temperature)
    
    def _predict_with_vertexai_json(self, context_json: str, model_id: str, 
                                   max_tokens: int = 1500, temperature: float = 0.1) -> Tuple[str, Dict[str, Any]]:
        """Fallback prediction using Vertex AI SDK with pre-serialized JSON (optimized)."""
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
                          os.getenv("INITIAL_PREDICTION_ENDPOINT_ID") or 
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
    def create_client(cls, platform_name: str, validation_mode: str = None) -> BasePredictionClient:
        """
        Create a prediction client instance for the specified platform.
        
        Args:
            platform_name: Name of the platform ('openai' or 'gemini')
            validation_mode: Validation mode ('fast', 'normal', 'strict'). 
                           If None, uses VALIDATION_MODE env var or defaults to 'fast'
            
        Returns:
            BasePredictionClient: Client instance for the platform
            
        Raises:
            ValueError: If platform is not supported
        """
        platform_key = platform_name.lower()
        if platform_key not in cls._clients:
            available = list(cls._clients.keys())
            raise ValueError(f"Unsupported platform '{platform_name}'. Available platforms: {available}")
        
        # Determine validation mode
        if validation_mode is None:
            validation_mode = os.getenv("VALIDATION_MODE", "fast")
        
        if validation_mode not in ["fast", "normal", "strict"]:
            print(f"⚠️ Invalid validation mode '{validation_mode}', defaulting to 'fast'")
            validation_mode = "fast"
        
        return cls._clients[platform_key](validation_mode=validation_mode)
    
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
