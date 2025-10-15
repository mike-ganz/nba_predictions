"""
Together.ai API client for fine-tuning operations.

This module provides a clean interface for interacting with Together.ai's
fine-tuning API, including file uploads, job management, and monitoring.
"""

import os
import time
import json
import requests
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TogetherFineTuningConfig:
    """Configuration for Together.ai fine-tuning jobs."""
    
    # Required parameters
    training_file_id: str
    base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference"
    
    # Optional parameters
    validation_file_id: Optional[str] = None
    n_epochs: Optional[int] = None  # Auto if None
    n_evals: int = 0  # Number of validation evaluations
    batch_size: Optional[int] = None  # Auto if None
    learning_rate: Optional[float] = None  # Auto if None
    
    # LoRA parameters (only for LoRA fine-tuning)
    lora_r: Optional[int] = None  # LoRA rank (auto if None)
    lora_alpha: Optional[int] = None  # LoRA alpha (auto if None)
    lora_dropout: Optional[float] = None  # LoRA dropout
    lora_trainable_modules: Optional[str] = None  # e.g., "all", "q_proj,v_proj"
    
    # Training type
    training_type: str = "lora"  # "lora" or "full"
    
    # Model suffix (appended to base model name)
    suffix: Optional[str] = None
    
    # W&B integration
    wandb_api_key: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None
    
    # Continuation
    from_checkpoint: Optional[str] = None  # Job ID or model name to continue from
    
    def to_api_dict(self) -> Dict[str, Any]:
        """Convert config to API request dictionary."""
        data = {
            "training_file": self.training_file_id,
            "model": self.base_model,
        }
        
        # Add optional parameters
        if self.validation_file_id:
            data["validation_file"] = self.validation_file_id
        if self.n_epochs is not None:
            data["n_epochs"] = self.n_epochs
        if self.n_evals > 0:
            data["n_evals"] = self.n_evals
        if self.batch_size is not None:
            data["batch_size"] = self.batch_size
        if self.learning_rate is not None:
            data["learning_rate"] = self.learning_rate
        
        # LoRA parameters
        if self.training_type == "lora":
            data["training_type"] = "lora"
            if self.lora_r is not None:
                data["lora_r"] = self.lora_r
            if self.lora_alpha is not None:
                data["lora_alpha"] = self.lora_alpha
            if self.lora_dropout is not None:
                data["lora_dropout"] = self.lora_dropout
            if self.lora_trainable_modules is not None:
                data["lora_trainable_modules"] = self.lora_trainable_modules
        else:
            data["training_type"] = "full"
        
        # Model suffix
        if self.suffix:
            data["suffix"] = self.suffix
        
        # W&B integration
        if self.wandb_api_key:
            data["wandb_api_key"] = self.wandb_api_key
            if self.wandb_project:
                data["wandb_project"] = self.wandb_project
            if self.wandb_name:
                data["wandb_name"] = self.wandb_name
        
        # Continuation
        if self.from_checkpoint:
            data["from_checkpoint"] = self.from_checkpoint
        
        return data


class TogetherClient:
    """Client for Together.ai API operations."""
    
    BASE_URL = "https://api.together.xyz/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Together.ai client.
        
        Args:
            api_key: Together.ai API key (if None, reads from TOGETHER_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Together.ai API key not found. Please set TOGETHER_API_KEY environment "
                "variable or pass api_key parameter."
            )
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def upload_file(self, file_path: str, purpose: str = "fine-tune") -> Dict[str, Any]:
        """
        Upload a file to Together.ai.
        
        Args:
            file_path: Path to the JSONL file to upload
            purpose: Purpose of the file (default: "fine-tune")
            
        Returns:
            dict: Response with file_id and metadata
            
        Raises:
            FileNotFoundError: If file doesn't exist
            requests.HTTPError: If upload fails
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"📤 Uploading file to Together.ai: {file_path}")
        
        # Get file size
        file_size = os.path.getsize(file_path)
        print(f"   File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
        
        # Upload file using multipart/form-data
        url = f"{self.BASE_URL}/files"
        
        with open(file_path, 'rb') as f:
            files = {
                'file': (os.path.basename(file_path), f, 'application/jsonl')
            }
            data = {
                'purpose': purpose
            }
            
            # Remove Content-Type from headers for multipart upload
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            response = requests.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()
        
        result = response.json()
        file_id = result.get('id')
        
        print(f"✓ Upload successful! File ID: {file_id}")
        
        return result
    
    def create_fine_tuning_job(self, config: TogetherFineTuningConfig) -> Dict[str, Any]:
        """
        Create a fine-tuning job on Together.ai.
        
        Args:
            config: Fine-tuning configuration
            
        Returns:
            dict: Response with job_id and metadata
            
        Raises:
            requests.HTTPError: If job creation fails
        """
        # Together fine-tuning API (current stable path): /fine-tunes
        url = f"{self.BASE_URL}/fine-tunes"

        # Restrict payload to documented fields per Together quickstart
        # https://docs.together.ai/docs/fine-tuning-quickstart
        cfg = config.to_api_dict()
        # Base payload
        data = {"training_file": cfg.get("training_file")}
        # If not continuing from a checkpoint, include model; otherwise, omit to inherit
        if not cfg.get("from_checkpoint"):
            data["model"] = cfg.get("model")
        # Optional: training type and suffix
        if cfg.get("training_type"):
            data["training_type"] = cfg["training_type"]
        if cfg.get("suffix"):
            data["suffix"] = cfg["suffix"]
        if cfg.get("validation_file"):
            data["validation_file"] = cfg["validation_file"]
        if isinstance(cfg.get("n_evals"), int) and cfg.get("n_evals", 0) > 0:
            data["n_evals"] = cfg["n_evals"]
        if cfg.get("from_checkpoint"):
            data["from_checkpoint"] = cfg["from_checkpoint"]
        if cfg.get("wandb_api_key"):
            data["wandb_api_key"] = cfg["wandb_api_key"]
        
        print(f"\n🚀 Creating fine-tuning job...")
        print(f"   Base model: {config.base_model}")
        print(f"   Training type: {config.training_type.upper()}")
        print(f"   Training file: {config.training_file_id}")
        if config.validation_file_id:
            print(f"   Validation file: {config.validation_file_id}")
        
        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code >= 400:
            # Raise with body for better diagnostics
            try:
                body = response.text
            except Exception:
                body = ""
            raise requests.HTTPError(f"{response.status_code} {response.reason}: {body}", response=response)
        
        result = response.json()
        job_id = result.get('id')
        
        print(f"✓ Fine-tuning job created! Job ID: {job_id}")
        
        return result
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a fine-tuning job.
        
        Args:
            job_id: Fine-tuning job ID
            
        Returns:
            dict: Job status and metadata
            
        Raises:
            requests.HTTPError: If request fails
        """
        # Together fine-tuning API (current stable path): /fine-tunes/{job_id}
        url = f"{self.BASE_URL}/fine-tunes/{job_id}"
        
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        
        return response.json()
    
    def list_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        List fine-tuning jobs.
        
        Args:
            limit: Maximum number of jobs to return
            
        Returns:
            list: List of job dictionaries
            
        Raises:
            requests.HTTPError: If request fails
        """
        # Together fine-tuning API: /fine-tunes?limit=
        url = f"{self.BASE_URL}/fine-tunes?limit={limit}"
        
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        
        result = response.json()
        return result.get('data', [])
    
    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel a fine-tuning job.
        
        Args:
            job_id: Fine-tuning job ID
            
        Returns:
            dict: Cancellation response
            
        Raises:
            requests.HTTPError: If request fails
        """
        # Together fine-tuning API: /fine-tunes/{job_id}/cancel
        url = f"{self.BASE_URL}/fine-tunes/{job_id}/cancel"
        
        print(f"⚠️  Cancelling fine-tuning job: {job_id}")
        
        response = requests.post(url, headers=self.headers)
        response.raise_for_status()
        
        print(f"✓ Job cancelled")
        
        return response.json()
    
    def list_checkpoints(self, job_id: str) -> List[Dict[str, Any]]:
        """
        List checkpoints for a fine-tuning job.
        
        Args:
            job_id: Fine-tuning job ID
            
        Returns:
            list: List of checkpoint dictionaries
            
        Raises:
            requests.HTTPError: If request fails
        """
        # Together fine-tuning API: /fine-tunes/{job_id}/checkpoints
        url = f"{self.BASE_URL}/fine-tunes/{job_id}/checkpoints"
        
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        
        result = response.json()
        return result.get('data', [])
    
    def monitor_job(self, job_id: str, poll_interval: int = 30, 
                   timeout_minutes: Optional[int] = None,
                   verbose: bool = True) -> Dict[str, Any]:
        """
        Monitor a fine-tuning job until completion.
        
        Args:
            job_id: Fine-tuning job ID
            poll_interval: Seconds between status checks (default: 30)
            timeout_minutes: Maximum time to wait (None = no timeout)
            verbose: Whether to print progress updates
            
        Returns:
            dict: Final job status
            
        Raises:
            TimeoutError: If timeout is reached
            RuntimeError: If job fails
        """
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60 if timeout_minutes else None
        
        if verbose:
            print(f"\n📊 Monitoring job: {job_id}")
            print(f"   Poll interval: {poll_interval}s")
            if timeout_minutes:
                print(f"   Timeout: {timeout_minutes} minutes")
            print()
        
        last_status = None
        iteration = 0
        
        while True:
            iteration += 1
            elapsed = time.time() - start_time
            
            # Check timeout
            if timeout_seconds and elapsed > timeout_seconds:
                raise TimeoutError(
                    f"Job monitoring timed out after {timeout_minutes} minutes. "
                    f"Job ID: {job_id} (Status: {last_status})"
                )
            
            # Get status
            try:
                job_info = self.get_job_status(job_id)
                status = job_info.get('status')
                
                # Print update if status changed or verbose mode
                if verbose and (status != last_status or iteration % 10 == 0):
                    elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
                    print(f"[{elapsed_str}] Status: {status}")
                    
                    # Print additional details if available
                    if 'training_progress' in job_info:
                        progress = job_info['training_progress']
                        print(f"          Progress: {progress}")
                    if 'events' in job_info and job_info['events']:
                        latest_event = job_info['events'][-1]
                        if 'message' in latest_event:
                            print(f"          Latest: {latest_event['message']}")
                
                last_status = status
                
                # Check terminal states
                if status == 'succeeded':
                    if verbose:
                        print(f"\n✓ Fine-tuning completed successfully!")
                        if 'output_name' in job_info:
                            print(f"  Model: {job_info['output_name']}")
                    return job_info
                
                elif status in ['failed', 'cancelled']:
                    error_msg = job_info.get('error', 'No error message provided')
                    raise RuntimeError(
                        f"Fine-tuning job {status}. Job ID: {job_id}\n"
                        f"Error: {error_msg}"
                    )
                
                # Wait before next poll
                time.sleep(poll_interval)
                
            except requests.HTTPError as e:
                if verbose:
                    print(f"⚠️  HTTP error checking status: {e}")
                    print(f"   Retrying in {poll_interval}s...")
                time.sleep(poll_interval)
            except KeyboardInterrupt:
                if verbose:
                    print(f"\n⚠️  Monitoring interrupted by user")
                    print(f"   Job is still running: {job_id}")
                    print(f"   You can check status later with: get_job_status('{job_id}')")
                raise
    
    def get_file_info(self, file_id: str) -> Dict[str, Any]:
        """
        Get information about an uploaded file.
        
        Args:
            file_id: File ID
            
        Returns:
            dict: File information
            
        Raises:
            requests.HTTPError: If request fails
        """
        url = f"{self.BASE_URL}/files/{file_id}"
        
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        
        return response.json()
    
    def delete_file(self, file_id: str) -> Dict[str, Any]:
        """
        Delete an uploaded file.
        
        Args:
            file_id: File ID to delete
            
        Returns:
            dict: Deletion response
            
        Raises:
            requests.HTTPError: If request fails
        """
        url = f"{self.BASE_URL}/files/{file_id}"
        
        print(f"🗑️  Deleting file: {file_id}")
        
        response = requests.delete(url, headers=self.headers)
        response.raise_for_status()
        
        print(f"✓ File deleted")
        
        return response.json()


def split_train_validation(input_file: str, train_ratio: float = 0.9,
                          output_dir: Optional[str] = None) -> tuple:
    """
    Split a JSONL file into training and validation sets.
    
    Args:
        input_file: Path to input JSONL file
        train_ratio: Ratio of data to use for training (default: 0.9)
        output_dir: Output directory (default: same as input file)
        
    Returns:
        tuple: (train_file_path, validation_file_path)
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Read all lines
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    train_lines = int(total_lines * train_ratio)
    
    print(f"\n✂️  Splitting dataset:")
    print(f"   Total examples: {total_lines:,}")
    print(f"   Train ratio: {train_ratio:.1%}")
    print(f"   Train examples: {train_lines:,}")
    print(f"   Validation examples: {total_lines - train_lines:,}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filenames
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    train_file = os.path.join(output_dir, f"{base_name}_train.jsonl")
    val_file = os.path.join(output_dir, f"{base_name}_val.jsonl")
    
    # Write training set
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(lines[:train_lines])
    
    # Write validation set
    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(lines[train_lines:])
    
    print(f"✓ Training set saved: {train_file}")
    print(f"✓ Validation set saved: {val_file}")
    
    return train_file, val_file


# Convenience function
def create_together_client(api_key: Optional[str] = None) -> TogetherClient:
    """
    Create a Together.ai client instance.
    
    Args:
        api_key: Together.ai API key (if None, reads from TOGETHER_API_KEY env var)
        
    Returns:
        TogetherClient: Initialized client
    """
    return TogetherClient(api_key)

