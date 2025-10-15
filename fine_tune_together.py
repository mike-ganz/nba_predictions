#!/usr/bin/env python3
"""
Together.ai Fine-Tuning Orchestrator

End-to-end fine-tuning workflow for NBA play-by-play prediction models using Together.ai.

Usage:
    # Fine-tune with existing training file
    python fine_tune_together.py --training-file data/training/train.jsonl
    
    # With validation split
    python fine_tune_together.py --training-file data/training/train.jsonl --validation-split 0.1
    
    # With validation file
    python fine_tune_together.py \
        --training-file data/training/train.jsonl \
        --validation-file data/training/val.jsonl \
        --n-evals 10
    
    # Custom model and config
    python fine_tune_together.py \
        --training-file data/training/train.jsonl \
        --base-model meta-llama/Meta-Llama-3.1-70B-Instruct-Reference \
        --n-epochs 5 \
        --training-type lora
    
    # Monitor job
    python fine_tune_together.py --monitor <job-id>
    
    # List jobs
    python fine_tune_together.py --list-jobs
"""

import argparse
import os
import sys
import json
import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any

from training.together_client import (
    TogetherClient, 
    TogetherFineTuningConfig,
    split_train_validation
)
from dotenv import load_dotenv


class FineTuningJobTracker:
    """Track fine-tuning jobs in SQLite database."""
    
    def __init__(self, db_path: str = "data/training/fine_tuning_jobs.db"):
        """Initialize job tracker."""
        self.db_path = db_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Create table if doesn't exist
        self._create_table()
    
    def _create_table(self):
        """Create jobs table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS together_jobs (
                job_id TEXT PRIMARY KEY,
                base_model TEXT NOT NULL,
                training_file_id TEXT NOT NULL,
                training_file_path TEXT,
                validation_file_id TEXT,
                validation_file_path TEXT,
                output_model_name TEXT,
                status TEXT NOT NULL,
                training_type TEXT,
                n_epochs INTEGER,
                n_evals INTEGER,
                start_time TEXT NOT NULL,
                end_time TEXT,
                duration_seconds REAL,
                config_json TEXT,
                final_loss REAL,
                validation_loss REAL,
                error_message TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_job(self, job_info: Dict[str, Any], config: TogetherFineTuningConfig,
                training_file_path: Optional[str] = None,
                validation_file_path: Optional[str] = None):
        """Add a new job to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO together_jobs (
                job_id, base_model, training_file_id, training_file_path,
                validation_file_id, validation_file_path, status, training_type,
                n_epochs, n_evals, start_time, config_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job_info.get('id'),
            config.base_model,
            config.training_file_id,
            training_file_path,
            config.validation_file_id,
            validation_file_path,
            job_info.get('status', 'pending'),
            config.training_type,
            config.n_epochs,
            config.n_evals,
            datetime.now().isoformat(),
            json.dumps(config.to_api_dict())
        ))
        
        conn.commit()
        conn.close()
    
    def update_job(self, job_id: str, job_info: Dict[str, Any]):
        """Update job status."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        status = job_info.get('status')
        output_model = job_info.get('output_name')
        error = job_info.get('error')
        
        # Calculate duration if completed
        end_time = None
        duration = None
        if status in ['succeeded', 'failed', 'cancelled']:
            end_time = datetime.now().isoformat()
            
            # Get start time
            cursor.execute("SELECT start_time FROM together_jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            if row:
                start = datetime.fromisoformat(row[0])
                end = datetime.fromisoformat(end_time)
                duration = (end - start).total_seconds()
        
        cursor.execute("""
            UPDATE together_jobs 
            SET status = ?, output_model_name = ?, error_message = ?,
                end_time = ?, duration_seconds = ?
            WHERE job_id = ?
        """, (status, output_model, error, end_time, duration, job_id))
        
        conn.commit()
        conn.close()
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job from database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM together_jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def list_jobs(self, limit: int = 20) -> list:
        """List recent jobs."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM together_jobs 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]


def print_job_info(job_info: Dict[str, Any], detailed: bool = False):
    """Print job information in a nice format."""
    print(f"\n{'=' * 60}")
    print(f"Job ID: {job_info.get('id')}")
    print(f"Status: {job_info.get('status')}")
    print(f"Base Model: {job_info.get('model')}")
    
    if job_info.get('output_name'):
        print(f"Output Model: {job_info.get('output_name')}")
    
    if job_info.get('created_at'):
        print(f"Created: {job_info.get('created_at')}")
    
    if job_info.get('updated_at'):
        print(f"Updated: {job_info.get('updated_at')}")
    
    if detailed:
        print(f"\nTraining Details:")
        print(f"  Training file: {job_info.get('training_file')}")
        if job_info.get('validation_file'):
            print(f"  Validation file: {job_info.get('validation_file')}")
        
        if job_info.get('hyperparameters'):
            params = job_info['hyperparameters']
            print(f"  Epochs: {params.get('n_epochs', 'auto')}")
            print(f"  Batch size: {params.get('batch_size', 'auto')}")
            print(f"  Learning rate: {params.get('learning_rate', 'auto')}")
        
        if job_info.get('events'):
            print(f"\nRecent Events:")
            for event in job_info['events'][-5:]:  # Last 5 events
                print(f"  [{event.get('created_at')}] {event.get('message')}")
    
    if job_info.get('error'):
        print(f"\n⚠️  Error: {job_info.get('error')}")
    
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Together.ai Fine-Tuning Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # File inputs (path or existing uploaded file IDs)
    parser.add_argument('--training-file', type=str,
                       help='Path to training JSONL file')
    parser.add_argument('--validation-file', type=str,
                       help='Path to validation JSONL file (optional)')
    parser.add_argument('--training-file-id', type=str,
                       help='Use an already uploaded training file ID (skips upload)')
    parser.add_argument('--validation-file-id', type=str,
                       help='Use an already uploaded validation file ID (skips upload)')
    parser.add_argument('--validation-split', type=float,
                       help='Automatically split training file (e.g., 0.1 for 90/10 split)')
    
    # Model configuration
    parser.add_argument('--base-model', type=str,
                       default='meta-llama/Meta-Llama-3.1-8B-Instruct-Reference',
                       help='Base model to fine-tune (default: Llama-3.1-8B)')
    parser.add_argument('--training-type', type=str, default='lora',
                       choices=['lora', 'full'],
                       help='Training type: lora (fast, recommended) or full (default: lora)')
    
    # Hyperparameters
    parser.add_argument('--n-epochs', type=int,
                       help='Number of training epochs (auto if not specified)')
    parser.add_argument('--n-evals', type=int, default=0,
                       help='Number of validation evaluations (requires validation file)')
    parser.add_argument('--batch-size', type=int,
                       help='Training batch size (auto if not specified)')
    parser.add_argument('--learning-rate', type=float,
                       help='Learning rate (auto if not specified)')
    
    # LoRA parameters
    parser.add_argument('--lora-rank', type=int,
                       help='LoRA rank/r (auto if not specified)')
    parser.add_argument('--lora-alpha', type=int,
                       help='LoRA alpha (auto if not specified)')
    
    # Model naming
    parser.add_argument('--suffix', type=str,
                       help='Suffix for output model name')
    
    # W&B integration
    parser.add_argument('--wandb-key', type=str,
                       help='Weights & Biases API key for tracking')
    parser.add_argument('--wandb-project', type=str,
                       help='W&B project name')
    parser.add_argument('--wandb-name', type=str,
                       help='W&B run name')
    
    # Continuation
    parser.add_argument('--from-checkpoint', type=str,
                       help='Continue from checkpoint (job ID or model name)')
    
    # Monitoring
    parser.add_argument('--monitor', type=str, metavar='JOB_ID',
                       help='Monitor an existing job')
    parser.add_argument('--poll-interval', type=int, default=30,
                       help='Seconds between status checks (default: 30)')
    parser.add_argument('--timeout-minutes', type=int,
                       help='Maximum time to wait for job completion')
    parser.add_argument('--no-monitor', action='store_true',
                       help='Start job but don\'t monitor (fire-and-forget)')
    
    # Job management
    parser.add_argument('--list-jobs', action='store_true',
                       help='List recent fine-tuning jobs')
    parser.add_argument('--job-status', type=str, metavar='JOB_ID',
                       help='Get status of a specific job')
    parser.add_argument('--cancel-job', type=str, metavar='JOB_ID',
                       help='Cancel a running job')
    
    # Other
    parser.add_argument('--api-key', type=str,
                       help='Together.ai API key (or set TOGETHER_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Load environment variables from .env (project root) if present
    try:
        # Load default .env in current working directory
        load_dotenv()
        # Also attempt to load .env next to this script for robustness
        script_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(script_dir, '.env')
        if os.path.exists(env_path):
            load_dotenv(env_path, override=False)
    except Exception:
        pass
    
    # Initialize client
    try:
        client = TogetherClient(api_key=args.api_key)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1
    
    # Initialize job tracker
    tracker = FineTuningJobTracker()
    
    # Handle monitoring existing job
    if args.monitor:
        print(f"📊 Monitoring job: {args.monitor}")
        try:
            result = client.monitor_job(
                args.monitor,
                poll_interval=args.poll_interval,
                timeout_minutes=args.timeout_minutes,
                verbose=True
            )
            
            # Update tracker
            tracker.update_job(args.monitor, result)
            
            print_job_info(result, detailed=True)
            return 0
            
        except (TimeoutError, RuntimeError) as e:
            print(f"❌ Error: {e}")
            return 1
    
    # Handle list jobs
    if args.list_jobs:
        print("\n📋 Recent Fine-Tuning Jobs:")
        jobs = tracker.list_jobs(limit=20)
        
        if not jobs:
            print("  No jobs found.")
        else:
            for job in jobs:
                status_emoji = {
                    'pending': '⏳',
                    'running': '🏃',
                    'succeeded': '✅',
                    'failed': '❌',
                    'cancelled': '⚠️'
                }.get(job['status'], '❓')
                
                print(f"\n  {status_emoji} {job['job_id']}")
                print(f"     Status: {job['status']}")
                print(f"     Model: {job['base_model']}")
                print(f"     Started: {job['start_time']}")
                if job.get('output_model_name'):
                    print(f"     Output: {job['output_model_name']}")
        
        print()
        return 0
    
    # Handle job status
    if args.job_status:
        try:
            job_info = client.get_job_status(args.job_status)
            print_job_info(job_info, detailed=True)
            
            # Update tracker
            tracker.update_job(args.job_status, job_info)
            
            return 0
        except Exception as e:
            print(f"❌ Error getting job status: {e}")
            return 1
    
    # Handle cancel job
    if args.cancel_job:
        try:
            result = client.cancel_job(args.cancel_job)
            print(f"✓ Job {args.cancel_job} cancelled")
            
            # Update tracker
            tracker.update_job(args.cancel_job, result)
            
            return 0
        except Exception as e:
            print(f"❌ Error cancelling job: {e}")
            return 1
    
    # Main workflow: Start new fine-tuning job
    if not args.training_file and not args.training_file_id:
        print("❌ Error: Provide either --training-file or --training-file-id")
        print("   Use --help for usage information")
        return 1
    
    # If a local training file path is provided, ensure it exists. Skip when using --training-file-id only
    if args.training_file and not os.path.exists(args.training_file):
        print(f"❌ Error: Training file not found: {args.training_file}")
        return 1
    
    print(f"\n{'=' * 60}")
    print("TOGETHER.AI FINE-TUNING ORCHESTRATOR")
    print(f"{'=' * 60}\n")
    
    # Handle validation split if requested
    validation_file = args.validation_file
    if args.validation_split:
        if not args.training_file:
            print("❌ Error: --validation-split requires --training-file (cannot split an uploaded file ID)")
            return 1
        print(f"📊 Splitting dataset (validation: {args.validation_split:.1%})...")
        train_file, val_file = split_train_validation(
            args.training_file,
            train_ratio=1.0 - args.validation_split
        )
        args.training_file = train_file
        validation_file = val_file
    
    # Resolve training file ID
    train_file_id = None
    if args.training_file_id:
        train_file_id = args.training_file_id
        print(f"\n📦 Using existing training file ID: {train_file_id}")
    else:
        try:
            print(f"\n📤 Step 1: Upload Training File")
            train_response = client.upload_file(args.training_file, purpose="fine-tune")
            train_file_id = train_response['id']
            print(f"   Training file ID: {train_file_id}")
        except Exception as e:
            print(f"❌ Error uploading training file: {e}")
            return 1
    
    # Resolve validation file ID if provided
    val_file_id = None
    if args.validation_file_id:
        val_file_id = args.validation_file_id
        print(f"\n📦 Using existing validation file ID: {val_file_id}")
    elif validation_file:
        try:
            print(f"\n📤 Step 2: Upload Validation File")
            val_response = client.upload_file(validation_file, purpose="fine-tune")
            val_file_id = val_response['id']
            print(f"   Validation file ID: {val_file_id}")
        except Exception as e:
            print(f"❌ Error uploading validation file: {e}")
            return 1
    else:
        print(f"\n⏭️  Step 2: Skipped (no validation file)")
    
    # Resolve W&B credentials from CLI or environment (.env)
    wandb_api_key = args.wandb_key or os.getenv('WANDB_API_KEY')
    wandb_project = args.wandb_project or os.getenv('WANDB_PROJECT')
    wandb_name = args.wandb_name or os.getenv('WANDB_NAME')

    # Create configuration
    config = TogetherFineTuningConfig(
        training_file_id=train_file_id,
        base_model=args.base_model,
        validation_file_id=val_file_id,
        n_epochs=args.n_epochs,
        n_evals=args.n_evals if val_file_id else 0,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        training_type=args.training_type,
        suffix=args.suffix,
        wandb_api_key=wandb_api_key,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        from_checkpoint=args.from_checkpoint
    )
    
    # Create fine-tuning job
    try:
        print(f"\n🚀 Step 3: Create Fine-Tuning Job")
        job_response = client.create_fine_tuning_job(config)
        job_id = job_response['id']
        
        # Save to tracker
        tracker.add_job(
            job_response, 
            config,
            training_file_path=args.training_file,
            validation_file_path=validation_file
        )
        
        print(f"\n✅ Job created successfully!")
        print(f"   Job ID: {job_id}")
        print(f"\n💡 Tips:")
        print(f"   - Monitor: python fine_tune_together.py --monitor {job_id}")
        print(f"   - Status:  python fine_tune_together.py --job-status {job_id}")
        print(f"   - Cancel:  python fine_tune_together.py --cancel-job {job_id}")
        
    except Exception as e:
        print(f"❌ Error creating fine-tuning job: {e}")
        return 1
    
    # Monitor job unless --no-monitor
    if not args.no_monitor:
        try:
            print(f"\n📊 Step 4: Monitor Job Progress")
            result = client.monitor_job(
                job_id,
                poll_interval=args.poll_interval,
                timeout_minutes=args.timeout_minutes,
                verbose=True
            )
            
            # Update tracker with final status
            tracker.update_job(job_id, result)
            
            print_job_info(result, detailed=True)
            
            if result.get('status') == 'succeeded':
                print(f"🎉 Success! Your model is ready to use:")
                print(f"   Model: {result.get('output_name')}")
                print(f"\n💡 Next steps:")
                print(f"   1. Update your prediction config with the new model")
                print(f"   2. Run test predictions to verify performance")
                return 0
            else:
                return 1
                
        except KeyboardInterrupt:
            print(f"\n⚠️  Monitoring interrupted. Job is still running.")
            print(f"   Job ID: {job_id}")
            return 0
        except Exception as e:
            print(f"❌ Error during monitoring: {e}")
            return 1
    else:
        print(f"\n✓ Job started (fire-and-forget mode)")
        print(f"   Job ID: {job_id}")
        return 0


if __name__ == "__main__":
    sys.exit(main())

