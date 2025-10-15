# Together.ai Fine-Tuning Guide

Complete guide for fine-tuning NBA play-by-play prediction models using Together.ai.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Data Format](#data-format)
- [Training Pipeline](#training-pipeline)
- [Fine-Tuning](#fine-tuning)
- [Monitoring Jobs](#monitoring-jobs)
- [Prediction Integration](#prediction-integration)
- [Cost Estimates](#cost-estimates)
- [Troubleshooting](#troubleshooting)

## Overview

Together.ai provides fast, cost-effective fine-tuning with support for:

- **LoRA fine-tuning**: Faster, cheaper, recommended for most use cases (default)
- **Full fine-tuning**: Updates all weights for maximum performance
- **Multiple base models**: Llama 3.1 8B, 70B, 405B
- **Auto-hyperparameters**: Automatically tunes learning rate, batch size, epochs
- **W&B integration**: Track training metrics with Weights & Biases
- **Continued training**: Resume from checkpoints for iterative improvement

### Why Together.ai?

- ✅ **OpenAI-compatible format**: Use existing training data
- ✅ **Fast training**: LoRA fine-tuning in minutes, not hours
- ✅ **Cost-effective**: Pay only for what you use
- ✅ **Flexible deployment**: Serverless or dedicated endpoints
- ✅ **Great for experimentation**: Quick iteration cycles

## Quick Start

### 1. Setup

```bash
# Install dependencies (if not already installed)
pip install -r requirements.txt

# Set API key
export TOGETHER_API_KEY=your_api_key_here
```

### 2. Generate Training Data

```bash
# Generate Together.ai format training data
python generate_2023_2024_season.py \
    --format together \
    --data-format compact \
    --season 2023-2024 \
    --sample 10000
```

This creates: `data/training/nba_2023_2024_together_compact_remaining_plays_YYYYMMDD_HHMMSS.jsonl`

### 3. Start Fine-Tuning

```bash
# Fine-tune with auto-split validation
python fine_tune_together.py \
    --training-file data/training/your_file.jsonl \
    --validation-split 0.1 \
    --n-evals 10
```

The script will:
1. Split data into train/validation (90/10)
2. Upload files to Together.ai
3. Create fine-tuning job
4. Monitor progress until completion
5. Display final model name

### 4. Use Fine-Tuned Model

```python
# Update your prediction config
export TOGETHER_MODEL_ENDPOINT=your_fine_tuned_model_name

# Run predictions (after integration - see below)
python predict_next_play.py
```

## Data Format

Together.ai uses the **same format as OpenAI** (conversational messages):

```json
{
  "messages": [
    {
      "role": "user",
      "content": "{\"A\":\"ORL\",\"H\":\"CLE\",...}"
    },
    {
      "role": "assistant",
      "content": "{\"y\":[1,705,[2,3],[\"H\",3],\"made2\",2,0]}"
    }
  ]
}
```

This means:
- ✅ No format conversion needed from OpenAI format
- ✅ Can reuse existing OpenAI training data
- ✅ Easy migration between platforms

### Supported Generation Modes

#### 1. `remaining_plays` (Standard)
- **Purpose**: Next-play prediction training
- **Volume**: ~400 examples per game
- **Use case**: Main fine-tuning approach

```bash
python generate_2023_2024_season.py \
    --format together \
    --generation-mode remaining_plays \
    --n-total 12
```

#### 2. `first_N_plays` (Sequence)
- **Purpose**: Game opening sequence generation
- **Volume**: 1 example per game
- **Use case**: Seed Stage 1 predictions

```bash
python generate_2023_2024_season.py \
    --format together \
    --generation-mode first_N_plays \
    --n-total 20
```

## Training Pipeline

### Complete Workflow

```bash
# Step 1: Generate training data
python generate_2023_2024_season.py \
    --format together \
    --data-format compact \
    --season 2023-2024 \
    --games 100  # Limit for testing

# Step 2: Fine-tune model
python fine_tune_together.py \
    --training-file data/training/nba_2023_2024_together_*.jsonl \
    --base-model meta-llama/Meta-Llama-3.1-8B-Instruct-Reference \
    --validation-split 0.1 \
    --n-evals 10 \
    --suffix "nba-v1"

# Step 3: Monitor (if not done automatically)
python fine_tune_together.py --monitor <job-id>

# Step 4: Use model in predictions
# (Update prediction pipeline with new model name)
```

### Data Format Options

#### Compact Format (Recommended)
```bash
--data-format compact
```
- Smaller file size
- Faster uploads
- Efficient token usage
- **Recommended for production**

#### Verbose Format
```bash
--data-format verbose
```
- Human-readable
- Easier debugging
- Larger file size
- Good for initial testing

### Training Data Size Recommendations

| Games | Examples | File Size | Training Time | Use Case |
|-------|----------|-----------|---------------|----------|
| 10    | ~4,000   | ~5 MB     | 5-10 min      | Quick test |
| 50    | ~20,000  | ~25 MB    | 10-20 min     | Development |
| 100   | ~40,000  | ~50 MB    | 20-40 min     | Small dataset |
| 500   | ~200,000 | ~250 MB   | 1-2 hours     | Medium dataset |
| 1000+ | ~400,000 | ~500 MB   | 2-4 hours     | Full dataset |

## Fine-Tuning

### Basic Fine-Tuning

```bash
python fine_tune_together.py \
    --training-file data/training/train.jsonl \
    --base-model meta-llama/Meta-Llama-3.1-8B-Instruct-Reference
```

### With Validation Set

```bash
# Option 1: Separate validation file
python fine_tune_together.py \
    --training-file data/training/train.jsonl \
    --validation-file data/training/val.jsonl \
    --n-evals 10

# Option 2: Auto-split from training file
python fine_tune_together.py \
    --training-file data/training/full.jsonl \
    --validation-split 0.1 \
    --n-evals 10
```

### Custom Hyperparameters

```bash
python fine_tune_together.py \
    --training-file data/training/train.jsonl \
    --n-epochs 5 \
    --learning-rate 2e-5 \
    --batch-size 4 \
    --lora-rank 16 \
    --lora-alpha 32
```

**Note**: Auto hyperparameters (default) are usually optimal. Only customize if you have specific requirements.

### Training Types

#### LoRA Fine-Tuning (Default, Recommended)
```bash
--training-type lora
```
- Fast training (minutes to hours)
- Low memory requirements
- Cost-effective
- Great for most use cases
- **Recommended**

#### Full Fine-Tuning
```bash
--training-type full
```
- Trains all model weights
- Longer training time
- Higher cost
- May provide better results for complex tasks
- Use only if LoRA doesn't meet requirements

### Base Model Selection

```bash
# Small, fast (recommended for development)
--base-model meta-llama/Meta-Llama-3.1-8B-Instruct-Reference

# Large, powerful (better accuracy)
--base-model meta-llama/Meta-Llama-3.1-70B-Instruct-Reference

# Huge (maximum performance, expensive)
--base-model meta-llama/Meta-Llama-3.1-405B-Instruct-Reference
```

### Weights & Biases Integration

```bash
python fine_tune_together.py \
    --training-file data/training/train.jsonl \
    --wandb-key your_wandb_api_key \
    --wandb-project nba-predictions \
    --wandb-name llama-8b-experiment-1
```

Benefits:
- Track training metrics in real-time
- Compare multiple runs
- Visualize learning curves
- Share results with team

### Continued Fine-Tuning

Resume training from a previous checkpoint:

```bash
# Continue from job ID
python fine_tune_together.py \
    --training-file data/training/new_data.jsonl \
    --from-checkpoint ft-job-abc123

# Continue from specific step
python fine_tune_together.py \
    --training-file data/training/new_data.jsonl \
    --from-checkpoint ft-job-abc123:500
```

Use cases:
- Iterative improvement with new data
- Domain adaptation
- Fixing underperformance

## Monitoring Jobs

### Real-Time Monitoring

```bash
# Monitor during creation (default)
python fine_tune_together.py --training-file train.jsonl

# Monitor existing job
python fine_tune_together.py --monitor <job-id>

# With timeout
python fine_tune_together.py --monitor <job-id> --timeout-minutes 120
```

### Fire-and-Forget Mode

```bash
# Start job without monitoring
python fine_tune_together.py \
    --training-file train.jsonl \
    --no-monitor

# Check status later
python fine_tune_together.py --job-status <job-id>
```

### Job Management

```bash
# List recent jobs
python fine_tune_together.py --list-jobs

# Get job status
python fine_tune_together.py --job-status <job-id>

# Cancel running job
python fine_tune_together.py --cancel-job <job-id>
```

### Job Tracking Database

All jobs are automatically saved to `data/training/fine_tuning_jobs.db` with:
- Job ID and status
- Model names (base and output)
- File paths (training, validation)
- Configuration (hyperparameters)
- Timestamps and duration
- Error messages (if failed)

Query the database:
```python
import sqlite3
conn = sqlite3.connect('data/training/fine_tuning_jobs.db')
cursor = conn.cursor()
cursor.execute("SELECT * FROM together_jobs ORDER BY created_at DESC LIMIT 10")
for row in cursor.fetchall():
    print(row)
```

## Prediction Integration

### Step 1: Get Model Name

After successful fine-tuning:
```
✓ Fine-tuning completed successfully!
  Model: your_account/ft-llama-3.1-8b-nba-v1-abc123
```

### Step 2: Update Prediction Configuration

**Note**: Full prediction pipeline integration is in progress. Current options:

#### Option A: Use Together.ai Python SDK
```python
import together

together.api_key = "your_api_key"

response = together.Complete.create(
    model="your_account/ft-llama-3.1-8b-nba-v1-abc123",
    prompt=json.dumps(game_context),
    max_tokens=200,
    temperature=0.7
)
```

#### Option B: Use OpenAI-Compatible Endpoint
Together.ai provides OpenAI-compatible API:
```python
from openai import OpenAI

client = OpenAI(
    api_key="your_together_api_key",
    base_url="https://api.together.xyz/v1"
)

response = client.chat.completions.create(
    model="your_account/ft-llama-3.1-8b-nba-v1-abc123",
    messages=[
        {"role": "user", "content": json.dumps(game_context)}
    ]
)
```

### Step 3: Update `predict_next_play.py` (Coming Soon)

Integration with existing prediction pipeline in development.

## Cost Estimates

### Training Costs (Approximate)

**LoRA Fine-Tuning:**
| Model Size | Dataset Size | Training Time | Estimated Cost |
|------------|--------------|---------------|----------------|
| 8B         | 10K examples | 10-20 min     | $1-2           |
| 8B         | 100K examples| 1-2 hours     | $5-10          |
| 8B         | 500K examples| 3-5 hours     | $20-40         |
| 70B        | 10K examples | 30-60 min     | $5-10          |
| 70B        | 100K examples| 2-4 hours     | $20-50         |

**Full Fine-Tuning:**
- 5-10x more expensive than LoRA
- Longer training times
- Higher memory requirements

### Inference Costs

- **Serverless**: Pay per token
- **Dedicated**: Reserved capacity with guaranteed throughput

See [Together.ai Pricing](https://www.together.ai/pricing) for current rates.

### Cost Optimization Tips

1. **Start with LoRA**: Much cheaper, usually sufficient
2. **Use validation split**: Catch issues early with 10% validation
3. **Start small**: Test with 10K examples before full dataset
4. **Use 8B model**: Fine-tune on 8B, upgrade to 70B only if needed
5. **Monitor training**: Stop early if metrics plateau
6. **Reuse checkpoints**: Continue from checkpoints instead of retraining

## Troubleshooting

### File Upload Errors

**Problem**: File upload fails or times out

**Solutions**:
- Check file size (max ~500MB recommended)
- Verify JSONL format (one JSON per line)
- Check internet connection
- Try splitting into smaller files

### Training Fails Immediately

**Problem**: Job fails at start

**Common causes**:
- Invalid data format
- Corrupted JSONL file
- Insufficient examples (<100)
- Invalid base model name

**Solutions**:
```bash
# Validate JSONL format
python -c "
import json
with open('train.jsonl') as f:
    for i, line in enumerate(f):
        try:
            json.loads(line)
        except:
            print(f'Error at line {i+1}')
"

# Check example count
wc -l data/training/train.jsonl

# Verify base model name
python fine_tune_together.py --list-jobs
```

### Training Takes Too Long

**Problem**: Training exceeds expected time

**Solutions**:
- Check dataset size (may be too large)
- Consider using LoRA instead of full fine-tuning
- Use smaller base model (8B instead of 70B)
- Reduce n_epochs if specified

### Low Model Performance

**Problem**: Fine-tuned model doesn't predict well

**Solutions**:
1. **More data**: Increase training examples
2. **Better data**: Improve data quality, remove outliers
3. **Validation**: Add validation set to monitor overfitting
4. **Hyperparameters**: Try different learning rates
5. **Longer training**: Increase n_epochs
6. **Larger model**: Upgrade from 8B to 70B
7. **Full fine-tuning**: Try full instead of LoRA

### API Key Errors

**Problem**: Authentication fails

**Solutions**:
```bash
# Check API key is set
echo $TOGETHER_API_KEY

# Set API key
export TOGETHER_API_KEY=your_key_here

# Or pass directly
python fine_tune_together.py --api-key your_key_here ...
```

### Job Tracking Database Errors

**Problem**: Database locked or corrupted

**Solutions**:
```bash
# Reset database
rm data/training/fine_tuning_jobs.db

# Database will be recreated automatically
```

## Advanced Topics

### Custom Data Preparation

Split existing OpenAI/Together.ai JSONL:
```python
from training.together_client import split_train_validation

train_file, val_file = split_train_validation(
    'data/training/full_dataset.jsonl',
    train_ratio=0.9,
    output_dir='data/training'
)
```

### Programmatic API Usage

```python
from training.together_client import TogetherClient, TogetherFineTuningConfig

# Initialize client
client = TogetherClient()

# Upload files
train_response = client.upload_file('train.jsonl')
val_response = client.upload_file('val.jsonl')

# Create configuration
config = TogetherFineTuningConfig(
    training_file_id=train_response['id'],
    validation_file_id=val_response['id'],
    base_model='meta-llama/Meta-Llama-3.1-8B-Instruct-Reference',
    n_evals=10,
    suffix='nba-v1'
)

# Start job
job = client.create_fine_tuning_job(config)

# Monitor
result = client.monitor_job(job['id'], poll_interval=30)
print(f"Model ready: {result['output_name']}")
```

### Checkpoint Management

```python
# List checkpoints for a job
checkpoints = client.list_checkpoints('ft-job-abc123')

for ckpt in checkpoints:
    print(f"Step {ckpt['step']}: Loss {ckpt['loss']}")

# Continue from best checkpoint
python fine_tune_together.py \
    --training-file new_data.jsonl \
    --from-checkpoint ft-job-abc123:1000
```

## Next Steps

1. **Generate training data**: Start with `--games 10` for testing
2. **Run first fine-tune**: Use LoRA on 8B model
3. **Evaluate results**: Test predictions on validation games
4. **Iterate**: Adjust data/hyperparameters based on performance
5. **Scale up**: Increase dataset size or model size as needed
6. **Deploy**: Integrate with prediction pipeline

## Resources

- [Together.ai Documentation](https://docs.together.ai/docs/fine-tuning-quickstart)
- [Together.ai Pricing](https://www.together.ai/pricing)
- [Llama 3.1 Model Card](https://ai.meta.com/llama/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## Support

For issues or questions:
1. Check this guide and troubleshooting section
2. Review Together.ai documentation
3. Check job logs: `python fine_tune_together.py --job-status <job-id>`
4. Review database: `sqlite3 data/training/fine_tuning_jobs.db`

---

**Last Updated**: 2025-01-14
**Version**: 1.0

