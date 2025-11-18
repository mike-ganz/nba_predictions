#!/usr/bin/env python
"""
FastAPI wrapper for NBA Betting Recommendations Pipeline

This API provides endpoints to:
- Trigger the daily predictions pipeline
- Get prediction results
- Check pipeline status

Usage:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    POST /api/v1/predictions/generate - Generate predictions for a specific date
    GET /api/v1/predictions/{date} - Get predictions for a specific date
    GET /api/v1/health - Health check
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import logging
import pandas as pd
import os
from dotenv import load_dotenv

# Import functions from the main script
from daily_betting_recommendations_champion import (
    get_eastern_date,
    analyze_predictions,
    format_email_body,
    send_email,
    get_confidence_level,
    MODEL_NAME,
    MODEL_SHORT_NAME,
    MODEL_ARTIFACT_PATH
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NBA Predictions API",
    description="API for generating NBA betting predictions using Rest-Aware Champion v2 model",
    version="1.0.0"
)

# In-memory task storage (in production, use a database or Redis)
tasks: Dict[str, Dict[str, Any]] = {}


# Request/Response Models
class PredictionRequest(BaseModel):
    date: Optional[str] = Field(None, description="Date in YYYY-MM-DD format (defaults to today)")
    skip_scrape: bool = Field(False, description="Skip scraping step (use existing data)")
    send_email: bool = Field(True, description="Send email with recommendations")
    
    class Config:
        json_schema_extra = {
            "example": {
                "date": "2025-11-18",
                "skip_scrape": False,
                "send_email": False
            }
        }


class PredictionResponse(BaseModel):
    task_id: str
    status: str
    message: str
    date: str
    predictions_file: Optional[str] = None


class RecommendationItem(BaseModel):
    game_id: str
    date: str
    game_time: Optional[str]
    away_team: str
    home_team: str
    market_spread_home: float
    baseline_margin: float
    pred_margin_mu: float
    recommended_side: str
    recommended_team: str
    edge: float
    confidence: str


class PredictionsResult(BaseModel):
    date: str
    model_name: str
    total_games: int
    recommendations: List[RecommendationItem]


class TaskStatus(BaseModel):
    task_id: str
    status: str
    date: str
    started_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


def run_pipeline(date_str: str, skip_scrape: bool = False) -> Dict[str, Any]:
    """Run the predictions pipeline and return results."""
    try:
        logger.info(f"Running pipeline for date: {date_str}")
        
        # Define file paths
        data_file = f"data/games_future_{date_str}_norm.jsonl"
        predictions_file = f"predictions/predictions_champion_{date_str}.csv"
        
        # STEP 1: Prepare today's games (with scraping)
        if not skip_scrape:
            logger.info("Step 1: Preparing today's games...")
            cmd = [sys.executable, "prepare_todays_games.py", "--scrape-all", "--date", date_str]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Step 1 completed")
        else:
            logger.info("Skipping scrape step as requested")
        
        # Verify data file exists
        if not Path(data_file).exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # STEP 2: Generate predictions
        logger.info("Step 2: Generating predictions...")
        cmd = [
            sys.executable,
            "predict_margin.py",
            "--data", data_file,
            "--model", MODEL_ARTIFACT_PATH,
            "--output", predictions_file
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Step 2 completed")
        
        # STEP 3: Analyze predictions
        logger.info("Step 3: Analyzing predictions...")
        df, recommendations = analyze_predictions(Path(predictions_file))
        logger.info("Step 3 completed")
        
        # Add confidence levels to recommendations
        for rec in recommendations:
            rec['confidence'] = get_confidence_level(rec)
        
        return {
            "success": True,
            "date": date_str,
            "model_name": MODEL_NAME,
            "predictions_file": predictions_file,
            "total_games": len(recommendations),
            "recommendations": recommendations
        }
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Pipeline step failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        raise Exception(f"Pipeline failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


def run_pipeline_background(task_id: str, date_str: str, skip_scrape: bool, send_email_flag: bool):
    """Run pipeline in background and update task status."""
    try:
        tasks[task_id]["status"] = "running"
        result = run_pipeline(date_str, skip_scrape)
        
        # Send email if requested
        if send_email_flag:
            logger.info("Sending email...")
            api_key = os.getenv("SENDGRID_API_KEY")
            from_email = os.getenv("SENDGRID_FROM_EMAIL")
            from_alias = os.getenv("SENDGRID_FROM_ALIAS")
            to_email = os.getenv("SENDGRID_TO_EMAIL")
            
            if not all([api_key, from_email, to_email]):
                logger.warning("Email credentials not configured, skipping email")
            else:
                subject = f"NBA Betting Recommendations - {date_str}"
                html_content = format_email_body(date_str, result["recommendations"])
                send_email(api_key, from_email, to_email, subject, html_content, from_alias)
                result["email_sent"] = True
        
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["completed_at"] = datetime.now().isoformat()
        tasks[task_id]["result"] = result
        
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["completed_at"] = datetime.now().isoformat()
        tasks[task_id]["error"] = str(e)
        logger.error(f"Task {task_id} failed: {e}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "NBA Predictions API",
        "version": "1.0.0",
        "model": MODEL_NAME,
        "endpoints": {
            "health": "/api/v1/health",
            "generate_predictions": "POST /api/v1/predictions/generate",
            "get_predictions": "GET /api/v1/predictions/{date}",
            "get_task_status": "GET /api/v1/tasks/{task_id}"
        }
    }


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "model_artifact": MODEL_ARTIFACT_PATH
    }


@app.post("/api/v1/predictions/generate", response_model=PredictionResponse)
async def generate_predictions(request: PredictionRequest, background_tasks: BackgroundTasks):
    """
    Generate predictions for a specific date.
    
    This endpoint triggers the full pipeline:
    1. Scrapes current odds and injury data (unless skip_scrape=True)
    2. Prepares game features
    3. Generates predictions using the model
    4. Analyzes recommendations
    5. Optionally sends email
    
    The task runs in the background. Use the task_id to check status.
    """
    # Get date
    date_str = request.date if request.date else get_eastern_date()
    
    # Validate date format
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    # Create task
    task_id = f"pred_{date_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "date": date_str,
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "error": None,
        "result": None
    }
    
    # Run pipeline in background
    background_tasks.add_task(
        run_pipeline_background,
        task_id,
        date_str,
        request.skip_scrape,
        request.send_email
    )
    
    logger.info(f"Created task {task_id} for date {date_str}")
    
    return PredictionResponse(
        task_id=task_id,
        status="pending",
        message=f"Pipeline started for {date_str}. Use task_id to check status.",
        date=date_str
    )


@app.get("/api/v1/tasks/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get the status of a prediction task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return TaskStatus(**tasks[task_id])


@app.get("/api/v1/predictions/generate-now")
async def generate_predictions_now(background_tasks: BackgroundTasks, send_email: bool = True):
    """
    Quick-trigger endpoint to generate predictions for today (GET request for easy browser access).
    
    This is a convenience endpoint that can be triggered from a browser or email link.
    For programmatic access, use POST /api/v1/predictions/generate instead.
    
    Parameters:
    - send_email: Whether to send email with recommendations (default: True)
    """
    from fastapi.responses import HTMLResponse
    
    # Get today's date
    date_str = get_eastern_date()
    
    # Create task
    task_id = f"pred_{date_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "date": date_str,
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "error": None,
        "result": None
    }
    
    # Run pipeline in background
    background_tasks.add_task(
        run_pipeline_background,
        task_id,
        date_str,
        False,  # skip_scrape
        send_email
    )
    
    logger.info(f"Created task {task_id} for date {date_str} via quick-trigger")
    
    # Return nice HTML response
    html_content = f"""
    <html>
    <head>
        <title>Generating Predictions</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 600px;
                margin: 50px auto;
                padding: 20px;
                text-align: center;
            }}
            .container {{
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                color: white;
                padding: 40px;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            h1 {{ margin: 0 0 20px 0; }}
            .spinner {{
                border: 4px solid rgba(255,255,255,0.3);
                border-top: 4px solid white;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 20px auto;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            .info {{
                background: white;
                color: #1f2937;
                padding: 20px;
                border-radius: 8px;
                margin-top: 20px;
            }}
            code {{
                background: #f3f4f6;
                padding: 2px 6px;
                border-radius: 4px;
                font-family: monospace;
            }}
        </style>
        <meta http-equiv="refresh" content="3;url=/api/v1/tasks/{task_id}">
    </head>
    <body>
        <div class="container">
            <h1>🏀 Generating NBA Predictions</h1>
            <div class="spinner"></div>
            <p>Pipeline started for {date_str}</p>
            <p style="font-size: 14px; opacity: 0.9;">This will take 2-3 minutes...</p>
        </div>
        <div class="info">
            <p><strong>Task ID:</strong> <code>{task_id}</code></p>
            <p>You'll be redirected to the status page in a moment.</p>
            <p>Or check status manually at: <a href="/api/v1/tasks/{task_id}">/api/v1/tasks/{task_id}</a></p>
            {"<p>📧 Email will be sent when complete</p>" if send_email else ""}
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)


@app.get("/api/v1/predictions/{date}", response_model=PredictionsResult)
async def get_predictions(date: str):
    """
    Get predictions for a specific date from the predictions file.
    
    This endpoint reads existing predictions from the CSV file.
    If predictions don't exist, you need to generate them first using POST /api/v1/predictions/generate
    """
    # Validate date format
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    predictions_file = Path(f"predictions/predictions_champion_{date}.csv")
    
    if not predictions_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Predictions for {date} not found. Generate them first using POST /api/v1/predictions/generate"
        )
    
    try:
        # Read and analyze predictions
        df, recommendations = analyze_predictions(predictions_file)
        
        # Add confidence levels
        for rec in recommendations:
            rec['confidence'] = get_confidence_level(rec)
        
        return PredictionsResult(
            date=date,
            model_name=MODEL_NAME,
            total_games=len(recommendations),
            recommendations=[RecommendationItem(**rec) for rec in recommendations]
        )
        
    except Exception as e:
        logger.error(f"Failed to read predictions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read predictions: {str(e)}")


@app.delete("/api/v1/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task from memory."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    del tasks[task_id]
    return {"message": f"Task {task_id} deleted"}


@app.get("/api/v1/tasks")
async def list_tasks():
    """List all tasks."""
    return {
        "total_tasks": len(tasks),
        "tasks": list(tasks.values())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

