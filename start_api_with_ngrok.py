#!/usr/bin/env python
"""
Startup script to run the NBA Predictions API with ngrok tunneling.

This script:
1. Starts the FastAPI server on a local port
2. Creates an ngrok tunnel to expose it publicly
3. Displays the public URL for accessing the API

Usage:
    python start_api_with_ngrok.py
    
    # With custom port
    python start_api_with_ngrok.py --port 8000
    
    # With ngrok auth token (for custom domains, longer sessions)
    python start_api_with_ngrok.py --ngrok-token YOUR_TOKEN

Requirements:
    - ngrok must be installed and available in PATH
    - Run: pip install -r requirements.txt
"""

import argparse
import os
import sys
import signal
import time
import logging
from pathlib import Path
from threading import Thread
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global process handles for cleanup
uvicorn_process = None
ngrok_process = None


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    logger.info("\n\nShutting down...")
    cleanup()
    sys.exit(0)


def cleanup():
    """Clean up processes on exit."""
    global uvicorn_process, ngrok_process
    
    if uvicorn_process:
        logger.info("Stopping FastAPI server...")
        uvicorn_process.terminate()
        try:
            uvicorn_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            uvicorn_process.kill()
    
    if ngrok_process:
        logger.info("Stopping ngrok tunnel...")
        ngrok_process.terminate()
        try:
            ngrok_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ngrok_process.kill()


def start_api_server(port: int):
    """Start the FastAPI server using uvicorn."""
    global uvicorn_process
    
    logger.info(f"Starting FastAPI server on port {port}...")
    
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--reload"
    ]
    
    uvicorn_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Wait a bit for server to start
    time.sleep(3)
    
    if uvicorn_process.poll() is not None:
        output = uvicorn_process.stdout.read()
        logger.error(f"FastAPI server failed to start:\n{output}")
        return False
    
    logger.info(f"✓ FastAPI server started on http://localhost:{port}")
    return True


def start_ngrok_tunnel(port: int, auth_token: str = None):
    """Start ngrok tunnel."""
    global ngrok_process
    
    logger.info(f"Starting ngrok tunnel on port {port}...")
    
    # Set auth token if provided
    if auth_token:
        logger.info("Setting ngrok auth token...")
        subprocess.run(
            ["ngrok", "config", "add-authtoken", auth_token],
            check=True,
            capture_output=True
        )
    
    # Start ngrok
    cmd = ["ngrok", "http", str(port)]
    
    ngrok_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for ngrok to start
    time.sleep(3)
    
    if ngrok_process.poll() is not None:
        _, stderr = ngrok_process.communicate()
        logger.error(f"ngrok failed to start:\n{stderr}")
        return False
    
    logger.info("✓ ngrok tunnel started")
    return True


def get_ngrok_url():
    """Get the public ngrok URL from the API."""
    import requests
    
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:4040/api/tunnels", timeout=2)
            if response.status_code == 200:
                data = response.json()
                for tunnel in data.get("tunnels", []):
                    if tunnel.get("proto") == "https":
                        return tunnel.get("public_url")
            time.sleep(1)
        except Exception as e:
            if i == max_retries - 1:
                logger.warning(f"Could not get ngrok URL: {e}")
            time.sleep(1)
    
    return None


def send_email(api_key: str, from_email: str, to_email: str, subject: str, html_content: str, from_alias: str = None) -> bool:
    """Send email via SendGrid API (copied from daily_betting_recommendations_champion.py)."""
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail, From
        
        # Build from_email with optional alias
        if from_alias:
            from_email_obj = From(from_email, from_alias)
        else:
            from_email_obj = from_email
        
        message = Mail(
            from_email=from_email_obj,
            to_emails=to_email,
            subject=subject,
            html_content=html_content
        )
        
        sg = SendGridAPIClient(api_key)
        response = sg.send(message)
        
        logger.info(f"✓ Startup email sent to {to_email}")
        return True
        
    except ImportError:
        logger.warning("SendGrid library not installed. Skipping email.")
        return False
    except Exception as e:
        logger.warning(f"Failed to send email: {e}")
        return False


def send_ngrok_email(public_url: str, recipient_email: str) -> bool:
    """Send email with ngrok URL using SendGrid."""
    from datetime import datetime
    
    # Get SendGrid credentials from environment
    api_key = os.getenv("SENDGRID_API_KEY")
    from_email = os.getenv("SENDGRID_FROM_EMAIL")
    from_alias = os.getenv("SENDGRID_FROM_ALIAS")
    
    if not api_key or not from_email:
        logger.warning("SendGrid credentials not configured. Skipping email.")
        logger.info("Set SENDGRID_API_KEY and SENDGRID_FROM_EMAIL in .env to enable email")
        return False
    
    # Create HTML email
    docs_url = f"{public_url}/docs"
    html_content = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                    line-height: 1.6;
                    color: #1f2937;
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f9fafb;
                }}
                .container {{
                    background-color: #ffffff;
                    border-radius: 12px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
                    overflow: hidden;
                }}
                .header {{
                    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                    color: #ffffff;
                    padding: 32px 40px;
                    text-align: center;
                }}
                .header h1 {{
                    margin: 0 0 8px 0;
                    font-size: 24px;
                    font-weight: 700;
                }}
                .header p {{
                    margin: 0;
                    font-size: 14px;
                    opacity: 0.95;
                }}
                .content {{
                    padding: 32px 40px;
                }}
                .status-badge {{
                    display: inline-block;
                    background-color: #d1fae5;
                    color: #065f46;
                    padding: 8px 16px;
                    border-radius: 6px;
                    font-weight: 600;
                    font-size: 14px;
                    margin-bottom: 24px;
                }}
                .url-box {{
                    background-color: #f3f4f6;
                    border: 2px solid #10b981;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 24px 0;
                    text-align: center;
                }}
                .url-box .label {{
                    font-size: 12px;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                    color: #6b7280;
                    margin-bottom: 8px;
                    font-weight: 600;
                }}
                .url-box .url {{
                    font-size: 16px;
                    color: #059669;
                    font-weight: 600;
                    word-break: break-all;
                    margin: 8px 0;
                }}
                .btn {{
                    display: inline-block;
                    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                    color: #ffffff;
                    padding: 14px 32px;
                    border-radius: 8px;
                    text-decoration: none;
                    font-weight: 600;
                    font-size: 16px;
                    margin: 16px 0;
                    box-shadow: 0 4px 6px rgba(16, 185, 129, 0.3);
                }}
                .info-section {{
                    background-color: #eff6ff;
                    border-left: 4px solid #3b82f6;
                    padding: 16px;
                    margin: 24px 0;
                    border-radius: 4px;
                }}
                .info-section h3 {{
                    margin: 0 0 12px 0;
                    color: #1e40af;
                    font-size: 16px;
                }}
                .info-section ul {{
                    margin: 8px 0;
                    padding-left: 20px;
                }}
                .info-section li {{
                    margin: 8px 0;
                    color: #1f2937;
                }}
                .footer {{
                    text-align: center;
                    padding: 20px;
                    color: #9ca3af;
                    font-size: 12px;
                    border-top: 1px solid #e5e7eb;
                }}
                code {{
                    background-color: #f3f4f6;
                    padding: 2px 6px;
                    border-radius: 4px;
                    font-family: 'Courier New', monospace;
                    font-size: 13px;
                    color: #1f2937;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 Your API is Live!</h1>
                    <p>NBA Predictions API is now accessible</p>
                </div>
                
                <div class="content">
                    <div style="text-align: center;">
                        <span class="status-badge">✓ Online</span>
                    </div>
                    
                    <p style="font-size: 16px; margin-bottom: 24px;">
                        Your NBA Predictions API is now running and accessible from anywhere via ngrok.
                    </p>
                    
                    <div class="url-box">
                        <div class="label">🚀 Quick Action</div>
                        <a href="{public_url}/api/v1/predictions/generate-now" class="btn" style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); box-shadow: 0 4px 6px rgba(245, 158, 11, 0.4); font-size: 18px; margin: 10px 0;">
                            🏀 Generate Today's Predictions
                        </a>
                        <p style="font-size: 12px; color: #6b7280; margin: 10px 0;">Click to run the daily betting recommendations pipeline</p>
                    </div>
                    
                    <div class="url-box" style="border: 1px solid #d1d5db; margin-top: 20px;">
                        <div class="label">API Documentation URL</div>
                        <div class="url">{docs_url}</div>
                        <a href="{docs_url}" class="btn" style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);">Open API Docs</a>
                    </div>
                    
                    <div class="info-section">
                        <h3>📚 Quick Start</h3>
                        <ul>
                            <li><strong>🏀 One-Click Predictions:</strong> Just click the orange button above!</li>
                            <li><strong>Interactive Docs:</strong> Visit the docs URL to try all endpoints</li>
                            <li><strong>Health Check:</strong> Use <code>GET /api/v1/health</code></li>
                            <li><strong>Get Predictions:</strong> Use <code>GET /api/v1/predictions/{{date}}</code></li>
                        </ul>
                    </div>
                    
                    <div class="info-section">
                        <h3>🔥 Alternative: Command Line</h3>
                        <p style="margin: 8px 0;">Or use these simple curl commands:</p>
                        <pre style="background-color: #1f2937; color: #10b981; padding: 16px; border-radius: 6px; overflow-x: auto; font-size: 12px;"># Generate predictions (simple)
curl {public_url}/api/v1/predictions/generate-now

# Or with full control (POST)
curl -X POST {public_url}/api/v1/predictions/generate \\
     -H "Content-Type: application/json" \\
     -d '{{"send_email": true}}'</pre>
                    </div>
                    
                    <div style="margin-top: 24px; padding: 16px; background-color: #fef3c7; border-left: 4px solid #f59e0b; border-radius: 4px;">
                        <strong style="color: #92400e;">⚠️ Important:</strong>
                        <p style="margin: 8px 0 0 0; color: #78350f;">
                            This ngrok URL is temporary. It will change when you restart the server.
                            Save this email or bookmark the URL!
                        </p>
                    </div>
                </div>
                
                <div class="footer">
                    <p>NBA Predictions API</p>
                    <p>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    # Use the same send_email function as daily_betting_recommendations_champion.py
    subject = "🚀 Your NBA Predictions API is Live!"
    return send_email(api_key, from_email, recipient_email, subject, html_content, from_alias)


def print_info(public_url: str, port: int):
    """Print usage information."""
    print("\n" + "=" * 80)
    print("🚀 NBA PREDICTIONS API - NOW LIVE!")
    print("=" * 80)
    print(f"\n📡 Public URL (accessible from anywhere):")
    print(f"   {public_url}")
    print(f"\n🏠 Local URL (only accessible from this machine):")
    print(f"   http://localhost:{port}")
    print(f"\n📊 ngrok Dashboard:")
    print(f"   http://localhost:4040")
    print(f"\n📚 API Documentation:")
    print(f"   {public_url}/docs")
    print(f"   {public_url}/redoc")
    print(f"\n🔥 Example API Calls:")
    print(f"\n   # Health check")
    print(f"   curl {public_url}/api/v1/health")
    print(f"\n   # Generate predictions for today")
    print(f'   curl -X POST {public_url}/api/v1/predictions/generate \\')
    print(f'        -H "Content-Type: application/json" \\')
    print(f'        -d \'{{"send_email": false}}\'')
    print(f"\n   # Check task status")
    print(f"   curl {public_url}/api/v1/tasks/TASK_ID")
    print(f"\n   # Get predictions for a specific date")
    print(f"   curl {public_url}/api/v1/predictions/2025-11-18")
    print("\n" + "=" * 80)
    print("\n⚡ Press Ctrl+C to stop the server\n")


def tail_uvicorn_logs():
    """Tail uvicorn logs in a separate thread."""
    global uvicorn_process
    
    if not uvicorn_process or not uvicorn_process.stdout:
        return
    
    try:
        for line in uvicorn_process.stdout:
            if line.strip():
                print(f"[API] {line.rstrip()}")
    except:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Start NBA Predictions API with ngrok tunneling",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for FastAPI server (default: 8000)"
    )
    parser.add_argument(
        "--ngrok-token",
        type=str,
        default=None,
        help="ngrok auth token (optional, for custom domains and longer sessions)"
    )
    
    args = parser.parse_args()
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start FastAPI server
        if not start_api_server(args.port):
            logger.error("Failed to start FastAPI server")
            cleanup()
            sys.exit(1)
        
        # Start ngrok tunnel
        if not start_ngrok_tunnel(args.port, args.ngrok_token):
            logger.error("Failed to start ngrok tunnel")
            cleanup()
            sys.exit(1)
        
        # Get and display public URL
        public_url = get_ngrok_url()
        if public_url:
            print_info(public_url, args.port)
            
            # Send email with ngrok URL
            logger.info("Sending startup email...")
            send_ngrok_email(public_url, "michael.j.ganz@gmail.com")
        else:
            logger.warning("Could not retrieve ngrok public URL")
            logger.info(f"Check the ngrok dashboard at http://localhost:4040")
        
        # Start log tailing in background
        log_thread = Thread(target=tail_uvicorn_logs, daemon=True)
        log_thread.start()
        
        # Keep the script running
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if uvicorn_process and uvicorn_process.poll() is not None:
                logger.error("FastAPI server stopped unexpectedly")
                cleanup()
                sys.exit(1)
            
            if ngrok_process and ngrok_process.poll() is not None:
                logger.error("ngrok tunnel stopped unexpectedly")
                cleanup()
                sys.exit(1)
    
    except Exception as e:
        logger.error(f"Error: {e}")
        cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()

