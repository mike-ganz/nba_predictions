#!/usr/bin/env python3
"""
Thread Monitoring Demo for NBA Simulation Orchestrator

This script demonstrates the new thread monitoring and control capabilities.
"""

import subprocess
import time
import sys
from pathlib import Path

def run_demo():
    """Run a demonstration of thread monitoring capabilities."""
    print("🏀 NBA Simulation Orchestrator - Thread Monitoring Demo")
    print("=" * 60)
    
    print("\n📋 Available Thread Control Features:")
    print("1. Real-time thread status monitoring")
    print("2. Interactive thread control interface")
    print("3. Individual thread cancellation")
    print("4. Bulk thread cancellation")
    print("5. Graceful shutdown handling")
    
    print("\n🎮 Demo Commands:")
    print()
    
    # Basic multithreaded run
    print("💡 Basic multithreaded run:")
    print("   python enhanced_orchestrator.py --games \"22300803\" --runs-per-game 3 --threads 3")
    print()
    
    # Interactive monitoring
    print("💡 Interactive thread monitoring:")
    print("   python enhanced_orchestrator.py --games \"22300803,22300804\" --runs-per-game 2 --threads 4 --monitor-threads")
    print()
    
    print("📋 Interactive Commands (when using --monitor-threads):")
    print("   status, s          - Show active threads")
    print("   cancel <id>, c <id> - Cancel specific thread")
    print("   cancel-all, ca     - Cancel all threads")
    print("   help, h            - Show command help")
    print("   quit, q            - Exit monitoring")
    print()
    
    print("🔥 Advanced Features:")
    print("   • Thread-safe SQLite database operations")
    print("   • Real-time progress tracking across threads")
    print("   • Graceful shutdown with Ctrl+C")
    print("   • Background status monitoring every 30 seconds")
    print("   • Thread cancellation before and during execution")
    print()
    
    # Ask user if they want to run a demo
    response = input("🚀 Would you like to run a demo? (y/n): ").strip().lower()
    if response in ['y', 'yes']:
        run_interactive_demo()
    else:
        print("Demo cancelled. You can manually run the commands above!")

def run_interactive_demo():
    """Run an interactive demo."""
    print("\n🎯 Starting Interactive Demo...")
    print("This will run 2 games with 2 simulations each using 3 threads.")
    print("You'll have interactive control to monitor and cancel threads.")
    print()
    
    # Construct the command
    cmd = [
        sys.executable, "enhanced_orchestrator.py",
        "--games", "22300803,22300804",
        "--runs-per-game", "2", 
        "--threads", "3",
        "--monitor-threads"
    ]
    
    print(f"🔧 Running command: {' '.join(cmd)}")
    print()
    print("💡 Try these commands during execution:")
    print("   • Type 'status' to see active threads")
    print("   • Type 'cancel <thread_id>' to cancel a specific thread")
    print("   • Type 'cancel-all' to cancel all threads")
    print("   • Type 'quit' to exit monitoring")
    print()
    
    input("Press Enter to start the demo...")
    
    try:
        # Run the enhanced orchestrator with monitoring
        subprocess.run(cmd, check=True)
        print("✅ Demo completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Demo failed with error: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")

if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n👋 Demo cancelled by user")
        sys.exit(0)
