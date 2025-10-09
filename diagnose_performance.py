#!/usr/bin/env python3
"""
Performance diagnostic tool for NBA prediction orchestrator.
Run this WHILE your orchestrator is running to identify bottlenecks.
"""

import psutil
import sqlite3
import time
import os
import sys
from datetime import datetime
from pathlib import Path

def format_bytes(bytes_val):
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"

def check_system_resources():
    """Check CPU, RAM, and disk usage."""
    print("=" * 70)
    print("🖥️  SYSTEM RESOURCES")
    print("=" * 70)
    
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    
    print(f"\n💻 CPU:")
    print(f"   Usage: {cpu_percent:.1f}%")
    print(f"   Cores: {cpu_count} ({psutil.cpu_count(logical=False)} physical)")
    if cpu_freq:
        print(f"   Frequency: {cpu_freq.current:.0f} MHz")
    
    # CPU per core
    cpu_per_core = psutil.cpu_percent(interval=0.5, percpu=True)
    if cpu_per_core:
        max_core = max(cpu_per_core)
        avg_core = sum(cpu_per_core) / len(cpu_per_core)
        print(f"   Per-core avg: {avg_core:.1f}% | Max core: {max_core:.1f}%")
        
        if cpu_percent < 20:
            print(f"   ✅ CPU is MOSTLY IDLE - not the bottleneck!")
        elif cpu_percent < 50:
            print(f"   ✅ CPU usage is normal for I/O-bound workload")
        else:
            print(f"   ⚠️ CPU usage is higher than expected")
    
    # RAM
    memory = psutil.virtual_memory()
    print(f"\n💾 RAM:")
    print(f"   Total: {format_bytes(memory.total)}")
    print(f"   Used: {format_bytes(memory.used)} ({memory.percent:.1f}%)")
    print(f"   Available: {format_bytes(memory.available)}")
    
    if memory.percent > 90:
        print(f"   ⚠️ RAM usage is VERY HIGH - may cause swapping!")
    elif memory.percent > 75:
        print(f"   ⚠️ RAM usage is high")
    else:
        print(f"   ✅ RAM usage is fine")
    
    # Disk I/O
    disk = psutil.disk_usage('.')
    disk_io = psutil.disk_io_counters()
    
    print(f"\n💿 Disk:")
    print(f"   Total: {format_bytes(disk.total)}")
    print(f"   Used: {format_bytes(disk.used)} ({disk.percent:.1f}%)")
    print(f"   Free: {format_bytes(disk.free)}")
    
    if disk_io:
        print(f"   Read: {format_bytes(disk_io.read_bytes)}")
        print(f"   Write: {format_bytes(disk_io.write_bytes)}")
        
        if disk.percent > 95:
            print(f"   ⚠️ Disk almost FULL - may slow down SQLite!")
        else:
            print(f"   ✅ Disk space is fine")

def check_python_processes():
    """Check running Python processes and their resource usage."""
    print("\n" + "=" * 70)
    print("🐍 PYTHON PROCESSES")
    print("=" * 70)
    
    python_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'num_threads']):
        try:
            if 'python' in proc.info['name'].lower():
                python_procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if not python_procs:
        print("   ⚠️ No Python processes found (orchestrator not running?)")
        return
    
    print(f"\n   Found {len(python_procs)} Python process(es):\n")
    print(f"   {'PID':<8} {'CPU%':<8} {'RAM':<12} {'Threads':<10} {'Name'}")
    print(f"   {'-'*60}")
    
    for proc in python_procs:
        try:
            cpu = proc.cpu_percent(interval=0.1)
            mem = proc.memory_info().rss
            threads = proc.num_threads()
            name = proc.name()
            pid = proc.pid
            
            print(f"   {pid:<8} {cpu:<8.1f} {format_bytes(mem):<12} {threads:<10} {name}")
            
            # Analyze
            if threads > 50:
                print(f"      ⚠️ High thread count - may indicate thread leak")
            
            if mem > 1024 * 1024 * 1024:  # > 1GB
                print(f"      ⚠️ High memory usage")
                
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

def check_database_performance(db_path='enhanced_simulation_results.db'):
    """Check database performance and statistics."""
    print("\n" + "=" * 70)
    print("💾 DATABASE PERFORMANCE")
    print("=" * 70)
    
    # Check if database exists
    db_files = list(Path('.').glob('*.db'))
    if not db_files:
        print("   ⚠️ No database files found in current directory")
        return
    
    print(f"\n   Database files found:")
    for db_file in db_files:
        size = os.path.getsize(db_file)
        print(f"   • {db_file.name}: {format_bytes(size)}")
    
    # Try to connect to the specified or most recent database
    if not os.path.exists(db_path) and db_files:
        db_path = str(db_files[0])
    
    if not os.path.exists(db_path):
        print(f"   ⚠️ Database {db_path} not found")
        return
    
    print(f"\n   Analyzing: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        cursor = conn.cursor()
        
        # Check record count
        cursor.execute("SELECT COUNT(*) FROM simulation_results")
        total_records = cursor.fetchone()[0]
        print(f"   Total records: {total_records}")
        
        # Check recent records (last 5 minutes)
        five_min_ago = datetime.now().timestamp() - 300
        cursor.execute("""
            SELECT COUNT(*) FROM simulation_results 
            WHERE start_time >= ?
        """, (datetime.fromtimestamp(five_min_ago).isoformat(),))
        recent_records = cursor.fetchone()[0]
        
        if recent_records > 0:
            print(f"   Records in last 5 min: {recent_records} ({recent_records/5:.1f} per minute)")
        
        # Check status distribution
        cursor.execute("""
            SELECT status, COUNT(*) 
            FROM simulation_results 
            GROUP BY status
        """)
        status_counts = cursor.fetchall()
        
        print(f"\n   Status Distribution:")
        for status, count in status_counts:
            pct = (count / total_records * 100) if total_records > 0 else 0
            print(f"   • {status}: {count} ({pct:.1f}%)")
            
            if status == 'error' and pct > 10:
                print(f"      ⚠️ High error rate!")
            elif status == 'validation_failure' and pct > 5:
                print(f"      ⚠️ Many validation failures - slowing down predictions!")
        
        # Check average duration
        cursor.execute("""
            SELECT AVG(duration_seconds), MIN(duration_seconds), MAX(duration_seconds)
            FROM simulation_results
            WHERE status IN ('completed', 'game_ended')
            AND duration_seconds > 0
        """)
        duration_stats = cursor.fetchone()
        
        if duration_stats and duration_stats[0]:
            avg_dur, min_dur, max_dur = duration_stats
            print(f"\n   Duration Statistics (completed games):")
            print(f"   • Average: {avg_dur:.1f}s ({avg_dur/60:.1f} min)")
            print(f"   • Min: {min_dur:.1f}s")
            print(f"   • Max: {max_dur:.1f}s ({max_dur/60:.1f} min)")
            
            if avg_dur > 300:  # > 5 minutes
                print(f"      ⚠️ Games taking VERY LONG (avg {avg_dur/60:.1f} min)")
                print(f"      This may indicate:")
                print(f"      • Slow API responses from Gemini")
                print(f"      • Many validation retries")
                print(f"      • Network latency issues")
        
        # Check for validation failures
        cursor.execute("""
            SELECT 
                validation_most_common_reason,
                validation_most_common_reason_count,
                COUNT(*) as games_with_failures
            FROM simulation_results
            WHERE validation_most_common_reason IS NOT NULL
            GROUP BY validation_most_common_reason
            ORDER BY games_with_failures DESC
            LIMIT 5
        """)
        validation_issues = cursor.fetchall()
        
        if validation_issues:
            print(f"\n   ⚠️ Common Validation Failures (causing retries/slowdown):")
            for reason, count, games in validation_issues:
                if reason:
                    print(f"   • {reason[:60]}: {games} games affected")
        
        # Check WAL mode
        cursor.execute("PRAGMA journal_mode")
        journal_mode = cursor.fetchone()[0]
        print(f"\n   Database Configuration:")
        print(f"   • Journal mode: {journal_mode}")
        
        if journal_mode != 'wal':
            print(f"      ⚠️ Not using WAL mode - may slow down concurrent writes!")
        else:
            print(f"      ✅ Using WAL mode - good for concurrency")
        
        conn.close()
        
    except sqlite3.OperationalError as e:
        print(f"   ⚠️ Database is locked (orchestrator actively writing)")
        print(f"   This is normal during execution")
    except Exception as e:
        print(f"   ❌ Error analyzing database: {e}")

def check_network_latency():
    """Check network latency to Google APIs."""
    print("\n" + "=" * 70)
    print("🌐 NETWORK LATENCY")
    print("=" * 70)
    
    try:
        import socket
        
        # Test DNS resolution time
        start = time.time()
        socket.gethostbyname('generativelanguage.googleapis.com')
        dns_time = (time.time() - start) * 1000
        
        print(f"\n   DNS resolution: {dns_time:.1f}ms")
        
        if dns_time > 100:
            print(f"      ⚠️ DNS resolution is slow")
        else:
            print(f"      ✅ DNS resolution is fast")
        
        # Test connection time
        start = time.time()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('generativelanguage.googleapis.com', 443))
        connect_time = (time.time() - start) * 1000
        sock.close()
        
        print(f"   Connection to Gemini API: {connect_time:.1f}ms")
        
        if result != 0:
            print(f"      ❌ Cannot connect to Gemini API!")
        elif connect_time > 500:
            print(f"      ⚠️ Connection is SLOW - this will bottleneck API calls!")
        elif connect_time > 200:
            print(f"      ⚠️ Connection is somewhat slow")
        else:
            print(f"      ✅ Connection is fast")
            
    except Exception as e:
        print(f"   ⚠️ Network check failed: {e}")

def estimate_completion_time(db_path='enhanced_simulation_results.db'):
    """Estimate time to completion based on current progress."""
    print("\n" + "=" * 70)
    print("⏱️  COMPLETION ESTIMATE")
    print("=" * 70)
    
    if not os.path.exists(db_path):
        db_files = list(Path('.').glob('*.db'))
        if db_files:
            db_path = str(db_files[0])
        else:
            print("   ⚠️ No database found")
            return
    
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        cursor = conn.cursor()
        
        # Get first and last record times
        cursor.execute("""
            SELECT 
                MIN(start_time) as first_start,
                MAX(end_time) as last_end,
                COUNT(*) as total,
                AVG(duration_seconds) as avg_duration
            FROM simulation_results
            WHERE status IN ('completed', 'game_ended')
        """)
        
        result = cursor.fetchone()
        if result and result[0]:
            first_start, last_end, completed, avg_duration = result
            
            # Parse timestamps
            try:
                first_dt = datetime.fromisoformat(first_start)
                last_dt = datetime.fromisoformat(last_end)
                elapsed = (last_dt - first_dt).total_seconds()
                
                if elapsed > 0 and completed > 0:
                    rate = completed / (elapsed / 60)  # games per minute
                    
                    print(f"\n   Completed: {completed} games")
                    print(f"   Elapsed time: {elapsed/60:.1f} minutes")
                    print(f"   Rate: {rate:.2f} games/minute ({rate*60:.1f} games/hour)")
                    print(f"   Avg duration per game: {avg_duration:.1f}s")
                    
                    # Estimate based on typical full season
                    for target in [10, 50, 100, 500, 1000]:
                        if completed < target:
                            remaining = target - completed
                            est_minutes = remaining / rate if rate > 0 else 0
                            print(f"   • {target} games: ~{est_minutes:.0f} min ({est_minutes/60:.1f} hours) remaining")
                            break
                    
                    # Performance assessment
                    if rate < 2:
                        print(f"\n   ⚠️ SLOW PERFORMANCE ({rate:.2f} games/min)")
                        print(f"   Expected for 8 threads: 8-12 games/min")
                        print(f"   Current is {(8/rate):.1f}x SLOWER than expected!")
                    elif rate < 5:
                        print(f"\n   ⚠️ Below expected performance")
                        print(f"   Expected: 8-12 games/min | Actual: {rate:.2f} games/min")
                    else:
                        print(f"\n   ✅ Good performance ({rate:.2f} games/min)")
                        
            except Exception as e:
                print(f"   Could not parse timestamps: {e}")
        
        conn.close()
        
    except Exception as e:
        print(f"   ⚠️ Could not estimate: {e}")

def print_recommendations():
    """Print performance recommendations."""
    print("\n" + "=" * 70)
    print("💡 RECOMMENDATIONS")
    print("=" * 70)
    print("""
Based on the diagnostics above, common bottlenecks and fixes:

1️⃣ SLOW API RESPONSES (Most Common):
   • Symptom: Low CPU/RAM, games taking 5+ minutes
   • Cause: Gemini API is slow or rate limiting
   • Fix: Just wait - this is normal sometimes
   • Alternative: Try different time of day, or increase threads slightly

2️⃣ VALIDATION FAILURES:
   • Symptom: High error/validation_failure count in database
   • Cause: Model producing invalid outputs → retry loops
   • Fix: Check database for common validation reasons
   • Command: python -c "import sqlite3; conn=sqlite3.connect('enhanced_simulation_results.db'); 
              cursor=conn.execute('SELECT validation_most_common_reason FROM simulation_results 
              WHERE validation_most_common_reason IS NOT NULL LIMIT 10'); 
              print(list(cursor))"

3️⃣ NETWORK LATENCY:
   • Symptom: High connection time (>500ms) in network check
   • Cause: Poor internet connection or routing to Google
   • Fix: Check your internet connection, try VPN, or wait

4️⃣ DATABASE CONTENTION:
   • Symptom: Frequent "database is locked" errors
   • Cause: Too many threads writing simultaneously
   • Fix: Reduce thread count to 4-6

5️⃣ INSUFFICIENT RAM/CPU:
   • Symptom: CPU >80% or RAM >90%
   • Cause: Too many threads for your system
   • Fix: Reduce thread count, close other apps

📊 Expected Performance (8 threads, optimal conditions):
   • Games per minute: 8-12
   • Games per hour: 480-720
   • 100 games: 8-12 minutes
   • 1000 games: 1.5-2 hours

⚠️ If your actual performance is 2-3x slower, the issue is likely:
   • Gemini API being slow (nothing you can do)
   • Validation failures causing retries (check database)
   • Network latency (check internet connection)
""")

def main():
    """Run all diagnostic checks."""
    print("\n" + "=" * 70)
    print("🔍 NBA PREDICTION ORCHESTRATOR - PERFORMANCE DIAGNOSTICS")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Run this WHILE your orchestrator is running for best results")
    print("=" * 70)
    
    check_system_resources()
    check_python_processes()
    check_database_performance()
    check_network_latency()
    estimate_completion_time()
    print_recommendations()
    
    print("\n" + "=" * 70)
    print("✅ Diagnostics complete!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()

