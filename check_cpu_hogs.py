import psutil

print("Top 10 CPU-consuming processes:\n")
print(f"{'Process':<30} {'PID':<8} {'CPU%':<8} {'Threads':<10} {'RAM(MB)':<10}")
print("-" * 75)

# Get all processes and sort by CPU
procs = []
for proc in psutil.process_iter(['name', 'pid', 'cpu_percent', 'num_threads', 'memory_info']):
    try:
        proc.cpu_percent(interval=0.1)  # Prime the counter
    except:
        pass

# Wait a moment then get actual CPU
import time
time.sleep(1)

for proc in psutil.process_iter(['name', 'pid', 'cpu_percent', 'num_threads', 'memory_info']):
    try:
        info = proc.info
        cpu = proc.cpu_percent(interval=0.1)
        if cpu > 0.1:  # Only show processes using CPU
            mem_mb = info['memory_info'].rss / (1024 * 1024)
            procs.append((info['name'], info['pid'], cpu, info['num_threads'], mem_mb))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

# Sort by CPU descending
procs.sort(key=lambda x: x[1], reverse=True)

for name, pid, cpu, threads, mem in procs[:15]:
    print(f"{name[:28]:<30} {pid:<8} {cpu:<8.1f} {threads:<10} {mem:<10.1f}")

print("\n" + "="*75)

# Check total system CPU
cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
print(f"\nPer-core CPU usage:")
for i, pct in enumerate(cpu_percent):
    bar = "█" * int(pct / 5)
    print(f"  Core {i}: {pct:5.1f}% [{bar:<20}]")

print(f"\n  Average: {sum(cpu_percent)/len(cpu_percent):.1f}%")

