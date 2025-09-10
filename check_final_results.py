import sqlite3

print("🏆 FINAL SIMULATION TEST RESULTS")
print("=" * 50)

# Check the database results
conn = sqlite3.connect('final_test_results.db')
cursor = conn.cursor()

cursor.execute('SELECT * FROM simulation_runs')
row = cursor.fetchone()

if row:
    print(f"✅ SIMULATION COMPLETED!")
    print(f"   Game ID: {row[1]}")
    print(f"   Season: {row[2]}")
    print(f"   Status: {row[7]}")
    print(f"   Final Score: {row[10]}")
    print(f"   Duration: {row[6]:.2f} seconds" if row[6] else "N/A")
    print(f"   Predictions Made: {row[9]}")
    print(f"   Success: {'🎉 YES!' if row[7] == 'game_ended' else '❌ NO'}")
    
    if row[10]:  # Check if we have a final score
        print(f"\n🏀 GAME DETAILS:")
        print(f"   Final Score: {row[10]}")
        print(f"   Final Quarter: {row[11] if len(row) > 11 else 'N/A'}")
        print(f"   Final Time: {row[12] if len(row) > 12 else 'N/A'}")
        
        # Check if this looks like the right teams (Washington @ Chicago)
        if row[10] and ('WAS' in str(row[10]) or 'CHI' in str(row[10]) or 
                        'Washington' in str(row[10]) or 'Chicago' in str(row[10])):
            print("   ✅ CORRECT TEAMS: Found Washington/Chicago in results!")
        else:
            print(f"   ❓ Teams unclear from score: {row[10]}")
            
else:
    print("❌ No simulation results found in database")

conn.close()

# Check the log file
print(f"\n📋 LOG FILE SUMMARY:")
print("=" * 30)

try:
    with open('final_test_log.txt', 'r') as f:
        log_content = f.read()
        
    if log_content.strip():
        lines = log_content.strip().split('\n')
        print(f"   Log lines: {len(lines)}")
        
        # Check for key indicators
        if 'WAS' in log_content or 'CHI' in log_content:
            print("   ✅ Found WAS/CHI team indicators in logs")
        if 'Washington' in log_content or 'Chicago' in log_content:
            print("   ✅ Found team names in logs")
        if 'game_ended' in log_content:
            print("   ✅ Game completed successfully")
        if 'Data loading system not available' in log_content:
            print("   ❌ Still using sample data (real data system failed)")
        else:
            print("   ✅ Real data system appears to be working")
            
        # Show last few lines
        print(f"\n   Last few log entries:")
        for line in lines[-5:]:
            if line.strip():
                print(f"     {line[:100]}...")
    else:
        print("   ⚠️ Log file is empty")
        
except Exception as e:
    print(f"   ❌ Could not read log file: {e}")

print(f"\n🎯 FINAL ASSESSMENT:")
print("=" * 30)
