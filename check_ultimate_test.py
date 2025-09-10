import sqlite3

conn = sqlite3.connect('ultimate_test.db')
cursor = conn.cursor()

cursor.execute('SELECT * FROM simulation_runs')
row = cursor.fetchone()

print("🏆 ULTIMATE TEST RESULTS:")
print("=" * 40)
print(f"Game ID: {row[1]}")
print(f"Status: {row[7]}")
print(f"Final Score: {row[10]}")
print(f"Success: {'🎉 YES!' if row[7] == 'game_ended' else '❌ NO'}")
print(f"Duration: {row[6]:.2f} seconds")
print(f"Predictions: {row[9]}")

# Check if the score contains the right team abbreviations
if row[10]:
    score_text = str(row[10])
    has_was = 'WAS' in score_text or 'Washington' in score_text
    has_chi = 'CHI' in score_text or 'Chicago' in score_text
    
    print(f"\n🏀 TEAM VERIFICATION:")
    print(f"   Contains Washington (WAS): {'✅' if has_was else '❌'}")
    print(f"   Contains Chicago (CHI): {'✅' if has_chi else '❌'}")
    
    if has_was and has_chi:
        print("   🎉 CORRECT! Using real Washington @ Chicago data!")
    else:
        print("   ❌ Still using sample/synthetic data")

conn.close()
