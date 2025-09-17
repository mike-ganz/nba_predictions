import sys

def get_nth_jsonl_entry(file_path, n):
    """
    Return the Nth (1-based) entry from a JSONL file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            if idx == n:
                print(line.rstrip('\n'))
                return
    print(f"Entry {n} not found in file.")

if __name__ == "__main__":
    # Usage: python this_script.py N
    file_path = r"C:\Users\micha\nba_predictions\data\training\nba_2023_2024_gemini_compact_remaining_plays_20250916_175518.jsonl"
    if len(sys.argv) != 2:
        print("Usage: python script.py N")
        sys.exit(1)
    try:
        n = int(sys.argv[1])
        if n < 1:
            raise ValueError
    except ValueError:
        print("N must be a positive integer.")
        sys.exit(1)
    get_nth_jsonl_entry(file_path, n)
