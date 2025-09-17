import os

# ---- EDIT THESE VARIABLES DIRECTLY ----
file1_path = r"C:\Users\micha\nba_predictions\data\training\nba_2023_2024_gemini_compact_first_N_plays.jsonl"     # Path to the first .jsonl file
file2_path = r"C:\Users\micha\nba_predictions\data\training\nba_2022_2023_gemini_compact_first_N_plays.jsonl"     # Path to the second .jsonl file
# file3_path = r"C:\Users\micha\nba_predictions\data\training\gemini_remainingplays_2324_part_003.jsonl"  # Optionally set a third file path, or leave as None if not used
output_dir = r"C:\Users\micha\nba_predictions\data\training"           # Directory to save the appended file
# ---------------------------------------

def append_jsonl_files(file_paths, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Compose output file path based on input file names
    bases = [os.path.splitext(os.path.basename(p))[0] for p in file_paths]
    joined_parts = "_".join([b.split("_")[-1] for b in bases if b])  # e.g., 001_002_003
    output_file = os.path.join(output_dir, f"gemini_remainingplays_2324_part_{joined_parts}.jsonl")

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for path in file_paths:
            with open(path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)
    print(f"Appended files saved to: {output_file}")

def main():
    # Collect file paths, skipping any that are None or empty
    # file_paths = [p for p in [file1_path, file2_path, file3_path] if p]
    file_paths = [p for p in [file1_path, file2_path] if p]
    if len(file_paths) < 2:
        print("Error: Please specify at least two file paths to append.")
        return

    # Check if files exist
    for path in file_paths:
        if not os.path.isfile(path):
            print(f"Error: {path} does not exist or is not a file.")
            return

    append_jsonl_files(file_paths, output_dir)

if __name__ == "__main__":
    main()
