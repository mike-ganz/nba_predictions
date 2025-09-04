import os

# ---- EDIT THESE VARIABLES DIRECTLY ----
file1_path = r"C:\Users\micha\nba_predictions\data\training\gemini_firstnplays_2324.jsonl"     # Path to the first .jsonl file
file2_path = r"C:\Users\micha\nba_predictions\data\training\gemini_firstnplays_2223.jsonl"     # Path to the second .jsonl file
output_dir = r"C:\Users\micha\nba_predictions\data\training"           # Directory to save the appended file
# ---------------------------------------

def append_jsonl_files(file1_path, file2_path, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Compose output file path
    base1 = os.path.splitext(os.path.basename(file1_path))[0]
    base2 = os.path.splitext(os.path.basename(file2_path))[0]
    output_file = os.path.join(output_dir, f"gemini_firstnplays_2223_2324.jsonl")

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for path in [file1_path, file2_path]:
            with open(path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)
    print(f"Appended files saved to: {output_file}")

def main():
    # Check if files exist
    if not os.path.isfile(file1_path):
        print(f"Error: {file1_path} does not exist or is not a file.")
        return
    if not os.path.isfile(file2_path):
        print(f"Error: {file2_path} does not exist or is not a file.")
        return

    append_jsonl_files(file1_path, file2_path, output_dir)

if __name__ == "__main__":
    main()
