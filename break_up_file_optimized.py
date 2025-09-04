#!/usr/bin/env python3
"""
Memory-Efficient JSONL File Splitter with Shuffling Support

This optimized version can handle very large JSONL files (multi-GB) by using
streaming processing and reservoir sampling for shuffling, avoiding memory issues.

Key Features:
- Memory-efficient streaming processing for large files
- Reservoir sampling for shuffling without loading entire file into memory  
- Automatic conversion from Python dict format to proper JSON format
- Progress tracking for large file operations
- Direct upload to OpenAI storage (optional)

Usage:
    python break_up_file_optimized.py <jsonl_file_path> <number_of_parts> [--shuffle]

Examples:
    python break_up_file_optimized.py data.jsonl 25 --shuffle
    python break_up_file_optimized.py large_file.jsonl 10 --no-convert
"""

import argparse
import ast
import json
import os
import random
import sys
from pathlib import Path
import math
from typing import List, Optional, Iterator, TextIO
import tempfile
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def convert_python_dict_to_json(line: str) -> str:
    """
    Convert a Python dictionary string to proper JSON format.
    Same as original but optimized for single-line processing.
    """
    line = line.strip()
    if not line:
        return line
    
    try:
        # Try to parse as JSON first
        json.loads(line)
        return line
    except json.JSONDecodeError:
        pass
    
    # Try Python dict format
    try:
        python_obj = ast.literal_eval(line)
        return json.dumps(python_obj, ensure_ascii=True, separators=(',', ':'))
    except (ValueError, SyntaxError):
        pass
    
    # Handle malformed JSON (same logic as original)
    if '"content":"' in line and '{' in line:
        try:
            result = ""
            i = 0
            while i < len(line):
                content_start = line.find('"content":"', i)
                if content_start == -1:
                    result += line[i:]
                    break
                
                result += line[i:content_start + 11]
                json_start = content_start + 11
                
                if json_start < len(line) and line[json_start] == '{':
                    brace_count = 0
                    json_end = json_start
                    while json_end < len(line):
                        char = line[json_end]
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end += 1
                                break
                        json_end += 1
                    
                    json_content = line[json_start:json_end]
                    escaped_content = json_content.replace('\\', '\\\\').replace('"', '\\"')
                    result += escaped_content
                    i = json_end
                else:
                    result += line[json_start]
                    i = json_start + 1
            
            parsed = json.loads(result)
            return json.dumps(parsed, ensure_ascii=True, separators=(',', ':'))
        except:
            pass
    
    raise ValueError("Cannot parse line as Python dictionary or valid JSON")


def count_lines_efficient(file_path: str) -> int:
    """Count lines in file efficiently without loading into memory."""
    print("Counting lines in file...")
    line_count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for _ in f:
            line_count += 1
            if line_count % 100000 == 0:  # Progress indicator
                print(f"  Lines counted: {line_count:,}")
    return line_count


def reservoir_sampling_split(file_path: str, num_parts: int, convert_format: bool = True) -> List[str]:
    """
    Use reservoir sampling to shuffle and split large files without loading everything into memory.
    """
    input_path = Path(file_path)
    total_lines = count_lines_efficient(file_path)
    lines_per_part = math.ceil(total_lines / num_parts)
    
    print(f"Total lines: {total_lines:,}")
    print(f"Lines per part: {lines_per_part:,}")
    print("Using reservoir sampling for shuffled distribution...")
    
    # Initialize output files
    base_name = input_path.stem
    output_dir = input_path.parent
    output_files = []
    file_handles = []
    
    # Create all output files
    for part_num in range(num_parts):
        output_filename = f"{base_name}_part_{part_num + 1:03d}.jsonl"
        output_path = output_dir / output_filename
        output_files.append(str(output_path))
        file_handles.append(open(output_path, 'w', encoding='utf-8'))
    
    try:
        # Reservoir sampling: each line has equal probability of going to any part
        with open(file_path, 'r', encoding='utf-8') as input_file:
            lines_processed = 0
            conversion_errors = 0
            
            for line in input_file:
                lines_processed += 1
                
                # Progress indicator
                if lines_processed % 50000 == 0:
                    print(f"  Processed: {lines_processed:,} / {total_lines:,} lines ({100 * lines_processed / total_lines:.1f}%)")
                
                # Convert format if requested
                if convert_format:
                    try:
                        line = convert_python_dict_to_json(line.strip())
                        if line:
                            line += '\n'
                    except ValueError:
                        conversion_errors += 1
                        continue
                
                # Randomly assign to one of the output files
                part_index = random.randrange(num_parts)
                file_handles[part_index].write(line)
        
        print(f"Processing complete! Processed {lines_processed:,} lines")
        if conversion_errors > 0:
            print(f"Warning: Skipped {conversion_errors} lines due to conversion errors")
    
    finally:
        # Close all file handles
        for fh in file_handles:
            fh.close()
    
    # Report final file sizes
    print("\nCreated files:")
    for i, output_file in enumerate(output_files):
        lines_written = count_lines_efficient(output_file)
        print(f"  Part {i+1:03d}: {Path(output_file).name} - {lines_written:,} lines")
    
    return output_files


def sequential_split(file_path: str, num_parts: int, convert_format: bool = True) -> List[str]:
    """
    Split file sequentially using streaming processing (memory efficient).
    """
    input_path = Path(file_path)
    total_lines = count_lines_efficient(file_path)
    lines_per_part = math.ceil(total_lines / num_parts)
    
    print(f"Total lines: {total_lines:,}")
    print(f"Lines per part: {lines_per_part:,}")
    print("Processing in sequential order...")
    
    base_name = input_path.stem
    output_dir = input_path.parent
    created_files = []
    
    current_part = 0
    current_file = None
    lines_in_current_part = 0
    lines_processed = 0
    conversion_errors = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as input_file:
            for line in input_file:
                lines_processed += 1
                
                # Progress indicator
                if lines_processed % 50000 == 0:
                    print(f"  Processed: {lines_processed:,} / {total_lines:,} lines ({100 * lines_processed / total_lines:.1f}%)")
                
                # Convert format if requested
                if convert_format:
                    try:
                        line = convert_python_dict_to_json(line.strip())
                        if line:
                            line += '\n'
                        else:
                            continue
                    except ValueError:
                        conversion_errors += 1
                        continue
                
                # Check if we need to start a new part
                if current_file is None or lines_in_current_part >= lines_per_part:
                    # Close previous file if exists
                    if current_file:
                        current_file.close()
                        print(f"Completed part {current_part}: {lines_in_current_part:,} lines")
                    
                    # Start new part
                    current_part += 1
                    if current_part > num_parts:
                        break  # Safety check
                    
                    output_filename = f"{base_name}_part_{current_part:03d}.jsonl"
                    output_path = output_dir / output_filename
                    created_files.append(str(output_path))
                    current_file = open(output_path, 'w', encoding='utf-8')
                    lines_in_current_part = 0
                
                # Write line to current file
                current_file.write(line)
                lines_in_current_part += 1
    
    finally:
        # Close final file
        if current_file:
            current_file.close()
            print(f"Completed part {current_part}: {lines_in_current_part:,} lines")
    
    print(f"Processing complete! Processed {lines_processed:,} lines")
    if conversion_errors > 0:
        print(f"Warning: Skipped {conversion_errors} lines due to conversion errors")
    
    return created_files


def upload_file_to_openai(file_path: str, api_key: str, purpose: str = "fine-tune") -> Optional[str]:
    """Upload a file to OpenAI storage (same as original)."""
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI library not installed. Install with: pip install openai")
    
    try:
        client = OpenAI(api_key=api_key)
        print(f"Uploading {Path(file_path).name} to OpenAI...")
        
        with open(file_path, 'rb') as f:
            response = client.files.create(file=f, purpose=purpose)
        
        file_id = response.id
        print(f"Successfully uploaded {Path(file_path).name} - File ID: {file_id}")
        return file_id
        
    except Exception as e:
        print(f"Failed to upload {Path(file_path).name}: {e}")
        return None


def split_large_jsonl_file(file_path: str, num_parts: int, convert_format: bool = True, 
                          shuffle: bool = False, upload: bool = False, 
                          api_key: Optional[str] = None, purpose: str = "fine-tune") -> tuple[List[str], List[str]]:
    """
    Main function to split large JSONL files efficiently.
    """
    # Validate input
    input_path = Path(file_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    if not input_path.suffix.lower() == '.jsonl':
        raise ValueError(f"Input file must have .jsonl extension: {file_path}")
    
    if num_parts <= 0:
        raise ValueError(f"Number of parts must be positive: {num_parts}")
    
    print(f"Processing file: {file_path}")
    file_size = input_path.stat().st_size
    print(f"File size: {file_size / (1024**3):.2f} GB")
    
    # Choose splitting method based on shuffle preference
    if shuffle:
        created_files = reservoir_sampling_split(file_path, num_parts, convert_format)
    else:
        created_files = sequential_split(file_path, num_parts, convert_format)
    
    uploaded_file_ids = []
    
    # Upload if requested
    if upload:
        if not api_key:
            raise ValueError("API key is required for uploading files to OpenAI")
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")
        
        print("\nUploading files to OpenAI...")
        for file_path in created_files:
            file_id = upload_file_to_openai(file_path, api_key, purpose)
            if file_id:
                uploaded_file_ids.append(file_id)
        
        print(f"Uploaded {len(uploaded_file_ids)} out of {len(created_files)} files successfully")
    
    return created_files, uploaded_file_ids


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Memory-efficient JSONL file splitter for large files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Sequential split (memory efficient)
    python break_up_file_optimized.py large_file.jsonl 25
    
    # Shuffled split using reservoir sampling
    python break_up_file_optimized.py large_file.jsonl 25 --shuffle
    
    # Skip format conversion (if already proper JSON)
    python break_up_file_optimized.py large_file.jsonl 25 --no-convert
    
    # Split and upload to OpenAI
    python break_up_file_optimized.py large_file.jsonl 25 --upload --api-key YOUR_KEY
        """
    )
    
    parser.add_argument('file_path', help='Path to the JSONL file to split')
    parser.add_argument('num_parts', type=int, help='Number of parts to split into')
    parser.add_argument('--shuffle', action='store_true', help='Randomly shuffle lines before splitting')
    parser.add_argument('--no-convert', action='store_true', help='Skip Python dict to JSON conversion')
    parser.add_argument('--upload', action='store_true', help='Upload split files to OpenAI')
    parser.add_argument('--api-key', help='OpenAI API key')
    parser.add_argument('--purpose', default='fine-tune', choices=['fine-tune', 'batch', 'assistants'],
                       help='Purpose for OpenAI upload')
    
    try:
        args = parser.parse_args()
    except SystemExit:
        return
    
    try:
        # Get API key
        api_key = args.api_key or os.getenv('OPENAI_API_KEY')
        
        # Validate upload requirements
        if args.upload and not api_key:
            print("Error: API key required for upload. Use --api-key or OPENAI_API_KEY env var", 
                  file=sys.stderr)
            sys.exit(1)
        
        # Split the file
        created_files, uploaded_file_ids = split_large_jsonl_file(
            args.file_path, 
            args.num_parts, 
            convert_format=not args.no_convert,
            shuffle=args.shuffle,
            upload=args.upload,
            api_key=api_key,
            purpose=args.purpose
        )
        
        # Print results
        print(f"\n✅ Successfully split {args.file_path} into {len(created_files)} parts")
        
        if args.upload and uploaded_file_ids:
            print(f"\nOpenAI File IDs:")
            for i, file_id in enumerate(uploaded_file_ids, 1):
                print(f"  Part {i:03d}: {file_id}")
        
    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
