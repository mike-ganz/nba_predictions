#!/usr/bin/env python3
"""
JSONL File Splitter with Format Conversion and OpenAI Upload

This script splits a JSONL file into a specified number of parts, automatically converts
Python dictionary format to proper JSON format, and optionally uploads them to OpenAI 
storage for use with fine-tuning, batch processing, or assistants.

Key Features:
- Automatic conversion from Python dict format {'key': 'value'} to JSON format {"key": "value"}
- Smart splitting into equal parts (sequential or random distribution)
- Direct upload to OpenAI storage
- Support for different upload purposes (fine-tune, batch, assistants)

Usage:
    # Just split the file (with automatic format conversion)
    python break_up_file.py <jsonl_file_path> <number_of_parts>
    
    # Split and upload to OpenAI
    python break_up_file.py <jsonl_file_path> <number_of_parts> --upload --api-key <your_key>

Examples:
    python break_up_file.py data.jsonl 5                    # Sequential split
    python break_up_file.py data.jsonl 5 --shuffle          # Random distribution
    python break_up_file.py data.jsonl 5 --upload --api-key sk-...
    python break_up_file.py data.jsonl 5 --upload --purpose batch --shuffle
    python break_up_file.py data.jsonl 5 --no-convert       # Skip format conversion

Requirements:
    pip install openai  # Only needed for upload functionality
"""

import argparse
import ast
import json
import os
import random
import sys
from pathlib import Path
import math
from typing import List, Optional
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def convert_python_dict_to_json(line: str) -> str:
    """
    Convert a Python dictionary string to proper JSON format.
    Handles both Python dict format and malformed JSON with unescaped quotes.
    
    Args:
        line (str): Line containing Python dictionary syntax or malformed JSON
    
    Returns:
        str: Properly formatted JSON line
    
    Raises:
        ValueError: If the line cannot be parsed
    """
    line = line.strip()
    if not line:
        return line
    
    try:
        # Try to parse as JSON first (in case it's already correct)
        json.loads(line)
        return line
    except json.JSONDecodeError:
        # JSON parsing failed - could be Python dict format or malformed JSON
        pass
    
    # Try Python dict format first
    try:
        python_obj = ast.literal_eval(line)
        json_line = json.dumps(python_obj, ensure_ascii=True, separators=(',', ':'))
        return json_line
    except (ValueError, SyntaxError):
        # Not valid Python literal - might be malformed JSON with unescaped content
        pass
    
    # Last attempt: Try to fix malformed JSON with unescaped quotes in content fields
    try:
        # Check if this looks like malformed JSON with unescaped quotes in content
        if '"content":"' in line and '{' in line:
            print(f"  Attempting to fix malformed JSON with unescaped quotes in content...")
            
            # More sophisticated approach: find content fields and properly escape them
            result = ""
            i = 0
            while i < len(line):
                # Look for "content":" pattern
                content_start = line.find('"content":"', i)
                if content_start == -1:
                    # No more content fields, append the rest
                    result += line[i:]
                    break
                
                # Append everything up to the content field
                result += line[i:content_start + 11]  # includes '"content":"'
                
                # Find the start of the JSON content (should be '{')
                json_start = content_start + 11
                if json_start < len(line) and line[json_start] == '{':
                    # Find the matching closing brace
                    brace_count = 0
                    json_end = json_start
                    while json_end < len(line):
                        char = line[json_end]
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end += 1  # include the closing brace
                                break
                        json_end += 1
                    
                    # Extract the JSON content and properly escape it
                    json_content = line[json_start:json_end]
                    escaped_content = json_content.replace('\\', '\\\\').replace('"', '\\"')
                    result += escaped_content
                    
                    i = json_end
                else:
                    # Not a JSON object, just continue
                    result += line[json_start]
                    i = json_start + 1
            
            # Try to parse the fixed JSON
            try:
                parsed = json.loads(result)
                # Re-serialize to ensure consistent formatting
                json_line = json.dumps(parsed, ensure_ascii=True, separators=(',', ':'))
                print(f"  ✅ Successfully fixed malformed JSON")
                return json_line
            except json.JSONDecodeError as e:
                print(f"  ❌ Fix attempt still invalid: {e}")
                pass
        
        raise ValueError("Cannot parse line as Python dictionary or valid JSON - line format not recognized")
        
    except Exception as e:
        raise ValueError(f"Cannot parse line: {e}")


def upload_file_to_openai(file_path: str, api_key: str, purpose: str = "fine-tune") -> Optional[str]:
    """
    Upload a file to OpenAI storage.
    
    Args:
        file_path (str): Path to the file to upload
        api_key (str): OpenAI API key
        purpose (str): Purpose of the file upload (default: "fine-tune")
    
    Returns:
        str: File ID if successful, None if failed
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI library not installed. Install with: pip install openai")
    
    try:
        client = OpenAI(api_key=api_key)
        
        print(f"Uploading {Path(file_path).name} to OpenAI...")
        
        with open(file_path, 'rb') as f:
            response = client.files.create(
                file=f,
                purpose=purpose
            )
        
        file_id = response.id
        print(f"Successfully uploaded {Path(file_path).name} - File ID: {file_id}")
        return file_id
        
    except Exception as e:
        print(f"Failed to upload {Path(file_path).name}: {e}")
        return None


def split_jsonl_file(file_path, num_parts, convert_format=True, shuffle=False):
    """
    Split a JSONL file into the specified number of parts.
    
    Args:
        file_path (str): Path to the input JSONL file
        num_parts (int): Number of parts to split the file into
        convert_format (bool): Whether to convert Python dict format to JSON format
        shuffle (bool): Whether to randomly shuffle lines before splitting
    
    Returns:
        list: List of created file paths
    """
    # Validate input file
    input_path = Path(file_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    if not input_path.suffix.lower() == '.jsonl':
        raise ValueError(f"Input file must have .jsonl extension: {file_path}")
    
    # Validate number of parts
    if num_parts <= 0:
        raise ValueError(f"Number of parts must be positive: {num_parts}")
    
    # Count total lines in the file
    print(f"Reading file: {file_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()
    
    # Convert Python dictionary format to JSON format if requested
    if convert_format:
        print("Converting Python dictionary format to JSON format...")
        lines = []
        conversion_errors = 0
        
        for i, line in enumerate(raw_lines, 1):
            try:
                converted_line = convert_python_dict_to_json(line.strip())
                lines.append(converted_line + '\n' if converted_line else '\n')
            except ValueError as e:
                print(f"Warning: Skipping line {i} due to conversion error: {e}")
                conversion_errors += 1
                continue
        
        if conversion_errors > 0:
            print(f"Warning: Skipped {conversion_errors} lines due to conversion errors")
    else:
        print("Skipping format conversion (using original format)")
        lines = raw_lines
    
    total_lines = len(lines)
    print(f"Total lines: {total_lines}")
    
    if total_lines == 0:
        raise ValueError("Input file is empty")
    
    # Shuffle lines if requested
    if shuffle:
        print("Shuffling lines randomly...")
        random.shuffle(lines)
    else:
        print("Keeping lines in original order (sequential split)")
    
    if num_parts > total_lines:
        print(f"Warning: Number of parts ({num_parts}) is greater than number of lines ({total_lines})")
        print(f"Will create {total_lines} files instead")
        num_parts = total_lines
    
    # Calculate lines per part
    lines_per_part = math.ceil(total_lines / num_parts)
    print(f"Lines per part: {lines_per_part}")
    
    # Generate output file paths
    base_name = input_path.stem  # filename without extension
    output_dir = input_path.parent
    created_files = []
    
    # Split the file
    for part_num in range(num_parts):
        start_idx = part_num * lines_per_part
        end_idx = min(start_idx + lines_per_part, total_lines)
        
        # Skip if no lines for this part (shouldn't happen with our logic, but safety check)
        if start_idx >= total_lines:
            break
        
        # Create output filename
        output_filename = f"{base_name}_part_{part_num + 1:03d}.jsonl"
        output_path = output_dir / output_filename
        
        # Write the part
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(lines[start_idx:end_idx])
        
        lines_written = end_idx - start_idx
        print(f"Created {output_filename} with {lines_written} lines")
        created_files.append(str(output_path))
    
    return created_files


def split_and_upload_jsonl(file_path: str, num_parts: int, upload: bool = False, 
                          api_key: Optional[str] = None, purpose: str = "fine-tune",
                          convert_format: bool = True, shuffle: bool = False) -> tuple[List[str], List[str]]:
    """
    Split a JSONL file and optionally upload the parts to OpenAI.
    
    Args:
        file_path (str): Path to the input JSONL file
        num_parts (int): Number of parts to split the file into
        upload (bool): Whether to upload files to OpenAI
        api_key (str, optional): OpenAI API key (required if upload=True)
        purpose (str): Purpose for OpenAI file upload
        convert_format (bool): Whether to convert Python dict format to JSON format
        shuffle (bool): Whether to randomly shuffle lines before splitting
    
    Returns:
        tuple: (list of created file paths, list of OpenAI file IDs)
    """
    # Split the file first
    created_files = split_jsonl_file(file_path, num_parts, convert_format, shuffle)
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
        
        print(f"\nUploaded {len(uploaded_file_ids)} out of {len(created_files)} files successfully")
    
    return created_files, uploaded_file_ids


def main():
    """Main function to handle command line arguments and execute the split."""
    parser = argparse.ArgumentParser(
        description="Split a JSONL file into multiple parts and optionally upload to OpenAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Just split the file sequentially (with automatic format conversion)
    python break_up_file.py data.jsonl 5
    
    # Split with random distribution
    python break_up_file.py data.jsonl 5 --shuffle
    
    # Split without format conversion (if already proper JSON)
    python break_up_file.py data.jsonl 5 --no-convert
    
    # Split and upload to OpenAI (with format conversion and random shuffle)
    python break_up_file.py data.jsonl 5 --upload --shuffle --api-key YOUR_API_KEY
    
    # Split and upload with custom purpose
    python break_up_file.py data.jsonl 5 --upload --api-key YOUR_API_KEY --purpose batch
    
    # Use environment variable for API key
    export OPENAI_API_KEY=your_key_here
    python break_up_file.py data.jsonl 5 --upload --shuffle
        """
    )
    
    parser.add_argument(
        'file_path',
        type=str,
        help='Path to the JSONL file to split'
    )
    
    parser.add_argument(
        'num_parts',
        type=int,
        help='Number of parts to split the file into'
    )
    
    parser.add_argument(
        '--upload',
        action='store_true',
        help='Upload split files to OpenAI storage'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        help='OpenAI API key (can also be set via OPENAI_API_KEY environment variable)'
    )
    
    parser.add_argument(
        '--purpose',
        type=str,
        default='fine-tune',
        choices=['fine-tune', 'batch', 'assistants'],
        help='Purpose for OpenAI file upload (default: fine-tune)'
    )
    
    parser.add_argument(
        '--no-convert',
        action='store_true',
        help='Skip Python dict to JSON conversion (use when file is already in proper JSON format)'
    )
    
    parser.add_argument(
        '--shuffle',
        action='store_true',
        help='Randomly shuffle lines before splitting (default: sequential order)'
    )
    
    # Parse arguments
    try:
        args = parser.parse_args()
    except SystemExit:
        return
    
    try:
        # Get API key from argument or environment variable
        api_key = args.api_key or os.getenv('OPENAI_API_KEY')
        
        # Validate upload requirements
        if args.upload and not api_key:
            print("Error: API key is required for uploading. Provide via --api-key or OPENAI_API_KEY environment variable", 
                  file=sys.stderr)
            sys.exit(1)
        
        if args.upload and not OPENAI_AVAILABLE:
            print("Error: OpenAI library not installed. Install with: pip install openai", file=sys.stderr)
            sys.exit(1)
        
        # Split the file (and upload if requested)
        created_files, uploaded_file_ids = split_and_upload_jsonl(
            args.file_path, 
            args.num_parts, 
            upload=args.upload,
            api_key=api_key,
            purpose=args.purpose,
            convert_format=not args.no_convert,
            shuffle=args.shuffle
        )
        
        # Print results
        print(f"\nSuccessfully split {args.file_path} into {len(created_files)} parts:")
        for file_path in created_files:
            print(f"  - {file_path}")
        
        if args.upload and uploaded_file_ids:
            print(f"\nOpenAI File IDs:")
            for i, file_id in enumerate(uploaded_file_ids, 1):
                print(f"  Part {i}: {file_id}")
            
    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
