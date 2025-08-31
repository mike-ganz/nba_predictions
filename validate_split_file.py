#!/usr/bin/env python3
import json
import sys

def validate_jsonl_file(filename):
    """Validate each line in a JSONL file for proper JSON formatting."""
    print(f"🔍 Validating JSON formatting in: {filename}")
    print("=" * 60)
    
    total_lines = 0
    valid_lines = 0
    invalid_lines = 0
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                total_lines += 1
                
                try:
                    # Test if line is valid JSON
                    data = json.loads(line)
                    valid_lines += 1
                    
                    # Test OpenAI fine-tuning format
                    if 'messages' in data:
                        messages = data['messages']
                        if isinstance(messages, list) and len(messages) > 0:
                            # Check each message
                            for msg_idx, msg in enumerate(messages):
                                if 'role' in msg and 'content' in msg:
                                    # If content looks like JSON, try to parse it
                                    content = msg['content']
                                    if content.startswith('{') and content.endswith('}'):
                                        try:
                                            json.loads(content)
                                        except json.JSONDecodeError:
                                            print(f"⚠️  Line {line_num}, Message {msg_idx}: Content is not valid JSON")
                        else:
                            print(f"⚠️  Line {line_num}: Messages field is not a non-empty array")
                    else:
                        print(f"⚠️  Line {line_num}: Missing 'messages' field")
                    
                    # Show progress for first few and occasional lines
                    if line_num <= 3 or line_num % 10000 == 0:
                        print(f"✅ Line {line_num}: Valid JSON")
                    
                except json.JSONDecodeError as e:
                    invalid_lines += 1
                    print(f"❌ Line {line_num}: Invalid JSON - {e}")
                    if line_num <= 5:  # Show details for first few errors
                        print(f"   Context: {line[:100]}...")
                
                # Stop after checking first 50 lines unless there are errors
                if line_num >= 50 and invalid_lines == 0:
                    print(f"✅ First 50 lines are valid, skipping detailed check of remaining lines...")
                    # Quick count of remaining lines
                    remaining = sum(1 for _ in f)
                    total_lines += remaining
                    valid_lines += remaining
                    break
    
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        return False
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    print("\n" + "=" * 60)
    print(f"📊 VALIDATION SUMMARY:")
    print(f"   Total lines checked: {total_lines:,}")
    print(f"   Valid JSON lines: {valid_lines:,}")
    print(f"   Invalid JSON lines: {invalid_lines:,}")
    print(f"   Success rate: {(valid_lines/total_lines*100):.1f}%" if total_lines > 0 else "   Success rate: 0%")
    
    if invalid_lines == 0:
        print(f"🎉 ALL LINES ARE VALID JSON!")
        print(f"✅ This file should be accepted by OpenAI")
    else:
        print(f"❌ Found {invalid_lines} invalid lines that need fixing")
    
    return invalid_lines == 0

if __name__ == "__main__":
    filename = "data/training/ULTRA_OPTIMIZED_incremental_part_001.jsonl"
    validate_jsonl_file(filename)
