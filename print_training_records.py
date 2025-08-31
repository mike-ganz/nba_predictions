import json

def parse_inner_json_in_messages(record):
    """
    For a record with a 'messages' field, parse any inner JSON strings in the 'content' fields.
    Returns a new record with parsed content fields where possible.
    """
    if not isinstance(record, dict):
        return record
    if "messages" not in record or not isinstance(record["messages"], list):
        return record

    new_record = dict(record)  # shallow copy
    new_messages = []
    for msg in record["messages"]:
        new_msg = dict(msg)
        content = msg.get("content")
        if (
            isinstance(content, str)
            and content.strip().startswith("{")
            and content.strip().endswith("}")
        ):
            try:
                # Try to parse the content as JSON
                new_msg["content"] = json.loads(content)
            except Exception:
                # If parsing fails, leave as string
                pass
        new_messages.append(new_msg)
    new_record["messages"] = new_messages
    return new_record

def read_jsonl(file_path, n, start=0):
    """
    Reads n records from a .jsonl file starting at entry 'start' (0-based index)
    and prints them, parsing inner JSON in 'content' fields of 'messages'.

    :param file_path: Path to the .jsonl file
    :param n: Number of records to print
    :param start: The index of the first record to print (0-based)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if i >= start + n:
                break
            try:
                record = json.loads(line.strip())
                record = parse_inner_json_in_messages(record)
                # Pretty print with ensure_ascii=False for unicode, indent for readability
                print(json.dumps(record, ensure_ascii=False, indent=2))
            except json.JSONDecodeError as e:
                print(f"Error decoding line {i+1}: {e}")

if __name__ == "__main__":
    # Example usage
    file_path = "data/training/ULTRA_OPTIMIZED_incremental.jsonl"  # Replace with your file
    n = 1  # Number of records to print
    start = 22750  # Change this to the starting index you want (0-based)
    read_jsonl(file_path, n, start)
