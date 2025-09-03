import json

# Your escaped JSON string
escaped_json = """{\\"away_team\\":{\\"name\\":\\"MIN\\",\\"stats\\":{\\"OEFF\\":116.5,\\"DEFF\\":108.1,\\"PACE\\":98.3,\\"REST_DAYS\\":2}}}"""

print("Step 1: Escaped JSON string")
print(escaped_json)

print("\\nStep 2: Parse with json.loads() - handles null conversion automatically")
parsed_data = json.loads(escaped_json)
print(parsed_data)

print("\\nStep 3: Now you can use this as a normal Python dict")
print("Away team name:", parsed_data["away_team"]["name"])

