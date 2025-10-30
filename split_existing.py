import json, random

random.seed(42)
src = 'data/games_train_with_players.jsonl'
dst_train = 'data/games_train_with_players_90.jsonl'
dst_val = 'data/games_val_with_players.jsonl'

with open(src, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

random.shuffle(data)
cut = int(len(data) * 0.9)
print(f'Total games: {len(data)}')
print(f'Training: {cut} games')
print(f'Validation: {len(data) - cut} games')

with open(dst_train, 'w', encoding='utf-8') as f:
    for r in data[:cut]:
        f.write(json.dumps(r) + '\n')

with open(dst_val, 'w', encoding='utf-8') as f:
    for r in data[cut:]:
        f.write(json.dumps(r) + '\n')