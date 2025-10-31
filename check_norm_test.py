import json

with open('data/games_2025_2026_current_norm_test.jsonl') as f:
    game = json.loads(f.readline())

h = game['teams']['H']
norm_feats = [k for k in h.keys() if '_norm' in k]

print(f'Normalized features: {len(norm_feats)}')
print('Examples:', norm_feats[:5] if norm_feats else 'NONE')

if norm_feats:
    print('\nSample values:')
    for f in norm_feats[:3]:
        print(f'  {f}: {h[f]}')
else:
    print('\n⚠️  Still no normalized features!')
    print('\nChecking what we do have:')
    print('All keys:', list(h.keys()))

