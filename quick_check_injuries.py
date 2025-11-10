import json

games = [json.loads(line) for line in open('data/games_train_with_players_90_new.jsonl')]

injured_games = []
for g in games:
    players_a = g.get('players', {}).get('A', [])
    players_h = g.get('players', {}).get('H', [])
    
    if any(p.get('projected_minutes') == 0.0 for p in players_a + players_h):
        injured_games.append(g)

print(f'Total games: {len(games)}')
print(f'Games with injuries: {len(injured_games)} ({len(injured_games)/len(games)*100:.1f}%)')

if injured_games:
    ex = injured_games[0]
    print(f"\nExample: {ex['game_id']}")
    away_injured = [p['player_name'] for p in ex['players']['A'] if p.get('projected_minutes') == 0.0]
    home_injured = [p['player_name'] for p in ex['players']['H'] if p.get('projected_minutes') == 0.0]
    print(f"  Away injured: {away_injured}")
    print(f"  Home injured: {home_injured}")

