print('PROBE:start') 
from game.prediction_client import PredictionClientFactory 
client = PredictionClientFactory.create_client('gemini') 
try: 
    content, usage = client._predict_with_genai_sdk_json('{\\" "A\:\AWAY\}', client.get_model_config()['model_1_id'], max_tokens=64, temperature=0.1) 
    print('PROBE:ok', len(content)) 
except Exception as e: 
    print('PROBE:error', e) 
print('PROBE:done') 
