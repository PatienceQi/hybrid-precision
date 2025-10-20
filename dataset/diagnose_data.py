import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'hotpot_sample_200.json')

with open(file_path, 'r') as f:
    data = json.load(f)

sample = data[0]
print('Sample keys:', list(sample.keys()))
context = sample.get('context', {})
print('Context type:', type(context))
if isinstance(context, dict):
    print('Number of context items:', len(context))
    keys = list(context.keys())
    print('Keys type:', type(keys[0]))
    print('First key:', keys[0])
    print('Value type:', type(context[keys[0]]))
    print('First value length:', len(context[keys[0]]) if isinstance(context[keys[0]], list) else 'n/a')
else:
    print('Context:', context)