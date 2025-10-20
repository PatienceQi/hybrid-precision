import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'hotpot_sample_200.json')

with open(file_path, 'r') as f:
    data = json.load(f)

titles = set()
for item in data:
    context = item.get('context', [])
    if isinstance(context, list):
        for ctx_item in context:
            if isinstance(ctx_item, list) and len(ctx_item) > 0 and isinstance(ctx_item[0], str):
                titles.add(ctx_item[0])

# Save to JSON
with open(os.path.join(script_dir, 'titles.json'), 'w') as f:
    json.dump(sorted(list(titles)), f, indent=2)

print('Titles saved to titles.json')
print(f'Total unique titles: {len(titles)}')
print('Unique Wikipedia titles:')
print('\n'.join(sorted(list(titles))[:10]))  # Print first 10 for preview
print('... (and more)')