import json
import os
import requests
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
titles_file = os.path.join(script_dir, 'titles.json')

with open(titles_file, 'r') as f:
    titles = json.load(f)

kb_dir = os.path.join(script_dir, '..', 'knowledge_base')
os.makedirs(kb_dir, exist_ok=True)
documents_file = os.path.join(kb_dir, 'documents.json')

api_url = 'https://en.wikipedia.org/w/api.php'
documents = []

for i, title in enumerate(titles, 1):
    params = {
        'action': 'query',
        'format': 'json',
        'titles': title,
        'prop': 'extracts',
        'exintro': True,
        'explaintext': True,
        'exsentences': 10  # Limit to intro sentences
    }
    retries = 3
    for attempt in range(retries):
        try:
            headers = {'User-Agent': 'RAGAS-Wikipedia-Downloader/1.0 (https://github.com/yourusername/yourproject; your@email.com)'}
            response = requests.get(api_url, params=params, timeout=10, headers=headers)
            response.raise_for_status()
            data = response.json()
            pages = data['query']['pages']
            page = next(iter(pages.values()))
            if 'extract' in page:
                documents.append({'title': title, 'text': page['extract']})
                print(f'Downloaded {i}/{len(titles)}: {title}')
            else:
                print(f'No extract for {title}')
            time.sleep(0.5)  # Rate limit
            break
        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                print(f'Error downloading {title} after {retries} attempts: {e}')
            time.sleep(1 * (attempt + 1))  # Exponential backoff

with open(documents_file, 'w') as f:
    json.dump(documents, f, indent=2)

print(f'Knowledge base saved to {documents_file} with {len(documents)} documents.')