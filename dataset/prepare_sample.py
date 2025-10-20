import json
import random

random.seed(42)

with open('dataset/hotpot_train_v1.1.json', 'r') as f:
    data = json.load(f)

sampled = random.sample(data, 200)

with open('dataset/hotpot_sample_200.json', 'w') as f:
    json.dump(sampled, f, indent=2)

print('采样完成，共200个样本。')