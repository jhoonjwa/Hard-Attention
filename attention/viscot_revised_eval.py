import json


with open('/home/vqa/data/outputs/viscot/llava_v15_7b_3.json', 'r') as f:
        dataset_results = json.load(f)

good, bad = 0, 0
for data in dataset_results:
    if isinstance(data, dict):
        if data['revised_score'] >= 0.5 and data['score'] == 0:
            good += 1
        elif data['score'] >= 0.5 and data['revised_score'] == 0:
            bad += 1
    else:
         pass


print(good, bad)
