import os 
import json
import torch
import tqdm

data_dir = '/home/vqa/data/outputs/viscot/attention_llava_v15_7b'
restore_path = '/home/vqa/data/outputs/viscot/llava_v15_7b_1.json'

with open(restore_path, 'r') as f:
    full_data = json.load(f)

def extract_number(directory):
    # Split the directory name by underscore and convert the first part to integer
    return int(directory.split('_')[0])



#img_ids =  [path.split('_')[-1] for path in os.listdir(data_dir) if path[-5:] != '.json']
paths = [path for path in os.listdir(data_dir) if path[-5:] != '.json']
paths = sorted(paths, key=extract_number)
print(paths[0:5])
data_path = [data_dir + '/' +path for path in paths if path[-5:] != '.json']
print(len(os.listdir(data_dir)))
outputs = []
#for data in full_data:
    #outputs.append(data['output'])

count = 0
for i, path in enumerate(data_path):
    #print(path)
    data = torch.load(os.path.join(path, "extracted_attention.pt"))
    #print(data['input_tokens'])
    output = str()
    for key in data.keys():
        if key[-1] != '</s>':
            output += key[-1].replace('▁', ' ')
    full_data[i]['output'] = output
    print(output)
    print(i)


'''

for id, output in zip(img_ids, outputs):
    for i in range(len(full_data)):
        if full_data[i]['image_id'] == id:
            count += 1
            print(count)
            full_data[i]['output'] = output
            break
'''

with open(restore_path, 'w') as f:
    json.dump(full_data, f, indent=2)





'''
data_path = [data_dir + '/' +path for path in os.listdir(data_dir) if path[-5:] != '.json']

for i, path in enumerate(data_path):
    print(path)
    data = torch.load(os.path.join(path, "output_visualization_tensor_dict.pt"))
    print(data['input_tokens'])
    output = str()
    for key in data.keys():
        if key[-1] != '</s>':
            output += key[-1].replace('▁', ' ')
    full_data[i]['output'] = output
    #print(output)
    print(i)
    break

with open(restore_path, 'w') as f:
    json.dump(full_data, f, indent=2)
'''