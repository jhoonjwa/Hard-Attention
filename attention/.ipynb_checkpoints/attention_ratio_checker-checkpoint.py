import tqdm
import json
from PIL import Image
import requests
from io import BytesIO

def size_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    if x1 == x2 and y1 == y2:
        return 1/576
    
    else:
        if x1 == x2:
            return abs(y2 - y1)
        elif y1 == y2:
            return abs(x2 - x1)
        else:
            return abs((x2 - x1) * (y2 - y1))


dataset = json.load(open('/home/vqa/data/outputs/viscot/llava_v15_7b_1.json', 'r'))
save_path = '/home/vqa/bell_hoon/vqa-attention/llava/attention/image'
count = 0
for data in dataset:
     bbox = data['bbox']
     size = size_of_bbox(bbox)
     ratio = data['ratio']
     if  (size_of_bbox(bbox) <= 0.25 and size_of_bbox(bbox) >= 0.05 and (ratio / size_of_bbox(bbox)) <= ((1 - ratio) / (1 - size_of_bbox(bbox))) ):
         count += 1

print(count)