import tqdm
import json
from PIL import Image
import requests
from io import BytesIO


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        try:
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        except Exception:
            print(f"Failed to load image from {image_file}")
            image = None
    else:
        image = Image.open(image_file).convert('RGB')
    return image

dataset = json.load(open('/home/vqa/data/outputs/viscot/hard_llava_v15_7b_4.json', 'r'))
save_path = '/home/vqa/bell_hoon/vqa-attention/llava/attention/image'
for data in dataset:
        image_id = data['image_id']
        image_format = data['image_format']
        dataset_name = data['dataset']

        image_path = f"/home/vqa/data/dataset/{dataset_name}/images/{image_id}.{image_format}"
        image = load_image(image_path )
        image.save(save_path + f'/{image_id}.{image_format}')