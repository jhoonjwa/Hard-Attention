import argparse
import torch
from torch.nn.functional import pad
import copy
from PIL import Image

import requests
from io import BytesIO
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import sys
import seaborn as sns
from matplotlib.colors import Normalize
os.environ['CURL_CA_BUNDLE'] = ''
from torch.utils.data import Dataset, DataLoader


def load_single_image(image_file):
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


def modify_attention_mask(args, bboxes, ratio, grid_size=(24, 24),): #### modifies image attention mask
    image_attentions = torch.ones(24, 24) * args.all_amplifier
    original_sum = image_attentions.sum()    
    
    if args.amplifier > 0:              ## edit 4 to adjust the total attention ratio to bbox
        amplifier = args.amplifier    
    else:
        amplifier = ((1 / ratio) - 1)
    
    for bbox in bboxes:
        x_start, y_start = int(bbox[0] * grid_size[0]), int(bbox[1] * grid_size[1])
        x_end, y_end = min(24, int(bbox[2] * grid_size[0])+1), min(24, int(bbox[3] * grid_size[1])+1)
        if x_start <= x_end and y_start <= y_end:
            image_attentions[y_start:y_end+1, x_start:x_end+1] *= amplifier
        else:
            print('invalid bbox')
            print(bbox)
    new_sum = image_attentions.sum()
    if args.mask_normalize:
        image_attentions *= original_sum / new_sum
    print(amplifier,  x_start, y_start,  x_end, y_end)
    return image_attentions.reshape(576)


def load_images(Dataset):
    '''
    loads images from absolute paths
    require absolute path of directory and json data
    '''

    dataset_name = data['dataset']
    dataset_subdir = data['dataset_subdir'] if 'dataset_subdir' in data.keys() else 'images'
    
    path = f"/home/jovyan/fileviewer/vqa-attention/llava/attention/data/{dataset_name}/{dataset_subdir}"
    image_path = [f"{path}/{data['image_id']}.{data['image_format']}" for data in dataset]
    images = []
    for path in image_path:
        images.append(load_single_image(path))
    return images

def dict_to_string(option):
    options = str()
    for key in option.keys():
        options += f'{key}: {option[key]}\n'
    return options

def modify_batched_attention_mask(args, bboxes, grid_size=(24, 24),): #### modifies image attention mask
    image_attentions = torch.ones(24, 24) * args.all_amplifier
    original_sum = image_attentions.sum()    
    
    if args.amplifier > 0:              ## edit 4 to adjust the total attention ratio to bbox
        amplifier = args.amplifier    
    else:
        amplifier = ((1 / ratio) - 1)
    
    for bbox in bboxes:
        x_start, y_start = int(bbox[0] * grid_size[0]), int(bbox[1] * grid_size[1])
        x_end, y_end = min(24, int(bbox[2] * grid_size[0])+1), min(24, int(bbox[3] * grid_size[1])+1)
        if x_start <= x_end and y_start <= y_end:
            image_attentions[y_start:y_end+1, x_start:x_end+1] *= amplifier
        else:
            print('invalid bbox')
            print(bbox)
    new_sum = image_attentions.sum()
    if args.mask_normalize:
        image_attentions *= original_sum / new_sum
    print(amplifier,  x_start, y_start,  x_end, y_end)
    return image_attentions.reshape(576)  
    


class VQADataset(Dataset):
    def __init__(self, args, dataset, model, image_processor, multi_choice=False, transform=None):
        self.img = [self.extract_image_features(image, model, image_processor) for image in load_images(dataset)]
        self.qs = [data['question'] for data in dataset]
        self.bbox = [data['bbox'] for data in dataset]
        self.options = [dict_to_string(data['options']) for data in dataset] if multi_choice else False
        self.mask = [modify_attention_mask(args, bbox) for bbox in self.bbox]
        
    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        image, qs, bbox, img_mask= self.img[idx], self.qs[idx], self.bbox[idx], self.mask[idx]      
        
        if options:
            options = self.options[idx]
            return dict(image=image, question=qs, bbox=bbox, options=options, mask=img_mask)
        else:
            return dict(image=image, question=qs, bbox=bbox, mask=img_mask)

    
    def extract_image_features(self, image, model, image_processor):
        images = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = model.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = model.encode_images(images)
        return image_features
        
    
        




