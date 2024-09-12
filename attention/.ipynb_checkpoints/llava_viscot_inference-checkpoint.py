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
from transformers import AutoProcessor, LlavaForConditionalGeneration

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

instruction_prompt = '''

**Generating tools to solve the given image-based question**
**Purpose:** Given a query and its corresponding code solution, your task
is to generate "tools" that are needed to solve the question. A "tool" can analyze the image on a basic level, and can also perform atomic logical, mathematical operations.
The generated tools should not be query-dependent but query-agnostic. 

Consider the following principles:
1. Break down the query to atomic steps required to solve the query.
2. A single tool can only handle a single atomic step. 
3. The generated tools should solve queries of the same type, based on common reasoning steps rather than specific object types.
4. Name the tool honestly to ensure its functionality is not overclaimed.
5. Tool names should be general enough to be re-used to different queries. Do not refer to the name of the query-specific entities.
Generate the tool in the form shown:
[tool name]: [usage of the tool]

Tool names and its usage should clearly point out what it will analyze.

**Examples**
Query: What is written on blue vehicle in the image?

Low-Quality Example
Tools: Blue Vehicle Detector: Detects blue vehicles in the image
       Data Analyzer: Analyzes blue car.

Reason for being low-quality: Referred to query-specific entity(Blue Vehicle). Unclear tool name and usage(Data Analyzer)

High-Quality Example
Tools: Object Detector: Detects cars in the image
       Color Feature Extractor: Extracts the color of detected image patches
       Text Extractor: Extracts text in the detected blue car image patch. 
Reason for being high-quality: Used general tool names. Query was broken down to multiple steps

You do not need to generate the Reason. 
**Begin!**
Query: {query}
Tools:
'''


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


        
def run_llava(args):
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf" #"llava-hf/llava-1.5-7b-hf"
    
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        cache_dir="/home/vqa/model-weights",
    ).to(0)
    #processor = AutoProcessor.from_pretrained(model_id)      
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf") 
    print(f"Loading input data from {args.input}")
    dataset = json.load(open(args.input, 'r'))
    outputs = []
    for idx, data in tqdm(enumerate(dataset)):
        image_id = data['image_id']
        image_format = data['image_format']
        dataset_name = data['dataset']
        bbox = data['bbox']
        ratio = data['ratio']

        image_path = f"/home/vqa/data/dataset/{dataset_name}/images/{image_id}.{image_format}"
        image = load_image(image_path)
        if image is None:
            continue
        question = data['question']

        prompt = f"[INST] <image>\n{instruction_prompt.format(query=question)} [/INST]"

        inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
        if 'image_sizes' in inputs:
            del inputs['image_sizes']
        output = model.generate(inputs['input_ids'], max_new_tokens=2000)
        #print(processor.decode(output[0][2:], skip_special_tokens=True))
        
        response = output[0][inputs['input_ids'].shape[-1]:]
        #print(processor.decode(response, skip_special_tokens=True))
        if args.output_file:
            outputs.append({
                **data,
                'output': processor.decode(response, skip_special_tokens=True),
            })

    if args.output_file:
        print(f"Saving log file to {args.output_file}")
        with open(args.output_file, 'w') as f:
            json.dump(outputs, f, indent=4)
        

def main(args):
    run_llava(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model parameters
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    
    # data parameters
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--output-file", type=str, default="./outputs.json", help="outputs file (.json) path")
    args = parser.parse_args()

    main(args)



