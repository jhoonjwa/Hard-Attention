import argparse
import torch
from torch.nn.functional import pad
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from fast_hard_attention_utils import *
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



AVG_IMAGE_FEATURES = False
SUM_IMAGE_FEATURES = True

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

     

def llava_generate(args, qs, image_features, tokenizer, model, model_name, conv_mode, bbox, ratio, img_mask, option=None):
    #prompt = f'''Question: {qs} Options: {option}\n Answer:'''

    '''
    args: args defined in argparse
    qs->list: batch of questions
    image_features->list: batch of encoded images.
    '''
    
    prompt = qs
    ''' Here are a few examples: Question: What is the capital of France? Options: A: Paris \n B: London. Answer: A.\n Question: Where is the blue flag located next to? A: church\n B: bridge\n Answer: A.\n '''

    ## qs used to be prompt
    if model.config.mm_use_im_start_end:
        prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
    else:
        prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()  # text
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    
    input_token_len = input_ids.shape[1]    

    modified_attention_mask = modify_attention_mask(args, bbox, ratio)
    idx = torch.where(input_ids == IMAGE_TOKEN_INDEX)[-1].item()
    #print(modified_attention_mask)
    model.get_mask_attributes(modified_attention_mask, idx)
    with torch.inference_mode():
        input_output = model.generate(  # SampleDecoderOnlyOutput or GreedyDecoderOnlyOutput
            input_ids,
            image_features=image_features,
            do_sample=True if args.sample else False,
            temperature=0.0,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            output_attentions=True,
            return_dict_in_generate=True
        )
    
    input_output_ids = input_output.sequences
    outputs = tokenizer.batch_decode(input_output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    
    return outputs





def decode_image_token(tokenizer, input_ids, image_features_len=None):
    # Replace IMAGE_TOKEN_INDEX with the string <image>
    tokens = []
    for id in input_ids:
        if id == IMAGE_TOKEN_INDEX:
            if AVG_IMAGE_FEATURES or SUM_IMAGE_FEATURES:
                tokens.append(DEFAULT_IMAGE_TOKEN)
            elif image_features_len:
                tokens.extend([f"<patch_{i+1}>" for i in range(image_features_len)])
            else:
                raise ValueError("image_features_len must be provided if not using AVG_IMAGE_FEATURES")
        else:
            tokens.extend(tokenizer.convert_ids_to_tokens([id], skip_special_tokens=False))
    
    return tokens
 


def run_llava_with_attention(args):
    disable_torch_init()  
    
    print(f"Loading model from {args.model_path} with mask_normalize {args.mask_normalize}")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, args.model_base, model_name)
    vision_encoder_name = model.get_vision_tower().vision_tower_name
    
    print(f"Loading input data from {args.input}")
    dataset = json.load(open(args.input, 'r'))
    outputs = []

    use_options = args.multi_choice
    vqa_dataset = VQADataset(args, model, image_processor, dataset, use_options)
    vqa_loader = DataLoader(vqa_dataset, batch_size=arg.batch_size)
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", load_in_4bit=True)
    processor = AutoProcessor.from_pretrained(model_id, pad_token="<pad>")
    
    count = 0
    
    for batch in tqdm(enumerate(dataset)):        
        instance_output_dir = os.path.join(args.output_dir, f"{idx}_{image_id}")
        os.makedirs(instance_output_dir, exist_ok=True)
        print('model is generating')
        image_features, qs, bbox, img_mask = batch['image'], batch['question'], batch['bbox'], batch['mask']
        
        #output = llava_generate(args, qs, image_features, tokenizer, model, model_name, args.conv_mode, bbox, 1, , img_mask, option=options)

 
        
        print('new output', output)        
        if args.output_file:
            outputs.append({
                **data,
                f'output_with_{args.amplifier}_{args.mask_normalize}_{args.all_amplifier}': output,
                #f'ratio_with{args.amplifier}_{args.mask_normalize}_{args.all_amplifier}': attention_ratio                
            })

    if args.output_file:
        print(f"Saving log file to {args.output_file}")
        with open(args.output_file, 'w') as f:
            json.dump(outputs, f, indent=4)
        

def main(args):
    run_llava_with_attention(args)


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
    parser.add_argument("--instance-idx", type=lambda s: set(int(idx) for idx in s.split(',')), default=None, help="comma separated list of instance indices to visualize")
    parser.add_argument("--amplifier", type=float, default=3, required=False)
    parser.add_argument("--mask_normalize", type=bool, default=False, required=False)
    parser.add_argument("--all_amplifier", type=float, default=1, required=False)

    parser.add_argument("--subset", type=float, default=7, required=False)
   
    # generation parameters
    parser.add_argument("--sample", action="store_true", help="use sampling instead of greedy decoding")
    
    # visualization mode parameters
    parser.add_argument("--output-visualization-tensors", action="store_true", help="output dict of raw attention tensors for in-depth analysis")

    # attention aggregation hyperparameters
    parser.add_argument("--pool-method", type=str, choices=["mean", "max", "top_k_mean"], default="top_k_mean", help="pooling method for image attentions")
    parser.add_argument("--hidden-top-k", type=int, default=3, help="top k hidden states to average for image attentions")
    args = parser.parse_args()

    main(args)