import argparse
import torch
from torch.nn.functional import pad
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.eval import *
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
from fast_hard_attention import *

import os
import time
import re
from joblib import Parallel, delayed
from dotenv import load_dotenv
load_dotenv()

os.environ['CURL_CA_BUNDLE'] = ''

def eval_llava_response(args):
    with open(args.output_file, 'r', encoding='utf-8') as f:
        dataset_results = json.load(f)
    #dataset_results = dataset_results[:12]
    BATCH = 2
    dataset_results_evaluated = []
    count = 0
        
    print(len(dataset_results))
    
    for i in range(0, len(dataset_results), BATCH):
        print(f"Processing {i} to {min(i+BATCH, len(dataset_results))}")
        outputs = Parallel(n_jobs=BATCH, verbose=100, prefer="threads")(delayed(get_score)(dataset_result) for idx, dataset_result in enumerate(dataset_results[i:min(i+BATCH, len(dataset_results))]))
        for output in outputs:
            dataset_results_evaluated.append(output)
        
        print(f"Finished {i} to {min(i+BATCH, len(dataset_results))}")
        
    
    data_to_be_saved = [data for data in dataset_results_evaluated if isinstance(data, dict)]

    with open(args.output_file, 'w') as f:
        json.dump(data_to_be_saved, f, indent=2, ensure_ascii=False)
    
        

def main(args):
    run_llava_with_attention(args)
    eval_llava_respinse(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model parameters
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    
    # data parameters
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-file", type=str, default="./outputs.json", help="outputs file (.json) path")
    parser.add_argument("--instance-idx", type=lambda s: set(int(idx) for idx in s.split(',')), default=None, help="comma separated list of instance indices to visualize")
    parser.add_argument("--amplifier", type=float, default=3, required=False)
    parser.add_argument("--mask_normalize", type=bool, default=False, required=False)
    parser.add_argument("--all_amplifier", type=float, default=1, required=False)

    parser.add_argument("--subset", type=float, default=7, required=False)
    parser.add_argument("--multi_choice", type=bool, default=False, required=False)
   
    # generation parameters
    parser.add_argument("--sample", action="store_true", help="use sampling instead of greedy decoding")
    
    # visualization mode parameters
    parser.add_argument("--output-visualization-tensors", action="store_true", help="output dict of raw attention tensors for in-depth analysis")

    # attention aggregation hyperparameters
    parser.add_argument("--pool-method", type=str, choices=["mean", "max", "top_k_mean"], default="top_k_mean", help="pooling method for image attentions")
    parser.add_argument("--hidden-top-k", type=int, default=3, help="top k hidden states to average for image attentions")
    args = parser.parse_args()

    main(args)