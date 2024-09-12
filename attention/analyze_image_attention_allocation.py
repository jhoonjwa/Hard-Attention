import multiprocessing
import torch
import os
import argparse
import json
import pdb

from joblib import Parallel, delayed
from torch.nn.functional import pad
from tqdm import tqdm
from image_attention_heatmap import find_subdir_with_prefix
from typing import List, Dict, Any


def aggregate_tensor(tensor: torch.Tensor, k: int, dim: int, method=None, final_aggregate=None) -> torch.Tensor:
    tensor = tensor.float()

    if method == 'topk':
        aggregated_tensor, _ = torch.topk(tensor, k, dim=dim)
    elif method == 'max':
        aggregated_tensor, _ = torch.max(tensor, dim=dim)
        
    if final_aggregate == 'mean':
        return aggregated_tensor.mean(dim=dim)
    else:
        return aggregated_tensor


def process_tensor(tensor:torch.Tensor, k: int):
    # tensor shape is assumed to be (num_layers, num_heads, feature_len)
    # it can be (num_layers, num_heads)

    # Average top-k over num_heads dimension
    num_head_averaged = aggregate_tensor(tensor, k, dim=1, method='topk', final_aggregate='mean')

    # Average top-k over num_layers dimension
    num_layer_averaged = aggregate_tensor(num_head_averaged, k, dim=0, method='topk', final_aggregate='mean')
    
    # If the tensor is already a scalar, return it
    if len(num_layer_averaged.shape) == 0:
        return num_layer_averaged

    feature_averaged = aggregate_tensor(num_layer_averaged, k, dim=0, method='topk')
    # feature_averaged shape is now a scalar

    return feature_averaged


def analyze_attention(instance_idx, dataset_result, attention_dir):
    '''
    Extracts attention weights from the given attention directory and saves them in the result directory.
    Divides the attention weights into system, image, input, and output tokens.
    Args:
        attention_dir (str): The directory containing the attention weights.
        dataset_result_dir (str): The directory where the extracted attention weights will be saved.

    Returns:
        None


    Saved attention weights are in the following format:
    {
        'generated_token1': {
            'system': torch.FloatTensor(num_layers, num_heads, system_prompt_len),
            'image': torch.FloatTensor(num_layers, num_heads, image_feature_len),
            'input': {
                'input_token1': torch.FloatTensor(num_layers, num_heads),
                'input_token2': torch.FloatTensor(num_layers, num_heads),
                ...
            },
            'output': {
                'output_token1': torch.FloatTensor(num_layers, num_heads),
                'output_token2': torch.FloatTensor(num_layers, num_heads),
                ...
            }
        },
        'generated_token2': ...
    }
    '''
    subdir = f"{attention_dir}/{instance_idx}_{dataset_result['image_id']}"
    file_path = os.path.join(subdir, "extracted_attention.pt")

    if os.path.isfile(file_path):
        extracted_attention = torch.load(file_path)
        output_tokens = list(extracted_attention.keys())
        for output_token in output_tokens:
            top_k_attentions = {}
            output_attention = extracted_attention[output_token]
            for token_type, tensor in output_attention.items():
                if token_type == 'system' or token_type == 'image':
                    # Average top-k attention weights for image tokens
                    top_k_attentions[token_type] = process_tensor(tensor, 5)
                    print(token_type)
                    print(top_k_attentions[token_type])
                else:
                    for specific_token, specific_tensor in tensor.items():
                        top_k_attentions[specific_token] = process_tensor(specific_tensor, 5)
                        print(specific_token)
                        print(top_k_attentions[specific_token])




def main(args):
    with open(args.dataset_result_dir, "r") as f:
        dataset_results = json.load(f)


    for instance_idx, dataset_result in tqdm(enumerate(dataset_results), total=len(dataset_results)):
        analyze_attention(instance_idx, dataset_result, args.dataset_attention_dir)
        pdb.set_trace()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-attention-dir', type=str, default="/home/vqa/data/outputs/viscot/attention_llava_v15_7b", required=False)
    parser.add_argument('--dataset-result-dir', type=str, default="/home/vqa/data/outputs/viscot/llava_v15_7b.json", required=False)
    #parser.add_argument("--hidden-top-k", type=int, default=3, help="top k hidden states to average for image attentions")
    args = parser.parse_args()


    main(args)