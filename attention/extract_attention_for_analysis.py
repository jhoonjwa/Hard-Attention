import multiprocessing
import torch
import os
import argparse
import json

from joblib import Parallel, delayed
from torch.nn.functional import pad
from tqdm import tqdm
from image_attention_heatmap import find_subdir_with_prefix

SYSTEM_PROMPT_LEN = 35
IMAGE_FEATURE_LEN = 576

def update_attention_by_token(attention_dict, key, new_attention):
    if key not in attention_dict.keys():
        attention_dict[key] = []
    attention_dict[key].append(new_attention) 
    
def stack_attention_by_layers(attention_by_token):
    # Iterate over each token and its associated values in the dictionary
    for token, token_data in attention_by_token.items():
        # Iterate over each type within the token data (e.g., 'system', 'image', 'input', 'output')
        for token_type, tensor_list in token_data.items():
            if token_type in ['system', 'image']:
                # Stack the list of tensors for 'system' and 'image' types
                attention_by_token[token][token_type] = torch.stack(tensor_list, dim=0)
            else:
                # 'token_type' should be 'input' or 'output' here
                # Iterate over each specific token within the type and stack its tensors
                for specific_token, specific_tensors in tensor_list.items():
                    attention_by_token[token][token_type][specific_token] = torch.stack(specific_tensors, dim=0)



def extract_attention_and_save(relative_path, dataset_result, attention_dir):
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

    # Find the subdirectory containing the attention weights for the given instance
    subdir = f"{attention_dir}/{relative_path}"
    file_path = os.path.join(subdir, "output_visualization_tensor_dict.pt")
    if os.path.isfile(file_path):
        print(f"Extracting attention for instance {relative_path}...")
        output_visualization_tensors = torch.load(file_path)
        input_tokens = output_visualization_tensors['input_tokens'][1:]
        output_tokens = output_visualization_tensors['output_tokens']
        input_len, output_len = len(input_tokens), len(output_tokens)

        # Contains attention weights for each token in the output sequence allocated to system, input, image, and output tokens
        # tuple(tuple(torch.FloatTensor)): (generated_len (num_layers=32 (1*batch_size, num_heads=32, generated_len, input_output_sequence_len)))
        attention_output = output_visualization_tensors['input_output_attention']

        # Initialize the dictionary to store attention weights for each token
        attention_by_token = {key: {} for key in output_tokens}

        try:
            for generated_token_idx, generated_token_attentions in enumerate(attention_output):
                generated_token = output_tokens[generated_token_idx]
                attention_by_token[generated_token] = {
                    'system': [],
                    'image': [],
                    'input': {},
                    'output': {}
                }

                for layer_idx, layer_attention in enumerate(generated_token_attentions):
                    if generated_token_idx == 0:
                        layer_attention = layer_attention[:, :, -1, :]
                    layer_attention = layer_attention.squeeze()         # (num_heads, sequence_length)

                    # Extract attentions
                    layer_system_attentions = layer_attention[:, :SYSTEM_PROMPT_LEN]
                    layer_image_attentions = layer_attention[:, SYSTEM_PROMPT_LEN:SYSTEM_PROMPT_LEN+IMAGE_FEATURE_LEN]
                    start_index = SYSTEM_PROMPT_LEN + IMAGE_FEATURE_LEN
                    layer_input_attentions = layer_attention[:, start_index:][:, :input_len]
                    layer_output_attentions = layer_attention[:, start_index:][:, input_len:]

                    # Update 'system' and 'image' attentions
                    update_attention_by_token(attention_by_token[generated_token], 'system', layer_system_attentions)
                    update_attention_by_token(attention_by_token[generated_token], 'image', layer_image_attentions)

                    # Update 'input' attentions
                    for i in range(layer_input_attentions.shape[1]):
                        input_token = input_tokens[i]
                        update_attention_by_token(attention_by_token[generated_token]['input'], input_token, layer_input_attentions[:, i])

                    # Update 'output' attentions
                    for i in range(layer_output_attentions.shape[1]):
                        output_token = output_tokens[i]
                        update_attention_by_token(attention_by_token[generated_token]['output'], output_token, layer_output_attentions[:, i])
            
            stack_attention_by_layers(attention_by_token)              

            file_path = os.path.join(subdir, "extracted_attention.pt")
            torch.save(attention_by_token, file_path)
            print(f"Saved extracted attention for instance {relative_path}")

        except Exception as e:
            print(f"Error extracting attention for instance {relative_path}: {e}")


def main(args):
    with open(args.dataset_result_dir, "r") as f:
        dataset_results = json.load(f)

    num_processes = multiprocessing.cpu_count() // 2

    attention_folder_paths = os.listdir(args.dataset_attention_dir)
    print(len(attention_folder_paths))
    if '5_xkbv0228_3' in attention_folder_paths:
        pass
        attention_folder_paths.remove('5_xkbv0228_3')
    Parallel(n_jobs=8)(delayed(extract_attention_and_save)(attention_folder_paths[instance_idx], dataset_result, args.dataset_attention_dir) for instance_idx, dataset_result in tqdm(enumerate(dataset_results)))
        
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-attention-dir', type=str, default="/home/vqa/data/outputs/viscot/hard_attention_llava_v15_7b_llama_3", required=False)
    parser.add_argument('--dataset-result-dir', type=str, default="/home/vqa/data/outputs/viscot/llava_v15_7b_4_scored.json", required=False)
    #parser.add_argument("--hidden-top-k", type=int, default=3, help="top k hidden states to average for image attentions")
    args = parser.parse_args()


    main(args)
    