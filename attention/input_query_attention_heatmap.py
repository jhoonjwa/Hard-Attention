import torch
import os
import fnmatch
import argparse
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import imgkit
import json
import numpy as np
from PIL import Image

LEADING_SPACE = 9601
cmap = plt.get_cmap('BuGn')
alpha = 0.5
brightness_threshold = 0.6  # Set your brightness threshold here


def find_subdir_with_prefix(directory, prefix):
    for root, dirs, files in os.walk(directory):
        for dir in fnmatch.filter(dirs, prefix+'*'):
            return os.path.join(root, dir)
    return None

def preprocess_inputs(output_visualization_tensors):
    tokenized_input, input_tensors = [], []     
    for key in output_visualization_tensors.keys():
        key_tensors = []
        for _ in output_visualization_tensors[key]['input'].keys():
            if _ not in tokenized_input:
                tokenized_input.append(_)
            if output_visualization_tensors[key]['input'][_].shape[0] > 32:
                duplicate = int(output_visualization_tensors[key]['input'][_].shape[0] / 32)
                key_tensors.extend(output_visualization_tensors[key]['input'][_][int(0 + 32*i):int(32 + 32*i), :] for i in range(duplicate))

            else:
                key_tensors.append(output_visualization_tensors[key]['input'][_])
        input_tensors.append(torch.stack(key_tensors))                            
    
    print(torch.stack(input_tensors).shape)
    print('output_len', len(list(output_visualization_tensors.keys())), 'input_len: ', )
    return tokenized_input, torch.stack(input_tensors)
    


def create_text_heatmap(tokens, weights, save_path):
    tokens = [' ' + token[1:] if ord(token[0]) == LEADING_SPACE else token for token in tokens]  # Add a space if the token starts with a leading space
    off = (sum(weights) / len(weights)) * alpha
    normer = colors.Normalize(vmin=min(weights)-off, vmax=max(weights)+off)
    bg_colors = [cmap(normer(x)) for x in weights]  # Get RGBA colors
    hex_colors = [colors.to_hex(color) for color in bg_colors]  # Convert RGBA to hex
    print(f'token length: {len(tokens)}, hex colors: {len(hex_colors)}')
    if len(tokens) != len(hex_colors):
        raise ValueError("number of tokens and colors don't match")

    style_elems = []
    span_elems= []

    for i, token in enumerate(tokens):
        # Check if there is a leading space
        if ord(token[0]) == LEADING_SPACE:
            space = ' '  # A space will be added before this token
            token = token[1:]  # Remove the leading space from the token itself
        else:
            space = ''
        
        # Convert RGB to HSV, then set text color based on brightness
        hsv = colors.rgb_to_hsv(bg_colors[i][:3])  # Exclude alpha from RGBA
        text_color = '#ffffff' if hsv[2] < brightness_threshold else '#000000'
        
        # Append the styles, consider the leading space for background color
        style_elems.append(f'.c{i} {{ background-color: {hex_colors[i]}; color: {text_color}; }}')
        span_elems.append(f'{space}<span class="c{i}">{space}{token}</span>')

    html_text = f"""<!DOCTYPE html><html><head><link href="https://fonts.googleapis.com/css?family=Roboto+Mono&display=swap" rel="stylesheet"><style>span {{ font-family: "Helvetica Neue", monospace; font-size: 20px; }} {' '.join(style_elems)} {' '.join([f'.c{i}:not(:first-child)::before {{ content: " "; }}' for i in range(1, len(tokens))])}</style></head><body>{''.join(span_elems)}</body></html>"""
    
    print("Saving to", save_path)
    save_options = {
        'format': 'png',
        'quality': '100'
    }    
    imgkit.from_string(html_text, save_path, options=save_options)


def visualize_text_attention(output_tokens, image_attentions, hidden_top_k, save_path):
    weights = []
    
    image_attentions = torch.sum(image_attentions, dim=0)
    print('image_attentions_shape: ', image_attentions.shape)
    for image_attention in image_attentions:  # generated len
        image_attention = image_attention.float()
        image_attention_fused, _ = torch.topk(image_attention, k=hidden_top_k, dim=0)  # (hidden_top_k, num_image_features)
        image_attention_fused = torch.mean(image_attention_fused, dim=0)  # (num_image_features)
        image_attention_fused = torch.mean(image_attention_fused, dim=0)  # (1)
        weights.append(image_attention_fused.item())
    print(len(weights))
    create_text_heatmap(output_tokens, weights, save_path)


def main(args):
    with open(args.dataset_result_dir, "r") as f:
        dataset_results = json.load(f)

    idx = args.instance_idx
    subdir = f"{args.input_dir}/{idx}_{dataset_results[idx]['image_id']}"
    file_path = os.path.join(subdir, "output_visualization_tensor_dict.pt")

    if os.path.isfile((os.path.join(subdir, "extracted_attention.pt"))):
        output_visualization_tensors = torch.load(os.path.join(subdir, "extracted_attention.pt"))
        for key in output_visualization_tensors.keys():
            #print(output_visualization_tensors[key]['input'].keys())
            pass
        #temp_tensors = torch.load(file_path)        
        qinput_tokens, qinput_attentions = preprocess_inputs(output_visualization_tensors)
    else:
        raise ValueError("No output_visualization_tensor_dict.pt found in", subdir)
    
    save_path = os.path.join(args.output_dir, f"input_query_heatmap_{args.instance_idx}_q2i_attention.png")
    visualize_text_attention(qinput_tokens, qinput_attentions, args.hidden_top_k, save_path)
    if args.to_display:
        img = Image.open(save_path)
        img.show()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default="/home/vqa/data/outputs/viscot/attention_llava_v15_7b", required=True)
    parser.add_argument('--dataset-result-dir', type=str, default="/home/vqa/data/outputs/viscot/llava_v15_7b.json", required=False)
    parser.add_argument('--output-dir', type=str, default="/home/vqa/data/outputs/viscot/heatmap", required=True)
    parser.add_argument("--to_display", type=bool, default=False, required=True)
    parser.add_argument("--instance-idx", type=int, required=True)
    parser.add_argument("--hidden-top-k", type=int, default=3, help="top k hidden states to average for image attentions")
    args = parser.parse_args()
    
    main(args)