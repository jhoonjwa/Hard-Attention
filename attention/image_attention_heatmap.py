import torch
import os
import fnmatch
import argparse
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

from PIL import Image
from transformers import AutoTokenizer

max_cutoff_quantile = 0.995
cmap = plt.get_cmap('viridis')


def find_subdir_with_prefix(directory, prefix):
    for root, dirs, files in os.walk(directory):
        for dir in fnmatch.filter(dirs, prefix+'*'):
            return os.path.join(root, dir)
    return None


def extract_target_indices(model_path, output_tokens, target_tokens):
    """
    model_path (str): model name or path
    output_tokens (List[str]): generated output tokens
    target_tokens (str): target tokens to find in output_tokens

    Example:
    output_tokens = ['▁The', '▁name', '▁of', '▁the', '▁De', 'aler', '▁is', '▁C', '.', '▁Mon', 'k', '.', '</s>']
    target_tokens = 'the Dealer'
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, cache_dir="/home/vqa/model-weights")
    target_tokens_list = tokenizer.tokenize(target_tokens)
    for i in range(len(output_tokens) - len(target_tokens_list) + 1):
        if output_tokens[i:i + len(target_tokens_list)] == target_tokens_list:
            return i, i + len(target_tokens_list) - 1
    return -1, -1  


def create_image_heatmap(vision_encoder_name, all_attention_weights, save_path, image_path):
    items = vision_encoder_name.split("patch")[-1].split('-')  # 14 or 32
    patch_size = int(items[0])  # 14 or 32
    if len(items) == 2:
        resolution = int(items[1])  # 336
    else:
        resolution = 224
    num_patch_per_side = resolution // patch_size
    
    all_weights = torch.stack(list(all_attention_weights.values()))
    global_min = all_weights.min().item()
    global_max = all_weights.max().item()
    norm = colors.Normalize(vmin=global_min, vmax=global_max)

    #global_max_cutoff = torch.quantile(all_weights, max_cutoff_quantile, interpolation="nearest").item()
    #for key, weights in all_attention_weights.items():
    #    all_attention_weights[key] = torch.clamp(weights, max=global_max_cutoff)  
    #norm = colors.Normalize(vmin=global_min, vmax=global_max_cutoff)
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)

    nrows = 1
    ncols = len(all_attention_weights) + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))

    # Plot the image
    axes[0].imshow(image_array)
    axes[0].axis('off')  
    axes[0].set_title("Input Image")

    for i, (title, weights) in enumerate(all_attention_weights.items()):
        ax = axes[i+1]
        weights_reshaped = weights.view(num_patch_per_side, num_patch_per_side)
        sns.heatmap(weights_reshaped, ax=ax, cmap=cmap, xticklabels=False, yticklabels=False, norm=norm, cbar=False)
        ax.set_title(title)
 
    
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # need to adjust
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, cax=cbar_ax)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    print("Saving to", save_path)
    plt.savefig(save_path, bbox_inches='tight', dpi=400)
    
    #save_path_pdf = save_path.replace(".png", ".pdf")
    #print("Saving to", save_path_pdf)
    #plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')


def visualize_image_attention(model_path, vision_encoder_name, output_tokens, image_attentions, target_tokens_list, hidden_top_k, save_path, image_path):
    # image_attentions (torch.FloatTensor): (generated_len, num_layers, num_heads, image_features_len)

    num_image_features = image_attentions.shape[-1]
    print(image_attentions.shape)
    all_attention_weights = {}
    image_attentions = image_attentions[:-1]  # remove last EOS token
    print(image_attentions.shape)
    
    attention_weights = torch.empty((len(image_attentions), num_image_features), dtype=torch.float16)
    for i, image_attention in enumerate(image_attentions):  # output_len
        image_attention = image_attention.float()
        image_attention_fused, _ = torch.topk(image_attention, k=hidden_top_k, dim=0)  # (num_layers, num_head, num_image_features) -> (hidden_top_k, num_head, num_image_features)
        image_attention_fused = image_attention_fused.mean(dim=0)  # (num_head, num_image_features)
        image_attention_fused, _ = torch.topk(image_attention_fused, k=hidden_top_k, dim=0)  # (num_head, num_image_features) -> (hidden_top_k, num_image_features)
        image_attention_fused = image_attention_fused.mean(dim=0)  # (num_image_features)
        attention_weights[i] = image_attention_fused
        
    attention_weights_pooled = attention_weights.mean(dim=0)
    all_attention_weights["all tokens"] = attention_weights_pooled  # (num_image_features)

    # target tokens
    for target_tokens in target_tokens_list:
        target_start, target_end = extract_target_indices(model_path, output_tokens, target_tokens)
        if target_start is None:
            print("Target tokens not found:", target_tokens)
        
        if target_start == target_end:
            target_token_weights_pooled = attention_weights[target_start]
        else:
            target_token_weights_pooled = attention_weights[target_start:target_end].mean(dim=0)
        all_attention_weights[target_tokens] = target_token_weights_pooled
    
    create_image_heatmap(vision_encoder_name, all_attention_weights, save_path, image_path)


def main(args):
    '''
    output_visualization_tensor_dict.pt contains:
        input_tokens (List[str]): input tokens starting from the image token, with the image token representing averaged/summed attentions to image features
        output_tokens (List[str]): generated output tokens
        weights_matrix (torch.FloatTensor): (num_layers, generated_len, num_heads, final_seq_len)
        autoregressive_mask (np.ndarray(bool)): (final_seq_len, generated_len). for each generated token, False for input + itself, True for future tokens
        image_attentions (torch.FloatTensor): (generated_len, num_layers, num_heads, image_features_len)
        input_output_attention (torch.FloatTensor): (generated_len, num_layers, num_heads, final_seq_len)
    '''
    with open(args.dataset_result_dir, "r") as f:
        dataset_results = json.load(f)
    dataset_result = dataset_results[args.instance_idx]
    print(f"Question: {dataset_result['question']}\nOutput: {dataset_result['output_with_3.0_False_2.0']}\nAnswer: {dataset_result['answer']}\n")

    target_tokens = []
    while True:
        user_input = input("Word or phrase to create attention map (Enter to finish):")
        if user_input == '':
            break
        else:
            target_tokens.append(user_input)

    with open(args.dataset_result_dir, "r") as f:
        dataset_results = json.load(f)

    idx = args.instance_idx
    attention_folder_paths = [path for path in os.listdir(args.dataset_attention_dir) if os.path.exists(os.path.join( f"{args.dataset_attention_dir}/{path}", "output_visualization_tensor_dict.pt"))]
    print(len(attention_folder_paths))

    image_ids = ['_'.join(path.split('_')[-2:]) for path in attention_folder_paths]
    idx = image_ids.index(dataset_results[idx]['image_id'])
    subdir = f"{args.dataset_attention_dir}/{attention_folder_paths[idx]}"
    print(subdir)
    if os.path.isfile(os.path.join(subdir, "output_visualization_tensor_dict.pt")):
        output_visualization_tensors = torch.load(os.path.join(subdir, "output_visualization_tensor_dict.pt"))
        output_tokens = output_visualization_tensors['output_tokens']
        image_attentions = output_visualization_tensors['image_attentions']
    else:
        raise ValueError("No output_visualization_tensor_dict.pt found in", subdir)

    save_path = os.path.join(args.output_dir, f"image_heatmap_{args.instance_idx}.png")
    image_subdir = 'images' if 'dataset_subdir' not in dataset_result.keys() else dataset_result['dataset_subdir']
    
    image_path = f"/home/jovyan/fileviewer/vqa-attention/llava/attention/data/{dataset_result['dataset']}/{image_subdir}/{dataset_result['image_id']}.{dataset_result['image_format']}"
    #image_path = f"/home/jovyan/fileviewer/vqa-attention/llava/attention/data/viscot/flickr30k/1096305461.jpg"

    visualize_image_attention(args.model_path, args.vision_encoder_name, output_tokens, image_attentions, target_tokens, args.hidden_top_k, save_path, image_path)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default="liuhaotian/llava-v1.5-7b", required=True)
    parser.add_argument('--dataset-attention-dir', type=str, default="/home/vqa/data/outputs", required=True)
    parser.add_argument('--dataset-result-dir', type=str, default="/home/vqa/data/outputs", required=True)
    parser.add_argument('--output-dir', type=str, default="/home/vqa/data/outputs", required=True)
    parser.add_argument('--vision-encoder-name', type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--instance-idx", type=int, required=True)
    parser.add_argument("--hidden-top-k", type=int, default=3, help="top k hidden states to average for image attentions")
    args = parser.parse_args()
    
    main(args)