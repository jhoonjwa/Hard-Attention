import torch
import json
import numpy as np
import os
from tqdm import tqdm 
import argparse
import matplotlib.pyplot as plt



# File paths
json_file_path = '/home/vqa/data/outputs/viscot/llava_v15_7b.json'
attention_folder_path = '/home/vqa/data/outputs/viscot/attention_llava_v15_7b'
# Load JSON data

with open(json_file_path, 'r') as file:
    dataset = json.load(file)


def draw_histogram(outputs, name):
    outputs = [output for output in outputs if output != -1]
    bins = [i * 0.05 for i in range(int(1/0.05) + 1)]  # Creates bins from 0, 0.05, ..., to 1.0
    counts, _, _ = plt.hist(outputs, bins=bins, edgecolor='black')
    total_counts = sum(counts)
    ratios = [count / total_counts for count in counts]

    plt.clf()
    plt.bar(bins[:-1], ratios, width=0.05, align='edge', edgecolor='black')  
    plt.xlabel('Range')
    plt.ylabel('Ratio')
    plt.title('Ratio of Numbers in Each Bin')
    plt.savefig(f'llava/attention/graphs/{name}_attention_ratio_histogram.png')




def main(args):
    from joblib import Parallel, delayed
    from dotenv import load_dotenv
    def bbox_area(coords):
        x1, y1, x2, y2 = coords
        area = abs(x1-x2) * abs(y1-y2)
        if area >= args.bbox_size:
            return False
        return True
    if args.bbox_small:
        success_data = [item for item in dataset if item['score'] >= 0.5 and bbox_area(item['bbox'])]
        fail_data = [item for item in dataset if item['score'] == 0.0 and bbox_area(item['bbox'])]
    
    else:
        success_data = [item for item in dataset if item['score'] >= 0.5]
        fail_data = [item for item in dataset if item['score'] == 0.0]    

    # Function to get the path of the attention file
    print(f'num of correct: {len(success_data)}, num of wrongs: {len(fail_data)}')
    def get_attention_path(image_id):
        for dir_name in os.listdir(attention_folder_path):
            if image_id in dir_name:
                return os.path.join(attention_folder_path, dir_name, 'extracted_attention.pt')
        return None

    # Function to calculate total image attention using top-k
    def get_total_image_attention(attention_data, k=5):
        total_attention = torch.zeros((24,24))
        image_attentions = []
        printed = True
        for key, token_data in attention_data.items():
            if key[-1] != '</s>':
                image_attentions = token_data['image']
                image_attention = image_attentions.float().unsqueeze(0)[:, args.start_layer:, :, :]
                image_attention_fused, _ = torch.topk(image_attention, k, dim=1)  # (num_layers, num_head, num_image_features) -> (hidden_top_k, num_head, num_image_features)
                image_attention_fused = image_attention_fused.mean(dim=1)  # (1, num_head, num_image_features)
                
                image_attention_fused, _ = torch.topk(image_attention_fused, k, dim=1)  # (num_head, num_image_features) -> (hidden_top_k, num_image_features)
                image_attention_fused = image_attention_fused.mean(dim=1)  # (num_image_features)

                image_attention_fused = image_attention_fused[0].reshape(24, 24)
                total_attention += image_attention_fused
        total_attention /= (len(attention_data)-1)
        return total_attention

    # Function to calculate attention over bbox
    def calculate_attention_over_bbox(attention_tensor, bbox, grid_size=(24, 24)):
        x_start, y_start = int(bbox[0] * grid_size[0]), int(bbox[1] * grid_size[1])
        x_end, y_end = min(24, int(bbox[2] * grid_size[0]) + 1), min(24, int(bbox[3] * grid_size[1]) + 1)
        if x_start <= x_end and y_start <= y_end:
            attention_in_bbox = attention_tensor[y_start:y_end+1, x_start:x_end+1].sum()
            total_attention = attention_tensor.sum()
            return float(attention_in_bbox / total_attention), float(attention_in_bbox), float(total_attention)
        return -1

    # Function to process entries
    def process_entries(entry_list, k):
        ratios = []
        for entry in tqdm(entry_list, total=len(entry_list)):
            print(entry.keys())
            attention_path = get_attention_path(entry['image_id'])
            if attention_path and os.path.exists(attention_path):
                attention_data = torch.load(attention_path)
                total_attention = get_total_image_attention(attention_data, k)
                ratio = calculate_attention_over_bbox(total_attention, entry['bbox'])
                if ratio is None:
                    pass
                entry[f'exp_{str(args.topk)}_{str(args.bbox_small)}_{str(args.bbox_size)}'] = ratio
                if ratio is not None:
                    ratios.append(ratio)
        return ratios, entry_list
    
    def get_ratio(coords, bbox, j, grid_size=(24, 24)):
        x_start, y_start = int(bbox[0] * grid_size[0]), int(bbox[1] * grid_size[1])
        x_end, y_end = int(bbox[2] * grid_size[0]), int(bbox[3] * grid_size[1])
        count = 0
        for coord in coords:
            if coord[1] >= x_start and coord[1] <= x_end and coord[0] >= y_start and coord[0] <= y_end:
                count += 1
        return count / j

    def bbox_ratio(entry, k, j) -> tuple:    
        attention_path = get_attention_path(entry['image_id'])
        if attention_path and os.path.exists(attention_path):
            attention_data = torch.load(attention_path)
            top_attentions = get_total_image_attention(attention_data, k)            
            ### for top-k within bbox
            #indices = torch.argsort(top_attentions.flatten(), descending=True)[j:]
            #dim_2d = torch.unravel_index(indices, (24, 24))
            #ratio = get_ratio(dim_2d, entry['bbox'], j)
            ratio, bbox_sum, total_sum = calculate_attention_over_bbox(top_attentions, entry['bbox'])
            if ratio is None:
                print(entry['bbox'])
            entry[f'filtered_{args.topk}ratio'] = ratio
            entry[f'filtered_{args.topk}bbox_sum'] = bbox_sum
            entry[f'filtered_{args.topk}total_sum'] = total_sum
        return (ratio, entry, bbox_sum, total_sum)



    # Number of top attentions to consider
    if args.topk:
        k = args.topk
    else:
        k = 32

    j = 30     ### top-k attentions within dim. 
    # Calculate ratios
    #success_ratios, success_list = process_entries(success_data, k)
    #fail_ratios, fail_list = process_entries(fail_data, k)

    BATCHES = 16
    data_len = len(success_data)
    success_ratios, fail_ratios, new_data, bboxes_success, totals_success, bboxes_fail, totals_fail = [], [], [], [], [], [], []
    for i in range(0, data_len, BATCHES):
        end_index = min(i+BATCHES, data_len)
        try:        
            results = Parallel(n_jobs=BATCHES, verbose=100, prefer="threads")(
                delayed(bbox_ratio)(
                    entry = datum,
                    k = k,
                    j = j,
                )
                for datum in tqdm(success_data[i : end_index], total=len(success_data))
            )
            outputs, datums, bbox_sum, total_sum = zip(*results)
            bboxes_success.append(bbox_sum)
            totals_success.append(total_sum)
            for output, datum in zip(outputs, datums):
                success_ratios.append(output)
                new_data.append(datum)
        except Exception as e:
            print(f"Error processing batch {i} to {end_index}: {e}")
    
    if None in success_ratios:
        print(success_data[success_ratios.index(None)]['bbox'])
   
    
    data_len = len(fail_data)
    for i in range(0, data_len, BATCHES):
        end_index = min(i+BATCHES, data_len)
        try:        
            results = Parallel(n_jobs=BATCHES, verbose=100, prefer="threads")(
                delayed(bbox_ratio)(
                    entry = datum,
                    k = k,
                    j = j,
                )
                for datum in tqdm(fail_data[i : end_index], total=len(fail_data)) 
            )
            outputs, datums, bbox_sum, total_sum = zip(*results)
            bboxes_fail.append(bbox_sum)
            totals_fail.append(total_sum)
            for output, datum in zip(outputs, datums):
                fail_ratios.append(output)
                new_data.append(datum)
        except Exception as e:
            print(f"Error processing batch {i} to {end_index}: {e}")
        
    #[fail_data[i].update({'ratio': n}) for i, n in enumerate(outputs)

    with open("top_20.txt", "a") as file:
        file.write('new experiment\n')
        file.write(f"Success Ratios Mean when Top-K: {args.topk} with bbox filtering set {args.bbox_small}: {np.mean(success_ratios):.4f}, Std Dev: {np.std(success_ratios):.4f}\n")
        file.write(f"Fail Ratios Mean when Top-K: {args.topk} with bbox filtering set {args.bbox_small}: {np.mean(fail_ratios):.4f}, Std Dev: {np.std(fail_ratios):.4f}\n")
        file.write(f"Success bbox attention Mean when Top-K: {args.topk} with bbox filtering set {args.bbox_small}: {np.mean(bboxes_success):.4f}, Std Dev: {np.std(bboxes_success):.4f}\n")
        file.write(f"Success bbox attention Mean when Top-K: {args.topk} with bbox filtering set {args.bbox_small}: {np.mean(bboxes_fail):.4f}, Std Dev: {np.std(bboxes_fail):.4f}\n")
        file.write(f"Success total attention Mean when Top-K: {args.topk} with bbox filtering set {args.bbox_small}: {np.mean(totals_success):.4f}, Std Dev: {np.std(totals_success):.4f}\n")
        file.write(f"Success total attention Mean when Top-K: {args.topk} with bbox filtering set {args.bbox_small}: {np.mean(totals_fail):.4f}, Std Dev: {np.std(totals_fail):.4f}\n")
        file.write('end of experiment\n')

    with open('/home/vqa/data/outputs/viscot/hard_llava_v15_7b_4_1.json', 'w') as file:
        json.dump(new_data, file, indent = 2)

    # Statistics and plotting  
    print(f"Success Ratios Mean when Top-K: {args.topk} with bbox filtering set {args.bbox_small}: {np.mean(success_ratios):.4f}, Std Dev: {np.std(success_ratios):.4f}")
    print(f"Fail Ratios Mean when Top-K: {args.topk} with bbox filtering set {args.bbox_small}: {np.mean(fail_ratios):.4f}, Std Dev: {np.std(fail_ratios):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bbox_small', type=bool, default=False, required=False)
    parser.add_argument('--start_layer', type=int, default=0, required=False)
    parser.add_argument('--topk', type=int, default=3, required=True)
    parser.add_argument('--bbox_size', type=float, default=0.6, required=False)
    args = parser.parse_args()

print(args.bbox_small)
main(args)

