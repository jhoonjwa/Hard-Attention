#!/bin/bash

# Define file paths
repo_root=$(git rev-parse --show-toplevel)


# Environment setup for the Python script
export PYTHONPATH=/home/jovyan/fileviewer/vqa-attention/llava \ 
CUDA_VISIBLE_DEVICES=0 \

export SAVE_IMAGE_ATTENTION_MASK_PATH=/home/jovyan/fileviewer/vqa-attention/llava/attention/artifacts/image_attention_mask.pt \

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/data/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 1 \
--all_amplifier 1 \
--mask_normalize True \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 1.5 \
--all_amplifier 1 \
--mask_normalize True \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 5 \
--all_amplifier 1 \
--mask_normalize True \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 100 \
--all_amplifier 1 \
--mask_normalize True \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 1000 \
--all_amplifier 1 \
--mask_normalize True \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 1.5 \
--all_amplifier 3 \
--mask_normalize True \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 5 \
--all_amplifier 3 \
--mask_normalize True \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 100 \
--all_amplifier 3 \
--mask_normalize True \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 1000 \
--all_amplifier 3 \
--mask_normalize True \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 1.5 \
--all_amplifier 10 \
--mask_normalize True \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 5 \
--all_amplifier 10 \
--mask_normalize True \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 100 \
--all_amplifier 10 \
--mask_normalize True \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 1000 \
--all_amplifier 10 \
--mask_normalize True \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 1.5 \
--all_amplifier 100 \
--mask_normalize True \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 5 \
--all_amplifier 100 \
--mask_normalize True \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 100 \
--all_amplifier 100 \
--mask_normalize True \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 1000 \
--all_amplifier 100 \
--mask_normalize True \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 1.5 \
--all_amplifier 1 \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 5 \
--all_amplifier 1 \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 100 \
--all_amplifier 1 \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 1000 \
--all_amplifier 1 \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 1.5 \
--all_amplifier 3 \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 5 \
--all_amplifier 3 \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 100 \
--all_amplifier 3 \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 1000 \
--all_amplifier 3 \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 1.5 \
--all_amplifier 10 \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 5 \
--all_amplifier 10 \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 100 \
--all_amplifier 10 \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 1000 \
--all_amplifier 10 \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 1.5 \
--all_amplifier 100 \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 5 \
--all_amplifier 100 \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 100 \
--all_amplifier 100 \
--subset 1 \
--output-visualization-tensors

python llava/attention/fast_hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k.json \
--amplifier 1000 \
--all_amplifier 100 \
--subset 1 \
--output-visualization-tensors





echo "Python script execution completed."

echo "All processes completed successfully."
