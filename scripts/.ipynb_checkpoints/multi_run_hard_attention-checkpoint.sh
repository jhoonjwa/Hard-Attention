#!/bin/bash

# Define file paths
repo_root=$(git rev-parse --show-toplevel)


# Environment setup for the Python script
export PYTHONPATH="${PYTHONPATH}:/home/jovyan/fileviewer/vqa-attention/llava"
CUDA_VISIBLE_DEVICES=0 \
export SAVE_IMAGE_ATTENTION_MASK_PATH=/home/jovyan/fileviewer/vqa-attention/llava/attention/artifacts/image_attention_mask.pt

python llava/attention/hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/data/gqa_flickr30k.json \
--output-dir /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k_unnormalized.json \
--amplifier 1.5 \
--mask_normalize True \
--all_amplifier 1 \
--subset 7 \
--output-visualization-tensors

python llava/attention/hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k_unnormalized.json \
--output-dir /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k_unnormalized.json \
--amplifier 1 \
--all_amplifier 2 \
--mask_normalize True \
--subset 7 \
--output-visualization-tensors

python llava/attention/hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k_unnormalized.json \
--output-dir /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k_unnormalized.json \
--amplifier 5 \
--all_amplifier 1 \
--mask_normalize True \
--subset 7 \
--output-visualization-tensors

python llava/attention/hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k_unnormalized.json \
--output-dir /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k_unnormalized.json \
--amplifier 5 \
--all_amplifier 1 \
--mask_normalize False \
--subset 7 \
--output-visualization-tensors

python llava/attention/hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k_unnormalized.json \
--output-dir /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k_unnormalized.json \
--amplifier 5 \
--all_amplifier 2 \
--mask_normalize True \
--subset 5 \
--output-visualization-tensors

python llava/attention/hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k_unnormalized.json \
--output-dir /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k_2 \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/gqa_flickr30k_unnormalized.json \
--amplifier 1 \
--all_amplifier 1 \
--mask_normalize True \
--subset 1 \
--output-visualization-tensors








echo "Python script execution completed."

echo "All processes completed successfully."
