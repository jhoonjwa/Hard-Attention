#!/bin/bash

# Define file paths
repo_root=$(git rev-parse --show-toplevel)


# Environment setup for the Python script
export PYTHONPATH="${PYTHONPATH}:/home/jovyan/fileviewer/vqa-attention/llava" \
CUDA_VISIBLE_DEVICES=0 \

export SAVE_IMAGE_ATTENTION_MASK_PATH=/home/jovyan/fileviewer/vqa-attention/llava/attention/artifacts/image_attention_mask.pt \


python llava/llava_reasoning.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar_reasoned_extracted.json \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar_region.json \
--amplifier 1 \
--all_amplifier 1 \
--reasoning true \
--subset 1 \
--output-visualization-tensors