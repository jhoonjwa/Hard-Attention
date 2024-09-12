#!/bin/bash

# Define file paths
repo_root=$(git rev-parse --show-toplevel)


# Environment setup for the Python script
export PYTHONPATH="${PYTHONPATH}:/home/jovyan/fileviewer/vqa-attention/llava" \
CUDA_VISIBLE_DEVICES=0 \

export SAVE_IMAGE_ATTENTION_MASK_PATH=/home/jovyan/fileviewer/vqa-attention/llava/attention/artifacts/image_attention_mask.pt \

python llava/attention/fast_hard_attention_w_metric.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar_regions_topk_1.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar_regions_topk_1.json \
--amplifier 5 \
--all_amplifier 1.5 \
--mask_normalize True \
--subset 1 \
--output-visualization-tensors


python llava/attention/fast_hard_attention_w_metric.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar_regions_topk_1.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar_regions_topk_1.json \
--amplifier 1 \
--all_amplifier 1.5 \
--mask_normalize True \
--subset 1 \
--output-visualization-tensors


python llava/attention/fast_hard_attention_w_metric.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar_regions_topk_1.json  \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar_regions_topk_1.json \
--amplifier 1.5 \
--all_amplifier 1.5 \
--mask_normalize True \
--subset 1 \
--output-visualization-tensors
