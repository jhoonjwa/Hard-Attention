#!/bin/bash

# Define file paths
repo_root=$(git rev-parse --show-toplevel)


# Environment setup for the Python script
export PYTHONPATH="${PYTHONPATH}:/home/jovyan/fileviewer/vqa-attention/llava"
CUDA_VISIBLE_DEVICES=1 \
export SAVE_IMAGE_ATTENTION_MASK_PATH=/home/jovyan/fileviewer/vqa-attention/llava/attention/artifacts/image_attention_mask.pt



python llava/attention/hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json  \
--output-dir /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json \
--amplifier 2 \
--all_amplifier 3 \
--mask_normalize True \
--subset 1 \
--output-visualization-tensors

python llava/attention/hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json  \
--output-dir /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json \
--amplifier 5 \
--all_amplifier 3 \
--mask_normalize True \
--subset 1 \
--output-visualization-tensors

python llava/attention/hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json  \
--output-dir /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json \
--amplifier 10 \
--all_amplifier 3 \
--mask_normalize True \
--subset 1 \
--output-visualization-tensors


python llava/attention/hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json  \
--output-dir /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json \
--amplifier 2 \
--all_amplifier 5 \
--mask_normalize True \
--subset 1 \
--output-visualization-tensors

python llava/attention/hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json  \
--output-dir /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json \
--amplifier 5 \
--all_amplifier 5 \
--mask_normalize True \
--subset 1 \
--output-visualization-tensors

python llava/attention/hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json  \
--output-dir /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json \
--amplifier 10 \
--all_amplifier 5 \
--mask_normalize True \
--subset 1 \
--output-visualization-tensors

python llava/attention/hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json  \
--output-dir /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json \
--amplifier 2 \
--all_amplifier 10 \
--mask_normalize True \
--subset 1 \
--output-visualization-tensors

python llava/attention/hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json  \
--output-dir /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json \
--amplifier 5 \
--all_amplifier 10 \
--mask_normalize True \
--subset 1 \
--output-visualization-tensors

python llava/attention/hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json  \
--output-dir /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json \
--amplifier 10 \
--all_amplifier 10 \
--mask_normalize True \
--subset 1 \
--output-visualization-tensors


python llava/attention/hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json  \
--output-dir /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json \
--amplifier 2 \
--all_amplifier 3 \
--subset 1 \
--output-visualization-tensors

python llava/attention/hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json  \
--output-dir /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json \
--amplifier 5 \
--all_amplifier 3 \
--subset 1 \
--output-visualization-tensors

python llava/attention/hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json  \
--output-dir /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json \
--amplifier 10 \
--all_amplifier 3 \
--subset 1 \
--output-visualization-tensors


python llava/attention/hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json  \
--output-dir /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json \
--amplifier 2 \
--all_amplifier 5 \
--subset 1 \
--output-visualization-tensors

python llava/attention/hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json  \
--output-dir /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json \
--amplifier 5 \
--all_amplifier 5 \
--subset 1 \
--output-visualization-tensors

python llava/attention/hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json  \
--output-dir /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json \
--amplifier 10 \
--all_amplifier 5 \
--subset 1 \
--output-visualization-tensors

python llava/attention/hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json  \
--output-dir /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json \
--amplifier 2 \
--all_amplifier 10 \
--subset 1 \
--output-visualization-tensors

python llava/attention/hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json  \
--output-dir /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json \
--amplifier 5 \
--all_amplifier 10 \
--subset 1 \
--output-visualization-tensors

python llava/attention/hard_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json  \
--output-dir /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar \
--output-file /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json \
--amplifier 10 \
--all_amplifier 10 \
--subset 1 \
--output-visualization-tensors


echo "Python script execution completed."

echo "All processes completed successfully."
