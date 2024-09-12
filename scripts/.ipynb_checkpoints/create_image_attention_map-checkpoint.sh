#!/bin/bash

export PYTHONPATH="${PYTHONPATH}://home/jovyan/fileviewer/vqa-attention/llava"


python llava/attention/image_attention_heatmap.py \
--model-path liuhaotian/llava-v1.5-7b \
--dataset-attention-dir /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar  \
--dataset-result-dir /home/jovyan/fileviewer/vqa-attention/llava/attention/output/vstar.json \
--output-dir /home/jovyan/fileviewer/vqa-attention/llava/attention/data/heatmap/vstar \
--vision-encoder-name openai/clip-vit-large-patch14-336 \
--instance-idx 0 \



