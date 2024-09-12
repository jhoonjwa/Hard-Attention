#!/bin/bash

# Define file paths
repo_root=$(git rev-parse --show-toplevel)

original_file="$repo_root/llava/model/llava_arch.py"
replacement_file="$repo_root/llava/model/llava_arch_for_image_attention.py"
backup_file="$repo_root/llava/model/llava_arch_backup.py"

# Backup the original file
cp "$original_file" "$backup_file"
echo "Original file backed up."

# Replace the original file with the replacement file
cp "$replacement_file" "$original_file"
echo "Replacement file set as original."

# Environment setup for the Python script
export PYTHONPATH="${PYTHONPATH}:/home/vqa/joonyeongs/vqa-attention/llava"
CUDA_VISIBLE_DEVICES=0 SAVE_IMAGE_ATTENTION_MASK_PATH=llava/attention/artifacts/image_attention_mask.pt \
python llava/attention/run_llava_with_image_attention.py \
--model-path liuhaotian/llava-v1.5-7b \
--input /home/vqa/data/dataset/viscot_benchmark_modified.json \
--output-dir /home/vqa/data/outputs/viscot/hard_attention_llava_v15_7b_llama \
--output-file /home/vqa/data/outputs/viscot/hard_llava_v15_7b.json \
--output-visualization-tensors

echo "Python script execution completed."

# Restore the original files from backup
mv "$backup_file" "$original_file"
echo "Original file restored."

echo "All processes completed successfully."
