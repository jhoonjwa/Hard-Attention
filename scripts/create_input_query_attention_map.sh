export PYTHONPATH="${PYTHONPATH}://home/jovyan/fileviewer/vqa-attention/llava"

python llava/attention/input_query_attention_heatmap.py \
--input-dir /home/vqa/data/outputs/viscot/attention_llava_v15_7b_3 \
--output-dir /home/vqa/bell_hoon/vqa-attention/llava/attention/data/heatmap/text \
--to_display True \
--instance-idx 115