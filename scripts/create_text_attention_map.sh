export PYTHONPATH="${PYTHONPATH}:/home/vqa/joonyeongs/vqa-attention/llava"

python llava/attention/text_attention_heatmap.py \
--input-dir /home/vqa/data/outputs/viscot/attention_llava_v15_7b \
--output-dir /home/vqa/bell_hoon/vqa-attention/llava/attention/data/heatmap/text \
--to_display True \
--instance-idx 2212