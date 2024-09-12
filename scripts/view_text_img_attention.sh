export PYTHONPATH="${PYTHONPATH}://home/jovyan/fileviewer/vqa-attention/llava"

python llava/attention/text_attention_heatmap.py \
--input-dir /home/vqa/data/outputs/viscot/attention_llava_v15_7b \
--output-dir /home/vqa/bell_hoon/vqa-attention/llava/attention/data/heatmap/text \
--to_display True \
--instance-idx 91

python llava/attention/image_attention_heatmap.py \
--model-path liuhaotian/llava-v1.5-7b \
--dataset-attention-dir /home/vqa/data/outputs/viscot/attention_llava_v15_7b \
--dataset-result-dir /home/vqa/data/outputs/viscot/llava_v15_7b.json \
--output-dir /home/vqa/bell_hoon/vqa-attention/llava/attention/data/heatmap/image \
--vision-encoder-name openai/clip-vit-large-patch14-336 \
--instance-idx 90 \