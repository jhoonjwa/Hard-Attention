

export PYTHONPATH="${PYTHONPATH}:/home/vqa/bell_hoon/vqa-attention/llava"


echo bbox metric start
python llava/attention/bbox_metric.py \
--topk 5 \
--start_layer 0

echo bbox metric start
python llava/attention/bbox_metric.py \
--topk 5 \
--start_layer 7

echo bbox metric start
python llava/attention/bbox_metric.py \
--topk 5 \
--start_layer 12

echo bbox metric start
python llava/attention/bbox_metric.py \
--topk 5 \
--start_layer 17

echo bbox metric start
python llava/attention/bbox_metric.py \
--topk 5 \
--start_layer 22

echo bbox metric start
python llava/attention/bbox_metric.py \
--topk 5 \
--start_layer 27

echo bbox metric start
python llava/attention/bbox_metric.py \
--topk 32 \
--start_layer 0

echo bbox metric start
python llava/attention/bbox_metric.py \
--topk 25 \
--start_layer 7

echo bbox metric start
python llava/attention/bbox_metric.py \
--topk 20 \
--start_layer 12

echo bbox metric start
python llava/attention/bbox_metric.py \
--topk 15 \
--start_layer 17

echo bbox metric start
python llava/attention/bbox_metric.py \
--topk 10 \
--start_layer 22

echo bbox metric start
echo start layer '27' is equivalent to top-5

echo finished