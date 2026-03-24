export PYTHONPATH="/home/mingyi/AI-Projects/VLM2Vec-pro:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 python3 eval_video_mret.py \
 --encode_output_path eval_results/eval_baselines_results/gme_ms/rest_of_ms \
 --model_name Alibaba-NLP/gme-Qwen2-VL-2B-Instruct \
 --model_type gme \
 --model_backbone gme \
 --resize_use_processor true \
 --dataset_config configs/data_configs/darth_eval/test.yaml \
 --per_device_eval_batch_size 32
