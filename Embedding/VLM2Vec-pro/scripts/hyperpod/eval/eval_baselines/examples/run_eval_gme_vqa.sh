export PYTHONPATH=/home/mingyi/AI-Projects/VLM2Vec-pro:$PYTHONPATH

CUDA_VISIBLE_DEVICES=2 python3 eval_video_qa.py \
 --encode_output_path eval_results/eval_baselines_results/eval_gme_vqa \
 --model_name Alibaba-NLP/gme-Qwen2-VL-2B-Instruct \
 --model_type gme \
 --model_backbone gme \
 --resize_use_processor true \
 --dataset_config configs/data_configs/darth_eval/vqa.yaml \
 --per_device_eval_batch_size 32