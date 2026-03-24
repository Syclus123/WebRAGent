export PYTHONPATH="/home/mingyi/AI-Projects/VLM2Vec-pro:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=2 python3 /home/mingyi/AI-Projects/VLM2Vec-pro/evaluation/mmeb_pro_baselines/eval_baselines_class.py \
  --encode_output_path /home/mingyi/AI-Projects/VLM2Vec-pro/eval_results/eval_baselines_results/eval_e5v_class \
  --model_backbone llava_next \
  --model_name royokong/e5-v \
  --pooling eos --normalize True --per_device_eval_batch_size 16 --dataloader_num_workers 8 --resize_use_processor true --dataset_config configs/data_configs/eval/video_cls.yaml
