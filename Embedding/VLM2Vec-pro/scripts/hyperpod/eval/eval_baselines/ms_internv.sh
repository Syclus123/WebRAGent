export PYTHONPATH="/home/mingyi/AI-Projects/VLM2Vec-pro:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=6 python3 eval_video_mret.py \
  --encode_output_path eval_results/eval_baselines_results/internv/ms/rest_of_ms \
  --model_backbone internvideo2 \
  --model_type internvideo2 \
  --model_name /home/ziyan/VLM2Vec-pro/src/model/vlm_backbone/internvideo2/ \
  --pooling eos --normalize True --per_device_eval_batch_size 16 --dataloader_num_workers 8 --resize_use_processor true --dataset_config configs/data_configs/darth_eval/test.yaml
