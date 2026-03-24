export PYTHONPATH="/home/mingyi/AI-Projects/VLM2Vec-pro:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=2 python3 eval_video_cls.py \
 --pooling eos --normalize True --per_device_eval_batch_size 4 --resize_use_processor true \
 --model_type lamra --model_backbone lamra \
 --model_name code-kunkun/LamRA-Ret \
 --encode_output_path eval_results/eval_baselines_results/eval_lamra_cls \
 --dataset_config configs/data_configs/darth_eval/video_cls.yaml
