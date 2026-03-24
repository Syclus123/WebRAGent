#!/bin/bash
source /fsx/home/ruimeng/.bashrc
eval "$(/fsx/home/ruimeng/envs/miniconda3/bin/conda shell.bash hook)"
conda_env=/fsx/home/ruimeng/envs/vlm2vec
conda activate $conda_env
echo "conda location: $(which conda)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

CUDA_VISIBLE_DEVICES=2 python3 eval_video_ret.py --dataset_config configs/data_configs/eval/video_ret.yaml \
 --pooling eos --normalize True --per_device_eval_batch_size 4 --resize_use_processor false --image_resolution high \
  --lora --model_name Qwen/Qwen2-VL-7B-Instruct --checkpoint_path TIGER-Lab/VLM2Vec-Qwen2VL-7B \
  --encode_output_path /fsx/home/ruimeng/runs/v3vec-baseline/VLM2Vec-Qwen2VL-7B/video/