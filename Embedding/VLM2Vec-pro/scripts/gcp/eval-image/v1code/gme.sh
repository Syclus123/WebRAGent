#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nccl2/lib:/usr/local/cuda/extras/CUPTI/lib64
export PATH=/usr/local/cuda/bin:/mnt/disks/embedding/envs/vlm2vec/bin:/usr/local/bin:/usr/bin:/bin:/usr/local/games:/usr/games

export HF_DATASETS_CACHE=/mnt/disks/embedding/data/huggingface/dataset
export HF_HOME=/mnt/disks/embedding/data/huggingface/hub
export WANDB_DISABLED=true

cd /home/ruimeng/project/multimodal/VLM2Vec-pro
cmd="CUDA_VISIBLE_DEVICES=5 python eval_mmeb.py \
    --pooling eos --normalize True --per_device_eval_batch_size 8 --resize_use_processor true \
    --dataset_name TIGER-Lab/MMEB-eval --subset_name ImageNet-1K \
    --dataset_split test --image_dir /home/ziyan/images_test \
     --model_backbone gme --model_name Alibaba-NLP/gme-Qwen2-VL-2B-Instruct \
     --encode_output_path ./runs/v3vec-baseline/gme-Qwen2-VL-2B-Instruct/mmeb-test/"

echo $cmd
eval $cmd
