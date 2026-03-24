#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64
export PATH=/home/ruimeng/envs/vlm2vec/bin:/home/ruimeng/miniconda3/condabin:/home/ruimeng/miniconda3/bin:/home/ruimeng/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin:/usr/local/cuda-12.4/bin

export HF_DATASETS_CACHE=/home/ruimeng/.cache/huggingface/dataset
export HF_HOME=/home/ruimeng/.cache/huggingface/hub
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
