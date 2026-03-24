#!/bin/bash
source /fsx/home/ruimeng/.bashrc
eval "$(/fsx/home/ruimeng/envs/miniconda3/bin/conda shell.bash hook)"
conda_env=/fsx/home/ruimeng/envs/vlm2vec
conda activate $conda_env
echo "conda location: $(which conda)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

export LD_LIBRARY_PATH=/usr/local/cuda-12.1/targets/x86_64-linux/include/:/fsx/home/ruimeng/.local/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export PATH=/fsx/home/ruimeng/envs/vlm2vec/bin:/fsx/home/ruimeng/envs/vlm2vec/lib/python3.10/site-packages:$PATH
export NCCL_DEBUG=INFO

export HF_DATASETS_CACHE=/fsx/home/yeliu/xgen-embedding/data/.hfdata_cache
export HF_HOME=/fsx/home/yeliu/xgen-embedding/huggingface_cache
export WANDB_DISABLED=false
export WANDB_PROJECT=unified_embedding
export WANDB_API_KEY=local-8245e2785aa62f175259f5f7543b395dac1b2800
export WANDB_BASE_URL=https://salesforceairesearch.wandb.io
export HUGGING_FACE_HUB_TOKEN=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
export WANDB_PROJECT=mmeb
export WANDB_RUN_GROUP=train
export EXP_NAME=qwen2vl_2B.visdoc.qwenresize.lora8.bs1024pergpu128.GCq8p8.NormTemp002.lr2e5.step2kwarm100.8H100

export WANDB_NAME=$EXP_NAME
export EXP_DIR=/fsx/home/yeliu/runs/mmeb/$EXP_NAME
export WANDB_DIR=$EXP_DIR
echo $EXP_DIR

mkdir -p $EXP_DIR/wandb
rm -rf $EXP_DIR/wandb/*
export IMAGE_DIR=/fsx/sfr/data/MMEB/MMEB-train

cd /fsx/home/yeliu/vlm2vec-pro
cmd="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2207 --max_restarts=0 train.py --lora --lora_r 8 --model_name Qwen/Qwen2-VL-2B-Instruct --bf16 --pooling eos --normalize True --temperature 0.02 --dataloader_num_workers 8 --dataset_config configs/data_configs/mmeb/visdoc.yaml --image_dir $IMAGE_DIR --run_name $EXP_NAME --output_dir $EXP_DIR --grad_cache True --per_device_train_batch_size 128 --gc_q_chunk_size 8 --gc_p_chunk_size 8 --lr_scheduler_type linear --learning_rate 5e-5 --max_steps 2000 --warmup_steps 100 --save_steps 50 --logging_steps 1 --save_safetensors False --remove_unused_columns False --resume_from auto --resize_use_processor true --max_len 1536 --interleave_batch_size 64 --ddp_timeout 14400 --ignore_data_skip true 2>&1 | tee $EXP_DIR/train.log"

echo $cmd
eval $cmd