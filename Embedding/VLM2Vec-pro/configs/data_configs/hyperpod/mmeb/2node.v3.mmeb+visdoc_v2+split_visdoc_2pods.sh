#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --job-name=qwen2vl_2B-003-x-2pods.mmeb20_vidore1_videohound2_mteb15-v1.qwenresize.lora8.bs2048pergpu128ib64.GCq8p8.NormTemp002.lr5e5.step5kwarm200.maxlen2k.16H100
#SBATCH --nodes=2             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8   # This needs to match Trainer(devices=...)
#SBATCH --time=4-0:00:00
#SBATCH --partition=ml.p5.48xlarge-low
#SBATCH --account=low-pri
#SBATCH --output /fsx/home/ruimeng/runs/mmeb/%j/slurm_output.out
#SBATCH --error /fsx/home/ruimeng/runs/mmeb/%j/slurm_output.err

# Enable error handling
set -e
timestamp=$(date +%Y%m%d_%H%M%S)


source /fsx/home/ruimeng/.bashrc
eval "$(/fsx/home/ruimeng/envs/miniconda3/bin/conda shell.bash hook)"
conda_env=/fsx/home/ruimeng/envs/vlm2vec
conda activate $conda_env
echo "conda location: $(which conda)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

export LD_LIBRARY_PATH=/usr/local/cuda-12.1/targets/x86_64-linux/include/:/fsx/home/ruimeng/.local/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export PATH=/fsx/home/ruimeng/envs/vlm2vec/bin:/fsx/home/ruimeng/envs/vlm2vec/lib/python3.10/site-packages:$PATH

export HF_DATASETS_CACHE=/fsx/home/yeliu/xgen-embedding/data/.hfdata_cache
export HF_HOME=/fsx/home/yeliu/xgen-embedding/huggingface_cache
export WANDB_DISABLED=false
export WANDB_PROJECT=unified_embedding
export WANDB_API_KEY=f6b7049d558a297dfafa53312d9119ff0eda8aa6
export WANDB_BASE_URL=https://salesforceairesearch.wandb.io
export HUGGING_FACE_HUB_TOKEN=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
export WANDB_PROJECT=mmeb
export WANDB_RUN_GROUP=train

export EXP_NAME=qwen2vl_2B-003-x-2pods.mmeb+video_v2+split_visdoc-v1.qwenresize.lora16.IB128.bs2048pergpu128ib64.GCq8p8.NormTemp002.lr5e5.step5kwarm200.maxlen2k.16H100
export EXP_DIR=/fsx/home/yeliu/runs/mmeb/$EXP_NAME
export WANDB_NAME=$EXP_NAME
export WANDB_DIR=$EXP_DIR
echo $EXP_DIR

mkdir -p $EXP_DIR/wandb
rm -rf $EXP_DIR/wandb/*
export IMAGE_DIR=/fsx/sfr/data/MMEB/MMEB-train

cd /fsx/home/yeliu/vlm2vec-pro
srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$HOSTNAME:29500 --max_restarts=0 train.py --lora --lora_r 16 --model_name Qwen/Qwen2-VL-2B-Instruct --bf16 --pooling eos --normalize True --temperature 0.02 --dataloader_num_workers 8 --dataset_config configs/data_configs/mmeb/mmeb+video_v2+split_visdoc.yaml --image_dir $IMAGE_DIR --run_name $EXP_NAME --output_dir $EXP_DIR --grad_cache True --per_device_train_batch_size 128 --gc_q_chunk_size 6 --gc_p_chunk_size 6 --lr_scheduler_type linear --learning_rate 5e-5 --max_steps 5000 --warmup_steps 100 --save_steps 50 --logging_steps 1 --save_safetensors False --remove_unused_columns False --resume_from auto --resize_use_processor true --max_len 1536 --interleave_batch_size 128 --ddp_timeout 14400 --ignore_data_skip true 2>&1 | tee $EXP_DIR/train.log