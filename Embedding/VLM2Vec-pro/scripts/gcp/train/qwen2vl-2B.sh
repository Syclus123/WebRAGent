#!/bin/bash
source /mnt/disks/embedding/basics/.bashrc

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nccl2/lib:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:/mnt/disks/embedding/envs/vlm2vec/bin:/usr/local/bin:/mnt/disks/embedding/envs/vlm2vec/bin/:/mnt/disks/embedding/envs/vlm2vec/lib/python3.10/site-packages:$PATH
echo "conda location: $(which conda)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

export HF_DATASETS_CACHE=/mnt/disks/embedding/data/huggingface/dataset
export HF_HOME=/mnt/disks/embedding/data/huggingface/hub
export WANDB_DISABLED=false
export WANDB_PROJECT=unified_embedding
export WANDB_API_KEY=68a612a214a9e5f1b7ae1e0640c8f06923794b53
export HUGGING_FACE_HUB_TOKEN=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
export WANDB_PROJECT=mmeb
export WANDB_RUN_GROUP=train
#export EXP_NAME="qwen2vl_2B.mmeb.lora16.bs1024pergpu128.GCq8p8.NormTemp002.lr5e5.step2kwarm100.8H100"
export EXP_NAME="qwen2vl_2B.mmeb-visdoc.lora16.bs1024pergpu128.GCq8p8.NormTemp002.lr5e5.step2kwarm100.8H100"

export WANDB_NAME=$EXP_NAME
export EXP_DIR=/mnt/disks/embedding/exps/vlm2vec/train/$EXP_NAME
export WANDB_DIR=$EXP_DIR
echo $EXP_DIR

mkdir -p $EXP_DIR/wandb
rm -rf $EXP_DIR/wandb/*

cd /mnt/disks/embedding/projects/VLM2Vec-pro
cmd="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2207 --max_restarts=0 train.py --lora --lora_r 16 --model_name Qwen/Qwen2-VL-2B-Instruct --bf16 --pooling eos --normalize True --temperature 0.02 --dataloader_num_workers 8 --dataset_config configs/data_configs/gcp/train/image+visdoc4.yaml --run_name $EXP_NAME --output_dir $EXP_DIR --grad_cache True --per_device_train_batch_size 128 --gc_q_chunk_size 8 --gc_p_chunk_size 8 --interleave_batch_size 64 --lr_scheduler_type linear --learning_rate 5e-5 --max_steps 2000 --warmup_steps 100 --save_steps 50 --logging_steps 1 --save_safetensors False --remove_unused_columns False --resume_from auto --report_to wandb 2>&1 | tee $EXP_DIR/train.log"

echo $cmd
eval $cmd
