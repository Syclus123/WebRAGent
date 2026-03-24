#!/bin/bash

source /root/.bashrc
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda_env=/root/miniconda3/envs/vlm2vec/
conda activate $conda_env

echo "conda location: $(which conda)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

export NPROC_PER_NODE=8
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "Launching on node $RANK of $WORLD_SIZE; $NPROC_PER_NODE GPUs/node"

export WANDB_DISABLED=false
export WANDB_PROJECT=vlm2vec-gui
export WANDB_API_KEY=1ab16c43d60aa3be4c94202d3327bf636cb94c8b
export HUGGING_FACE_HUB_TOKEN=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
export WANDB_RUN_GROUP=train
export EXP_NAME=qwen2_vl-lite_full-lora8-bsz128x8x2-interleave_0.2-lr5e5-max_step_256-warmup_12-uigraph_select_0.5-lm_skip_all-vis_skip_all

export WANDB_NAME="${EXP_NAME}/node${RANK}"
export EXP_DIR="/data/embedding/experiments/train/${EXP_NAME}/node${RANK}"
export WANDB_DIR=$EXP_DIR
echo $EXP_DIR

mkdir -p $EXP_DIR/wandb
rm -rf $EXP_DIR/wandb/*

export MODEL_PATH=/data/models/Qwen2-VL-TokenSelection-2B
export MODEL_BACKBONE=qwen2_vl_tokenselection
export DATA_CONFIG=/data/embedding/VLM2Vec-pro/configs/data_configs/gui/lite/train.yaml

export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NPROC_PER_NODE - 1)))

cd /data/embedding/VLM2Vec-pro

TORCHRUN_ARGS=(
  --nnodes "$WORLD_SIZE"
  --nproc_per_node "$NPROC_PER_NODE"
  --node_rank "$RANK"
  --rdzv_id "train_${EXP_NAME}"
  --rdzv_backend c10d
  --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}"
  --max_restarts 0
)

cmd="torchrun ${TORCHRUN_ARGS[@]} train.py \
  --lora --lora_r 8 \
  --model_name $MODEL_PATH --model_backbone $MODEL_BACKBONE \
  --bf16 --pooling eos --normalize True --temperature 0.02 \
  --dataloader_num_workers 1 \
  --dataset_config $DATA_CONFIG \
  --run_name $EXP_NAME --output_dir $EXP_DIR \
  --grad_cache True --per_device_train_batch_size 128 \
    --gc_q_chunk_size 1 --gc_p_chunk_size 1 \
  --lr_scheduler_type linear --learning_rate 5e-5 \
  --max_steps 256 --warmup_ratio 0.05 \
  --save_steps 1 --save_total_limit 10 --logging_steps 1 \
  --interleave_batch_size 0.2 \
  --save_safetensors False --remove_unused_columns False \
  --max_len 65536 \
  --uigraph_use True --uimask_ratio 0.5 \
  --lm_skip_layer \"[1,28,1]\" --vis_skip_layer \"[1,32,1]\" \
  --resize_use_processor True --resume_from auto \
  --report_to wandb \
  2>&1 | tee $EXP_DIR/train.log"

echo $cmd
eval $cmd
