#!/bin/bash

source /root/.bashrc
# Initialize micromamba in the shell
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
# Activate the vlm2vec environment
env=/root/miniconda3/envs/vlm2vec/
conda activate $env

echo "conda location: $(which conda)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

export HUGGING_FACE_HUB_TOKEN=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
export PYTHONPATH=/data/embedding/VLM2Vec-pro:$PYTHONPATH

export EXP_DIR=/data/embedding/experiments/eval/VLM2Vec-V2.2
export MODEL_BACKBONE=qwen2_vl
export MODEL_PATH=/data/models/Qwen2-VL-2B-Instruct
export CKPT_PATH=/data/models/VLM2Vec-V2.2
export DATA_CONFIG=/data/embedding/VLM2Vec-pro/configs/data_configs/gui/lite/eval_all.yaml

cd /data/embedding/VLM2Vec-pro

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cmd=(
  torchrun --nproc_per_node=8 --master_port=10086 --max_restarts=0 eval_gui.py \
    --pooling eos \
    --normalize True \
    --encode_output_path $EXP_DIR \
    --model_name $MODEL_PATH \
    --model_backbone $MODEL_BACKBONE \
    --checkpoint_path $CKPT_PATH \
    --dataset_config $DATA_CONFIG \
    --lora True \
    --resize_use_processor True \
    --max_len 65536 \
    --dataloader_num_workers 2 \
    --per_device_eval_batch_size 6 \
)

echo "${cmd[@]}"
"${cmd[@]}" 2>&1 | tee "$EXP_DIR/eval.log"
