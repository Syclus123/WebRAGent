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
export PYTHONPATH=/data/code/VLM2Vec-pro:$PYTHONPATH

export EXP_DIR=/data/output/Online-Mind2Web # to revise
export MODEL_BACKBONE=qwen2_vl_tokenselection
export MODEL_PATH=/data/models/Qwen2-VL-TokenSelection-2B
export CKPT_PATH=/data/embedding/experiments/train/qwen2_vl-lite_full-lora8-bsz128x8x2-interleave_0.2-lr5e5-max_step_256-warmup_12-uigraph_select_0.5-lm_skip_all-vis_skip_all/huggingface
export DATA_CONFIG=/data/code/VLM2Vec-pro/configs/data_configs/gui/lite/cand/embedding.yaml # to revise

cd /data/code/VLM2Vec-pro

export CUDA_VISIBLE_DEVICES=0,1 # to revise

cmd=(
  torchrun --nproc_per_node=2 --master_port=10086 --max_restarts=0 compute_embedding.py \
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
    --per_device_eval_batch_size 2 \
)

echo "${cmd[@]}"
"${cmd[@]}" 2>&1 | tee "$EXP_DIR/eval.log"
