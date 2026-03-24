#!/bin/bash

# Activate the embedding environment
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "embedding" ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate embedding
fi

echo "conda location: $(which conda)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

# Set environment variables - adapt to current environment
export PYTHONPATH=/home/ubuntu/data/csb/Embedding/VLM2Vec-pro:$PYTHONPATH

export EXP_DIR=/home/ubuntu/data/csb/Embedding/VLM2Vec-pro/output/Online-Mind2Web-steps # Output directory
export MODEL_BACKBONE=qwen2_vl_tokenselection
export MODEL_PATH=/data/models/Qwen2-VL-TokenSelection-2B  # Please adjust to actual model path
export CKPT_PATH=/data/embedding/experiments/train/qwen2_vl-lite_full-lora8-bsz128x8x2-interleave_0.2-lr5e5-max_step_256-warmup_12-uigraph_select_0.5-lm_skip_all-vis_skip_all/huggingface  # Please adjust to actual checkpoint path
export DATA_CONFIG=/home/ubuntu/data/csb/Embedding/VLM2Vec-pro/configs/data_configs/gui/lite/cand/embedding.yaml

# Create output directory
mkdir -p $EXP_DIR

cd /home/ubuntu/data/csb/Embedding/VLM2Vec-pro

# For testing, use single GPU first
export CUDA_VISIBLE_DEVICES=0

echo "Starting step-wise embedding computation..."
echo "Output directory: $EXP_DIR"

cmd=(
  python3 compute_new.py \
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
    --per_device_eval_batch_size 1 \
)

echo "Running command:"
echo "${cmd[@]}"
echo ""

"${cmd[@]}" 2>&1 | tee "$EXP_DIR/eval_steps.log"

echo ""
echo "Computation completed. Check log at: $EXP_DIR/eval_steps.log" 