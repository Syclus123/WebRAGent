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

export HF_DATASETS_CACHE=/fsx/home/ruimeng/data/.hfdata_cache
export HF_HOME=/fsx/home/ruimeng/data/.hfmodel_cache/
export HUGGING_FACE_HUB_TOKEN=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
export TOKENIZERS_PARALLELISM=false

cd /fsx/home/ruimeng/project/VLM2Vec

# 👇 Manually specify the list of task config YAMLs
TASK_CONFIGS=(
  /fsx/home/ruimeng/project/VLM2Vec/configs/data_configs/eval/video_qa/video_qa-activitynetqa.yaml
  /fsx/home/ruimeng/project/VLM2Vec/configs/data_configs/eval/video_qa/video_qa-videomme.yaml
  /fsx/home/ruimeng/project/VLM2Vec/configs/data_configs/eval/video_qa/video_qa-mvbench.yaml
  /fsx/home/ruimeng/project/VLM2Vec/configs/data_configs/eval/video_qa/video_qa-egoschema.yaml
  /fsx/home/ruimeng/project/VLM2Vec/configs/data_configs/eval/video_qa/video_qa-nextqa.yaml
)

# GPU setup
GPU_IDS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPU_IDS[@]}
gpu_index=0

# Model configuration
BASELINE_NAME="Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
SHORTNAME="gme2b"
BACKBONE="gme"
BATCH_SIZE=16
ENCODE_DIR="/fsx/home/ruimeng/runs/v3vec-baseline/${SHORTNAME}/video"
mkdir -p "$ENCODE_DIR"

run_experiment() {
    local yaml_path=$1
    local gpu_id=$2
    local config_name=$(basename "$yaml_path" .yaml)

    echo -e "\e[36mLaunching $SHORTNAME on task $config_name using GPU $gpu_id\e[0m"
    CUDA_VISIBLE_DEVICES=$gpu_id python3 eval_video_qa.py \
        --encode_output_path $ENCODE_DIR \
        --model_name $BASELINE_NAME \
        --model_backbone $BACKBONE \
        --resize_use_processor true \
        --dataset_config $yaml_path \
        --per_device_eval_batch_size $BATCH_SIZE \
        2>&1 | tee "${ENCODE_DIR}/${config_name}_gpu${gpu_id}.log"
}

# 👇 Launch evaluations round-robin over GPUs
for yaml_path in "${TASK_CONFIGS[@]}"; do
    gpu_id=${GPU_IDS[$gpu_index]}
    run_experiment "$yaml_path" "$gpu_id" &
    gpu_index=$(( (gpu_index + 1) % NUM_GPUS ))

    if [ "$gpu_index" -eq 0 ]; then
        echo "🔄 Waiting for current batch to finish before launching next..."
        wait
    fi
done

wait
echo -e "\e[32m✅ All selected task evaluations completed.\e[0m"
