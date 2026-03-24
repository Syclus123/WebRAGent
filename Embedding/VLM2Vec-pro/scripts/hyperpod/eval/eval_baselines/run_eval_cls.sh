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
#  configs/data_configs/eval/video_cls/breakfast.yaml
#  configs/data_configs/eval/video_cls/hmdb51.yaml
  configs/data_configs/eval/video_cls/k700.yaml
  configs/data_configs/eval/video_cls/ssv2.yaml
  configs/data_configs/eval/video_cls/ucf101.yaml
)

# 👇 Format: model_name|shortname|backbone
BASELINE_CONFIGS=(
#  "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct|gme2b|gme"
#  "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct|gme7b|gme"
  "code-kunkun/LamRA-Ret|lamra|lamra"
#  "vidore/colpali-v1.3|colpali|colpali"
)

GPU_IDS=(0 1 2 3 4 5 6 7)
GPU_IDS=(1)
NUM_GPUS=${#GPU_IDS[@]}
gpu_index=0
BATCH_SIZE=16

run_experiment() {
    local model_name=$1
    local shortname=$2
    local backbone=$3
    local yaml_path=$4
    local gpu_id=$5
    local config_name
    config_name=$(basename "$yaml_path" .yaml)
    local encode_dir="/fsx/home/ruimeng/runs/v3vec-baseline/${shortname}/video_cls"
    mkdir -p "$encode_dir"

    echo -e "\e[36mLaunching $shortname on task $config_name using GPU $gpu_id\e[0m"
    CUDA_VISIBLE_DEVICES=$gpu_id python3 eval_video_cls.py \
        --encode_output_path "$encode_dir" \
        --model_name "$model_name" \
        --model_backbone "$backbone" \
        --dataset_config "$yaml_path" \
        --per_device_eval_batch_size "$BATCH_SIZE" \
        2>&1 | tee "${encode_dir}/${config_name}_gpu${gpu_id}.log"
}

for config in "${BASELINE_CONFIGS[@]}"; do
    IFS='|' read -r model_name shortname backbone <<< "$config"

    for yaml_path in "${TASK_CONFIGS[@]}"; do
        gpu_id=${GPU_IDS[$gpu_index]}
        run_experiment "$model_name" "$shortname" "$backbone" "$yaml_path" "$gpu_id" &
        gpu_index=$(( (gpu_index + 1) % NUM_GPUS ))

        if [ "$gpu_index" -eq 0 ]; then
            echo "🔄 Waiting for current batch of jobs to finish..."
            wait
        fi
    done
done

wait
echo -e "\e[32m✅ All baseline evaluations completed.\e[0m"
