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

SUBSET_LIST=(
#  "ImageNet-1K N24News HatefulMemes VOC2007 SUN397 A-OKVQA MSCOCO"
#  "Place365 ImageNet-A ImageNet-R ObjectNet Country211 OK-VQA RefCOCO"
#  "DocVQA InfographicsVQA ChartQA NIGHTS FashionIQ"
#  "ScienceQA Visual7W VizWiz GQA TextVQA VisDial"
#  "CIRR VisualNews_t2i VisualNews_i2t MSCOCO_t2i MSCOCO_i2t"
#  "Wiki-SS-NQ"
#  "WebQA OVEN EDIS"
#  "RefCOCO-Matching Visual7W-Pointing"

#  "VisDial"
#  "VisualNews_t2i"
#  "MSCOCO_t2i"
#  "NIGHTS"
#  "Wiki-SS-NQ"
#  "OVEN EDIS"
#  "MSCOCO"
#  "RefCOCO-Matching"


#  "CIRR"
#  "VisualNews_i2t"
#  "MSCOCO_i2t"
#  "WebQA "
#  "FashionIQ"
#  "EDIS"
  "RefCOCO"
#  "Visual7W-Pointing"
)

# Format: model_name|shortname|backbone
BASELINE_CONFIGS=(
#  "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct|gme2b|gme"
#  "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct|gme7b|gme"
#  "code-kunkun/LamRA-Ret|lamra|lamra"
  "vidore/colpali-v1.3|colpali|colpali"
)


BATCH_SIZE=32
GPU_IDS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPU_IDS[@]}
gpu_index=0

run_experiment() {
    local model_name=$1
    local shortname=$2
    local model_backbone=$3
    local gpu_id=$4
    local subset_id=$5

    encode_path="/fsx/home/ruimeng/runs/v3vec-baseline/${shortname}/mmeb/"
    mkdir -p $encode_path

    echo -e "\e[36mLaunching $shortname on subset ${SUBSET_LIST[$subset_id]} using GPU $gpu_id\e[0m"

    CUDA_VISIBLE_DEVICES=$gpu_id python eval_mmeb.py \
        --model_name $model_name \
        --model_backbone $model_backbone \
        --encode_output_path $encode_path \
        --pooling eos --normalize True \
        --dataset_name TIGER-Lab/MMEB-eval \
        --subset_name ${SUBSET_LIST[$subset_id]} \
        --dataset_split test \
        --resize_use_processor true \
        --per_device_eval_batch_size $BATCH_SIZE \
        --image_dir /fsx/sfr/data/MMEB/MMEB_test/MMEB_Test_1K_New/images/ \
        2>&1 | tee "${encode_path}/subset${subset_id}_gpu${gpu_id}.log"
}

for config in "${BASELINE_CONFIGS[@]}"; do
    IFS='|' read -r model_name shortname model_backbone <<< "$config"

    for subset_id in "${!SUBSET_LIST[@]}"; do
        gpu_id=${GPU_IDS[$gpu_index]}
        run_experiment "$model_name" "$shortname" "$model_backbone" "$gpu_id" "$subset_id" &
        gpu_index=$(( (gpu_index + 1) % NUM_GPUS ))

        if [ "$gpu_index" -eq 0 ]; then
            echo "Waiting for the current batch of jobs to finish..."
            wait  # Wait for current batch to finish
        fi
    done
done

wait
echo -e "\e[32m✅ All baseline evaluations completed.\e[0m"
