#!/bin/bash
source /fsx/home/ruimeng/.bashrc
eval "$(/fsx/home/ruimeng/envs/miniconda3/bin/conda shell.bash hook)"
conda_env=/fsx/home/ruimeng/envs/vlm2vec
conda activate $conda_env
echo "conda location: $(which conda)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

export HF_DATASETS_CACHE=/fsx/home/ruimeng/data/.hfdata_cache
export HF_HOME=/fsx/home/ruimeng/data/.hfmodel_cache/

cd /fsx/home/ruimeng/project/VLM2Vec/
CUDA_VISIBLE_DEVICES=2 python3 eval_mmeb.py \
 --pooling eos --normalize True --per_device_eval_batch_size 16 --resize_use_processor true \
 --dataset_name TIGER-Lab/MMEB-eval --subset_name ImageNet-1K --dataset_split test \
 --image_dir /fsx/sfr/data/MMEB/MMEB_test/MMEB_Test_1K_New/images \
 --model_backbone lamra --model_name code-kunkun/LamRA-Ret \
 --encode_output_path /fsx/home/ruimeng/runs/v3vec-baseline/LamRA-Ret/mmeb-test/
