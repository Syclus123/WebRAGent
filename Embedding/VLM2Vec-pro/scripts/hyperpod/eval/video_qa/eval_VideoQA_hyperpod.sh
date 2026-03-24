#!/bin/bash
source /fsx/home/ruimeng/.bashrc
eval "$(/fsx/home/ruimeng/envs/miniconda3/bin/conda shell.bash hook)"
conda_env=/fsx/home/ruimeng/envs/vlm2vec
conda activate $conda_env
echo "conda location: $(which conda)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"


export LD_LIBRARY_PATH=/fsx/home/xyang/embed-env/bin/python:/usr/local/cuda-12.1/targets/x86_64-linux/include/:/fsx/home/ruimeng/.local/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export PATH=/fsx/home/ruimeng/envs/vlm2vec/bin:/fsx/home/ruimeng/envs/vlm2vec/lib/python3.10/site-packages:$PATH

export HF_DATASETS_CACHE=/fsx/home/ruimeng/data/.hfdata_cache
export HF_HOME=/fsx/home/ruimeng/data/.hfmodel_cache/
export HUGGING_FACE_HUB_TOKEN=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

CKPT_DIR="/fsx/home/ruimeng/runs/mmeb/qwen2vl_2B-002-7-1.mmeb20_visrag2_videohound2f8_NOmteb-v3-cap100k.qwenresize.lora16.bs1024pergpu128-ib64-droplast.GCq8p8.NormTemp002.lr5e5.step2kwarm100.maxlen1k5.8H100/checkpoint-2000/"

cd /fsx/home/ruimeng/project/VLM2Vec/
CUDA_VISIBLE_DEVICES=0 python eval_videoqa.py \
  --lora --model_name "$CKPT_DIR" \
  --encode_output_path "$CKPT_DIR/eval-video" \
  --pooling eos --normalize True --per_device_eval_batch_size 16 --dataloader_num_workers 1 --resize_use_processor true \
  --dataset_config configs/data_configs/eval/video_qa.yaml

