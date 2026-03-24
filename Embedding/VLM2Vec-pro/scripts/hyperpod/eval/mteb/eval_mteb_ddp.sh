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

cd /fsx/home/ruimeng/project/VLM2Vec/adhoc/eval_mteb
torchrun --nproc_per_node=8 --master_port=2277 --max_restarts=0 run_mteb.py --nproc_per_node=8 --master_port=22433 --max_restarts=0 adhoc/eval_mteb/run_mteb.py --model-name Qwen/Qwen2-VL-2B-Instruct --checkpoint-path /fsx/home/ruimeng/runs/mmeb/mmeb006-qwen2vl-2B-4-lateprocess-high_res.lora8.mmeb20_sub100k.bs1024pergpu128.GCq4p4.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-2000/ --eval-output-dir /fsx/home/ruimeng/runs/mmeb/mmeb006-qwen2vl-2B-4-lateprocess-high_res.lora8.mmeb20_sub100k.bs1024pergpu128.GCq4p4.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-2000/eval_mteb --pooling last --model-dtype fp16 --batch-size-per-device 128 --max_length 512 --lora --normalize --prompt_family e5mistral
