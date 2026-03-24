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

export HF_DATASETS_CACHE=/fsx/home/yeliu/xgen-embedding/data/.hfdata_cache
export HF_HOME=/fsx/home/yeliu/xgen-embedding/huggingface_cache
export HUGGING_FACE_HUB_TOKEN=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

CUDA_VISIBLE_DEVICES=0 python -m adhoc.eval_mteb.run_mteb --model_name Qwen/Qwen2-VL-2B-Instruct --checkpoint_path /fsx/home/ruimeng/runs/mmeb/qwen2vl_2B-002-3.mmeb20_vidore1_videohound2_mteb15-v1.qwenresize.lora8.bs1024pergpu128.GCq8p8.NormTemp002.lr5e5.step5kwarm200.maxlen2k.8H100/checkpoint-1000/ --eval_output_dir /fsx/home/yeliu/runs/mmeb/qwen2vl_2B-002-3.mmeb20_vidore1_videohound2_mteb15-v1.qwenresize.lora8.bs1024pergpu128.GCq8p8.NormTemp002.lr5e5.step5kwarm200.maxlen2k.8H100/checkpoint-1000/eval_mteb --pooling last --model_dtype fp16 --batch_size_per_device 128 --max_length 512 --lora --normalize --prompt_family e5mistral


#CUDA_VISIBLE_DEVICES=0 python -m adhoc.eval_mteb.run_mteb --model_name Qwen/Qwen2-VL-2B-Instruct --checkpoint_path /fsx/home/yeliu/runs/mmeb/mteb-qwen2vl-2B-3.1-lateprocess-mid_res-flashattn-leftpadforreal.lora8.mmeb20_sub100k.bs1024pergpu128.GCq16p16.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-1000/ --eval_output_dir /fsx/home/yeliu/runs/mmeb/mteb-qwen2vl-2B-3.1-lateprocess-mid_res-flashattn-leftpadforreal.lora8.mmeb20_sub100k.bs1024pergpu128.GCq16p16.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-1000/eval_mteb_no_prompt --pooling last --model_dtype fp16 --batch_size_per_device 128 --max_length 512 --lora --normalize

