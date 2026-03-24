#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64
export PATH=/home/ruimeng/envs/vlm2vec/bin:/home/ruimeng/miniconda3/condabin:/home/ruimeng/miniconda3/bin:/home/ruimeng/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin:/usr/local/cuda-12.4/bin

export HF_DATASETS_CACHE=/home/ruimeng/.cache/huggingface/dataset
export HF_HOME=/home/ruimeng/.cache/huggingface/hub
export WANDB_DISABLED=true

export EXP_NAME=qwen2_2b-mmeb-test
export EXP_DIR=./runs/darth_train/$EXP_NAME/
cd /home/ruimeng/project/multimodal/VLM2Vec-pro
cmd="CUDA_VISIBLE_DEVICES=6,7 torchrun \
    --nproc_per_node=2 --master_port=2202 --max_restarts=0 train.py \
    --lora --lora_r 8 --model_name Qwen/Qwen2-VL-2B-Instruct --bf16  \
    --pooling eos --normalize True --temperature 0.02 --dataloader_num_workers 8  \
    --dataset_config configs/data_configs/darth_train/mmeb20.yaml --run_name $EXP_NAME  \
    --output_dir $EXP_DIR --grad_cache True  \
    --per_device_train_batch_size 16 --gc_q_chunk_size 8 --gc_p_chunk_size 8  \
    --lr_scheduler_type linear --learning_rate 5e-5  \
    --max_steps 2000 --warmup_steps 100 --save_steps 100 --logging_steps 1  \
    --save_safetensors False --remove_unused_columns False --resume_from auto \
    --interleave_batch_size 0.5 --report_to none  \
    2>&1 | tee $EXP_DIR/train.log"

echo $cmd
eval $cmd
