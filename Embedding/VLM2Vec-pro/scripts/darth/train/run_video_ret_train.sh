export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=4,5,6,7,

export IMAGE_DIR=/data/yuepeng/video_retrieval/train/MSR-VTT
export EXP_NAME=test_video_ret
export EXP_DIR=/home/ziyan/$EXP_NAME

torchrun --nproc_per_node=4 --master_port=2207 --max_restarts=0 train.py \
  --lora --lora_r 8 --model_name Qwen/Qwen2-VL-2B-Instruct --bf16 --pooling eos --normalize True --temperature 0.02 \
  --dataloader_num_workers 8 --dataset_config configs/data_configs/darth/darth_train/video_ret.yaml --image_dir $IMAGE_DIR \
  --run_name $EXP_NAME --output_dir $EXP_DIR --grad_cache True --per_device_train_batch_size 128 --gc_q_chunk_size 2 --gc_p_chunk_size 2 \
  --report_to none --lr_scheduler_type linear --learning_rate 5e-5 --max_steps 2000 --warmup_steps 100 --save_steps 50 --logging_steps 1 \
  --save_safetensors False --remove_unused_columns False --resume_from auto --resize_use_processor true --max_len 1536 --interleave_batch_size 64 \
  --ddp_timeout 14400 --ignore_data_skip true 2>&1 | tee /home/ziyan/train.log
