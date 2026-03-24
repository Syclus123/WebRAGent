export PYTHONPATH=/home/mingyi/AI-Projects/VLM2Vec-pro:$PYTHONPATH
EXP_NAME='qwen2vl_2B-002-6.mmeb20_vidore1_videohound2_mteb15-v2-cap100k-rerun.qwenresize.lora16.bs1024pergpu128-ib64-droplast.GCq8p8.NormTemp002.lr5e5.step5kwarm200.maxlen2k.8H100_checkpoint-5000'
EXP_DIR=train_results/$EXP_NAME

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun \
    --nproc_per_node=2 --master_port=2202 --max_restarts=0 train.py \
    --lora --lora_r 8 --model_name Qwen/Qwen2-VL-2B-Instruct --bf16  \
    --pooling eos --normalize True --temperature 0.02 --dataloader_num_workers 8  \
    --dataset_config configs/data_configs/darth/darth_train/train_cls.yaml --run_name $EXP_NAME  \
    --output_dir $EXP_DIR --grad_cache True  \
    --per_device_train_batch_size 16 --gc_q_chunk_size 8 --gc_p_chunk_size 8  \
    --lr_scheduler_type linear --learning_rate 5e-5  \
    --max_steps 2000 --warmup_steps 100 --save_steps 100 --logging_steps 1000  \
    --save_safetensors False --remove_unused_columns False --resume_from auto \
    --interleave_batch_size 0.5 --report_to none  \
    2>&1 | tee $EXP_DIR/train.log