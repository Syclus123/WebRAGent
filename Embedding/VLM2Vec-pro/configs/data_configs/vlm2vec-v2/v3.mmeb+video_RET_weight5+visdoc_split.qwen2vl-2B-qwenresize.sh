#!/bin/bash
#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --job-name=qwen2vl_2B.mmeb+video_RET_weight5.qwenresize.lora16.bs512pergpu64.GCq8p8.NormTemp002.lr2e5.step2kwarm100.8H100
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:8
#SBATCH --exclusive # job has exclusive use of the resource, no sharing
#SBATCH --wait-all-nodes=1
#SBATCH --time=4-0:00:00
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --account=xgen-sales
#SBATCH --output /fsx/home/yeliu/runs/mmeb/%x/slurm_output.out
#SBATCH --error /fsx/home/yeliu/runs/mmeb/%x/slurm_output.err

# Enable error handling
set -e
timestamp=$(date +%Y%m%d_%H%M%S)

# Set distributed variables from SLURM environment
export WORLD_SIZE=$SLURM_NNODES
export NPROC_PER_NODE=${SLURM_GPUS_ON_NODE:-8}
export RANK=$SLURM_NODEID
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=19887

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "RANK: $RANK"

source /fsx/home/ruimeng/.bashrc
eval "$(/fsx/home/ruimeng/envs/miniconda3/bin/conda shell.bash hook)"
conda_env=/fsx/home/ruimeng/envs/vlm2vec
conda activate $conda_env
echo "conda location: $(which conda)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"


export LD_LIBRARY_PATH=/usr/local/cuda-12.1/targets/x86_64-linux/include/:/fsx/home/ruimeng/.local/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export PATH=/fsx/home/ruimeng/envs/vlm2vec/bin:/fsx/home/ruimeng/envs/vlm2vec/lib/python3.10/site-packages:$PATH


export HF_DATASETS_CACHE=/fsx/home/yeliu/xgen-embedding/data/.hfdata_cache
export HF_HOME=/fsx/home/yeliu/xgen-embedding/huggingface_cache
export WANDB_DISABLED=false
export WANDB_PROJECT=unified_embedding
export WANDB_API_KEY=local-8245e2785aa62f175259f5f7543b395dac1b2800
export WANDB_BASE_URL=https://salesforceairesearch.wandb.io
export HUGGING_FACE_HUB_TOKEN=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
export WANDB_PROJECT=mmeb
export WANDB_RUN_GROUP=train
export EXP_NAME=qwen2vl_2B.mmeb+video_RET_weight5.qwenresize.lora16.bs512pergpu64.GCq8p8.NormTemp002.lr2e5.step2kwarm100.8H100

export WANDB_NAME=$EXP_NAME
export EXP_DIR=/fsx/home/yeliu/runs/mmeb/$EXP_NAME
export WANDB_DIR=$EXP_DIR
echo $EXP_DIR

mkdir -p $EXP_DIR/wandb
rm -rf $EXP_DIR/wandb/*
export IMAGE_DIR=/fsx/sfr/data/MMEB/MMEB-train

cd /fsx/home/yeliu/vlm2vec-pro
declare -a TORCHRUN_ARGS=(
    # change this to match the number of gpus per node:
    --nproc_per_node=8 \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$(hostname) \
)
srun -l torchrun "${TORCHRUN_ARGS[@]}" \
 train.py --lora --lora_r 8 --model_name Qwen/Qwen2-VL-2B-Instruct --bf16 --pooling eos --normalize True --temperature 0.02 --dataloader_num_workers 2 --dataset_config configs/data_configs/vlm2vec-v2/mmeb+video_v4+split_visdoc.yaml --image_dir $IMAGE_DIR --run_name $EXP_NAME --output_dir $EXP_DIR --grad_cache True --per_device_train_batch_size 64 --gc_q_chunk_size 4 --gc_p_chunk_size 4 --lr_scheduler_type linear --learning_rate 5e-5 --max_steps 5000 --warmup_steps 100 --save_steps 50 --logging_steps 1 --save_safetensors False --remove_unused_columns False --resume_from auto --resize_use_processor true --max_len 1536 --interleave_batch_size 64 --ddp_timeout 14400 --ignore_data_skip true 2>&1 | tee $EXP_DIR/train.log