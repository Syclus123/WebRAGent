#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --nodes=4 # number of nodes to use, 4 p4d(e) = 32 A100 GPUs
#SBATCH --gres=gpu:8 # number of gpus per node
#SBATCH --job-name=e4d # name of your job
#SBATCH --exclusive # job has exclusive use of the resource, no sharing
#SBATCH --wait-all-nodes=1
#SBATCH --partition=ml.p5.48xlarge
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null


###########################
## Environment Variables ##
###########################


export NCCL_ASYNC_ERROR_HANDLING=1

export NCCL_PROTO="simple"
export NCCL_SOCKET_IFNAME="^lo,docker"
export RDMAV_FORK_SAFE=1
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_DEBUG_SUBSYS=off
export NCCL_DEBUG="INFO"
export SM_NUM_GPUS=8
export GPU_NUM_DEVICES=8
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0



## TODO
export CUDA_DEVICE_MAX_CONNECTIONS=1


export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:1024'


export WANDB_API_KEY=local-14b1aa5d31991453be52b57875994069f03e36aa
wandb enabled


declare -a TORCHRUN_ARGS=(
    # change this to match the number of gpus per node:
    --nproc_per_node=8 \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$(hostname) \
)

export CUDA_HOME="/usr/local/cuda-12.1"
export LIBRARY_PATH="/usr/local/cuda-12.1/lib64:$LIBRARY_PATH"

log_prefix=logs
exp_name=exp_3048

srun -l torchrun "${TORCHRUN_ARGS[@]}" \
    -m axolotl.cli.train /fsx/home/bpang/axolotl_1936/axolotl/train_configs/exp_3048_qwen_coder_instruct.yml \
2>&1 | tee ${log_prefix}/${exp_name}.log
