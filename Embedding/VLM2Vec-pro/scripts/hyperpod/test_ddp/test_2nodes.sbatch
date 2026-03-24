#!/bin/bash
#SBATCH --job-name=ddp-test
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --wait-all-nodes=1
#SBATCH --time=01:00:00
#SBATCH --partition=ml.p5.48xlarge
#SBATCH --account=xgen-sales
#SBATCH --output=%x.out
#SBATCH --error=%x.err

set -e
export NPROC_PER_NODE=${SLURM_GPUS_ON_NODE:-8}
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export RANK=$SLURM_NODEID

echo "SLURM_NNODES: $SLURM_NNODES"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "RANK: $RANK"

#export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
#echo "WORLD_SIZE: $WORLD_SIZE"

# Activate conda
source /fsx/home/ruimeng/.bashrc
eval "$(/fsx/home/ruimeng/envs/miniconda3/bin/conda shell.bash hook)"
conda_env=/fsx/home/ruimeng/envs/vlm2vec
conda activate $conda_env
echo "conda location: $(which conda)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

#echo "srun torchrun \
#    --nnodes=$SLURM_NNODES \
#    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
#    --node_rank=$SLURM_NODEID \
#    --master_addr=$MASTER_ADDR \
#    --master_port=$MASTER_PORT \
#    test_ddp.py"

#cd /fsx/home/ruimeng/project/VLM2Vec/
#srun torchrun \
#    --nnodes=$SLURM_NNODES \
#    --nproc_per_node=8 \
#    --node_rank=$SLURM_NODEID \
#    --master_addr=$MASTER_ADDR \
#    --master_port=$MASTER_PORT \
#    test_ddp.py 2>&1 | tee /fsx/home/ruimeng/project/VLM2Vec/configs/data_configs/visualdoc/train.log

cd /fsx/home/ruimeng/project/VLM2Vec/
declare -a TORCHRUN_ARGS=(
    # change this to match the number of gpus per node:
    --nproc_per_node=8 \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$(hostname) \
)
srun -l torchrun "${TORCHRUN_ARGS[@]}" \
    test_ddp.py 2>&1 | tee /fsx/home/ruimeng/project/VLM2Vec/configs/data_configs/visualdoc/train.log
