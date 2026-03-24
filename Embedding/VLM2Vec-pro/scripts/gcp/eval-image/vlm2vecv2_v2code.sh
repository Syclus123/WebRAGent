#!/bin/bash
source /mnt/disks/embedding/basics/.bashrc

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nccl2/lib:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:/mnt/disks/embedding/envs/vlm2vec/bin:/usr/local/bin:/mnt/disks/embedding/envs/vlm2vec/bin/:/mnt/disks/embedding/envs/vlm2vec/lib/python3.10/site-packages:$PATH
echo "conda location: $(which conda)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

export HF_DATASETS_CACHE=/mnt/disks/embedding/data/huggingface/dataset
export HF_HOME=/mnt/disks/embedding/data/huggingface/hub
export HUGGING_FACE_HUB_TOKEN=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

cd /mnt/disks/embedding/projects/VLM2Vec-pro/

BATCH_SIZE=8
#MODEL_NAME="TIGER-Lab/VLM2Vec-Qwen2VL-2B"
#OUTPUT_PATH="/mnt/disks/embedding/exps/vlm2vec/baselines/vlm2vec-qwen2vl-2b/image-v2code/"
#MODEL_NAME="TIGER-Lab/VLM2Vec-Qwen2VL-7B"
#OUTPUT_PATH="/mnt/disks/embedding/exps/vlm2vec/baselines/vlm2vec-qwen2vl-7b/image-v2code/"
MODEL_NAME="VLM2Vec/VLM2Vec-V2.0"
OUTPUT_PATH="/mnt/disks/embedding/exps/vlm2vec/baselines/vlm2vec-qwen2vl-v2.0-2b/image-v2code/"
#MODEL_NAME="VLM2Vec/VLM2Vec-V2.1"
#OUTPUT_PATH="/mnt/disks/embedding/exps/vlm2vec/baselines/vlm2vec-qwen2vl-v2.1-2b/image-v2code/"

#GPU_IDS=(0 2 3 4 6 7)
GPU_IDS=(0 1 2 3 4 5 6 7)
#GPU_IDS=(0)
NUM_GPUS=${#GPU_IDS[@]}  # Get the number of GPUs based on the array length

# Counter for tracking GPU assignment
gpu_index=0


print_success() {
    echo -e "\e[32mSUCCESS: $1\e[0m"
}
print_error() {
    echo -e "\e[31mERROR: $1\e[0m"
}
print_warning() {
    echo -e "\e[33mWARNING: $1\e[0m"
}
# Function to run an experiment on a specific GPU
run_experiment() {
    local checkpoint_path=$1
    local output_path=$2
    local gpu_id=$3

    local ckpt_name=$(basename "$checkpoint_path")
    local exp_dir=$(dirname "$checkpoint_path")
    local exp_basename=$(basename "$exp_dir")
    echo $gpu_id
    echo $exp_basename
    mkdir -p $output_path

    # Build the command string
    cmd="CUDA_VISIBLE_DEVICES=$gpu_id torchrun --nproc_per_node=1 --master_port=2227$gpu_id --max_restarts=0 eval.py --pooling eos --normalize true --per_device_eval_batch_size $BATCH_SIZE --resize_use_processor true --model_name $checkpoint_path --dataset_config configs/data_configs/gcp/eval/image.yaml --encode_output_path $output_path --data_basedir /mnt/disks/embedding/data/vlm2vec/MMEB-eval/"
    echo "Running on GPU $gpu_id: $cmd"
    eval $cmd  # Execute the command
}

echo "Processing $MODEL_NAME"
gpu_id=${GPU_IDS[$gpu_index]}
run_experiment "$MODEL_NAME" "$OUTPUT_PATH" $gpu_id &

# Increment GPU index and reset after reaching the maximum number of GPUs
gpu_index=$(( (gpu_index + 1) % NUM_GPUS ))

# If we completed a batch (all GPUs have been used), wait for the batch to finish
if [ $gpu_index -eq 0 ]; then
    echo "Waiting for the current batch of jobs to finish..."
    wait  # Wait for all background jobs to finish before starting the next batch
fi

# Wait for any remaining background jobs before exiting
wait
echo "All batches are complete."
