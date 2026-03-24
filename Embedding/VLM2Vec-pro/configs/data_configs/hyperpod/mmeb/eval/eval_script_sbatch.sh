#!/bin/bash
#SBATCH --job-name=eval_checkpoints    # Job name
#SBATCH --nodes=1           # Number of nodes
#SBATCH --gres=gpu:8        # number of gpus per node
#SBATCH --ntasks=1          # Number of tasks
#SBATCH --time=4-0:00:00      # Time limit
#SBATCH --output=/fsx/home/xyang/eval_checkpoints.log # Output file
#SBATCH --partition=ml.p5.48xlarge
#SBATCH --account xgen-sales

# Set up Conda
export CONDA_ROOT=/fsx/hyperpod/conda
source $CONDA_ROOT/etc/profile.d/conda.sh

# Load necessary environment
conda activate /fsx/home/xyang/miniconda3/bin/zero || { echo "Failed to activate Conda environment"; exit 1; }

# Print environment information for debugging
echo "Python path: $(which python)"
echo "Conda environment: $CONDA_PREFIX"
# debugging flags (optional)
#export NCCL_DEBUG=INFO
#export PYTHONFAULTHANDLER=1

# Change to the benchmark directory
cd /fsx/home/xyang/vidore-benchmark

# Define checkpoint paths as an array
checkpoint_paths=(
    "/fsx/home/yeliu/runs/mmeb/qwen2vl_2B.mmeb20.qwenresize.lora8.bs1024pergpu128.GCq8p8.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-2000"
    "/fsx/home/yeliu/runs/mmeb/qwen2vl_2B.video.qwenresize.lora8.bs1024pergpu128.GCq8p8.NormTemp002.lr2e5.step2kwarm100.8H100"
    "/fsx/home/yeliu/runs/mmeb/qwen2vl_2B.visdoc.qwenresize.lora8.bs1024pergpu128.GCq8p8.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-2000"
    "/fsx/home/yeliu/runs/mmeb/qwen2vl_2B.mmeb20+vidore+visrag.qwenresize.lora8.bs1024pergpu128.GCq8p8.NormTemp002.lr2e5.step2kwarm100.8H100"
    "/fsx/home/yeliu/runs/mmeb/qwen2vl_2B.mmeb20+visdoc+video.qwenresize.lora8.bs1024pergpu128.GCq8p8.NormTemp002.lr2e5.step2kwarm100.8H100"
    "/fsx/home/yeliu/runs/mmeb/qwen2vl_2B.mmeb20+visdoc+video.qwenresize.lora16.bs1024pergpu128.GCq8p8.NormTemp002.lr2e5.step2kwarm100.8H100"
    "/fsx/home/yeliu/runs/mmeb/qwen2vl_2B.mmeb20+visdoc+video.qwenresize.lora16.IB128.bs1024pergpu128.GCq8p8.NormTemp002.lr2e5.step2kwarm100.8H100"
    "/fsx/home/yeliu/runs/mmeb/qwen2vl_2B.mmeb20+visdoc+video.qwenresize.lora16.noIB.bs1024pergpu128.GCq8p8.NormTemp002.lr2e5.step2kwarm100.8H100"
)

# Function to run evaluation for a single checkpoint
run_evaluation() {
    local checkpoint_path=$1
    local gpu_id=$2
    local output_path="/fsx/home/xyang"
    # Extract the part of the path after /fsx/home/yeliu/runs/mmeb/ and replace / with _
    local checkpoint_relative_path="${checkpoint_path#/fsx/home/yeliu/runs/mmeb/}"
    local log_name="${checkpoint_relative_path//\//_}"
    local log_file="$output_path/eval_${log_name}.log"

    echo "Starting evaluation on GPU $gpu_id: $checkpoint_path"

    # Run the evaluation script with the current checkpoint on the assigned GPU
    python eval.py --checkpoint_path "$checkpoint_path" --device "cuda:$gpu_id" --output_path "$output_path" --normalize --lora > "$log_file" 2>&1

    echo "Evaluation complete for: $checkpoint_path (Log file: $log_file)"
}

# Detect number of available GPUs
# This is a placeholder - you may need to adjust based on your actual environment
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPUs"

# Launch evaluations in parallel with GPU assignment
for i in "${!checkpoint_paths[@]}"; do
    gpu_id=$((i % NUM_GPUS))
    run_evaluation "${checkpoint_paths[$i]}" "$gpu_id" &

    # Optional: add a small delay to stagger launches
    sleep 2
done

# Wait for all background processes to complete
wait

echo "All checkpoints evaluated successfully!"

