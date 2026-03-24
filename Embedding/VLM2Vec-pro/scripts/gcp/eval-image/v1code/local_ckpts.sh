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
# Define the list of specific experiment directories you want to process
SUBSET_LIST=(
  "ImageNet-1K N24News HatefulMemes VOC2007 SUN397 A-OKVQA MSCOCO"
  "Place365 ImageNet-A ImageNet-R ObjectNet Country211 OK-VQA RefCOCO"
  "DocVQA InfographicsVQA ChartQA NIGHTS FashionIQ"
  "ScienceQA Visual7W VizWiz GQA TextVQA VisDial"
  "CIRR VisualNews_t2i VisualNews_i2t MSCOCO_t2i MSCOCO_i2t "
  "Wiki-SS-NQ"
  "WebQA OVEN EDIS"
  "RefCOCO-Matching Visual7W-Pointing"
)
BATCH_SIZE=32
EXPERIMENTS_LIST=(
     # qwen unified data
#    "/mnt/disks/embedding/exps/vlm2vec/train/qwen2vl_2B.mmeb.lora16.bs1024pergpu128.GCq8p8.NormTemp002.lr5e5.step2kwarm100.8H100/checkpoint-2000/"
    "/mnt/disks/embedding/exps/vlm2vec/train/qwen2vl_2B.mmeb-visdoc.lora16.bs1024pergpu128.GCq8p8.NormTemp002.lr5e5.step2kwarm100.8H100/checkpoint-2000/"
)

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
    local gpu_id=$2
    local subset_id=$3

    local ckpt_name=$(basename "$checkpoint_path")
    local exp_dir=$(dirname "$checkpoint_path")
    local exp_basename=$(basename "$exp_dir")
    echo $gpu_id $subset_id
    echo $exp_basename

    # Build the command string
    cmd="CUDA_VISIBLE_DEVICES=$gpu_id torchrun --nproc_per_node=1 --master_port=2229$gpu_id --max_restarts=0 eval_image.py --model_name $checkpoint_path --encode_output_path $checkpoint_path/eval $lora $max_len_arg --pooling eos --normalize True --dataset_name TIGER-Lab/MMEB-eval --subset_name ${SUBSET_LIST[$subset_id]} --dataset_split test --per_device_eval_batch_size $BATCH_SIZE --image_dir /mnt/disks/embedding/data/vlm2vec/MMEB-eval/image-tasks/"

    echo "Running on GPU $gpu_id: $cmd"
    eval $cmd  # Execute the command
}


# Loop through the list of specific experiment checkpoint directories
for checkpoint_path in "${EXPERIMENTS_LIST[@]}"; do
    echo "Processing $checkpoint_path"
    if [ -d "$checkpoint_path" ]; then
        # Get the current GPU ID from the GPU_IDS array

        for subset_id in "${!SUBSET_LIST[@]}"; do
            gpu_id=${GPU_IDS[$gpu_index]}
            # Run the experiment on the current GPU
#            echo $gpu_id $subset_id
            run_experiment "$checkpoint_path" $gpu_id $subset_id &

            # Increment GPU index and reset after reaching the maximum number of GPUs
            gpu_index=$(( (gpu_index + 1) % NUM_GPUS ))

            # If we completed a batch (all GPUs have been used), wait for the batch to finish
            if [ $gpu_index -eq 0 ]; then
                echo "Waiting for the current batch of jobs to finish..."
                wait  # Wait for all background jobs to finish before starting the next batch
            fi
        done
    else
        echo "Checkpoint directory $checkpoint_path does not exist"
    fi
done

# Wait for any remaining background jobs before exiting
wait
echo "All batches are complete."
