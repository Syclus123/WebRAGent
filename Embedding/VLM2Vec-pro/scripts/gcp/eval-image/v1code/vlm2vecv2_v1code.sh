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
#SUBSET_NAMES="ImageNet-1K N24News CIFAR-100 HatefulMemes VOC2007 SUN397 ImageNet-A ImageNet-R ObjectNet Country211 VisDial CIRR FashionIQ VisualNews_t2i VisualNews_i2t MSCOCO_t2i MSCOCO_i2t NIGHTS WebQA Wiki-SS-NQ OVEN EDIS OK-VQA A-OKVQA DocVQA InfographicsVQA ChartQA ScienceQA Visual7W VizWiz GQA TextVQA"
#SUBSET_NAMES="WebQA Wiki-SS-NQ OVEN EDIS"
#SUBSET_NAMES="ImageNet-1K N24News CIFAR-100 HatefulMemes VOC2007 SUN397 ImageNet-A ImageNet-R ObjectNet Country211"
#SUBSET_NAMES="OK-VQA A-OKVQA DocVQA InfographicsVQA ChartQA ScienceQA Visual7W VizWiz GQA TextVQA"
#SUBSET_NAMES="VisDial CIRR FashionIQ VisualNews_t2i VisualNews_i2t MSCOCO_t2i MSCOCO_i2t NIGHTS WebQA Wiki-SS-NQ OVEN EDIS"
#SUBSET_NAMES="MSCOCO RefCOCO-Matching Visual7W-Pointing"

SUBSET_LIST=(
#  "ImageNet-1K N24News HatefulMemes VOC2007 SUN397 Place365 ImageNet-A ImageNet-R ObjectNet Country211"
#  "OK-VQA A-OKVQA DocVQA InfographicsVQA ChartQA ScienceQA Visual7W VizWiz GQA TextVQA"
#  "VisDial CIRR VisualNews_t2i VisualNews_i2t MSCOCO_t2i MSCOCO_i2t NIGHTS WebQA FashionIQ Wiki-SS-NQ OVEN EDIS"
#  "MSCOCO RefCOCO RefCOCO-Matching Visual7W-Pointing"

#"ImageNet-1K N24News"
#"HatefulMemes VOC2007 SUN397"
#"Place365 ImageNet-A"
#"ImageNet-R ObjectNet Country211"
#"OK-VQA A-OKVQA"
#"DocVQA InfographicsVQA ChartQA"
#"ScienceQA Visual7W"
#"VizWiz GQA TextVQA"

#"Wiki-SS-NQ"
#"VisualNews_t2i"
#"MSCOCO_t2i"
#"FashionIQ OVEN EDIS"
#"MSCOCO Visual7W-Pointing"
#"RefCOCO RefCOCO-Matching"
#"VisDial CIRR NIGHTS"
#"WebQA VisualNews_i2t MSCOCO_i2t"

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
#MODEL_NAME="TIGER-Lab/VLM2Vec-Qwen2VL-2B"
#OUTPUT_PATH="/mnt/disks/embedding/exps/vlm2vec/baselines/vlm2vec-qwen2vl-2b/mmeb-image-v1_eval-highres/"
#MODEL_NAME="TIGER-Lab/VLM2Vec-Qwen2VL-7B"
#OUTPUT_PATH="/mnt/disks/embedding/exps/vlm2vec/baselines/vlm2vec-qwen2vl-7b/mmeb-image-v1_eval-highres/"
MODEL_NAME="VLM2Vec/VLM2Vec-V2.0"
OUTPUT_PATH="/mnt/disks/embedding/exps/vlm2vec/baselines/vlm2vec-qwen2vl-v2.0-2b/mmeb-image-v1_eval/"
#MODEL_NAME="VLM2Vec/VLM2Vec-V2.1"
#OUTPUT_PATH="/mnt/disks/embedding/exps/vlm2vec/baselines/vlm2vec-qwen2vl-v2.1-2b/mmeb-image-v1_eval/"

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
    local subset_id=$4

    local ckpt_name=$(basename "$checkpoint_path")
    local exp_dir=$(dirname "$checkpoint_path")
    local exp_basename=$(basename "$exp_dir")
    echo $gpu_id $subset_id
    echo $exp_basename
    mkdir -p $output_path

    # Build the command string
    cmd="CUDA_VISIBLE_DEVICES=$gpu_id torchrun --nproc_per_node=1 --master_port=2227$gpu_id --max_restarts=0 eval_image.py --model_backbone qwen2_vl --model_name $checkpoint_path --lora --pooling eos --normalize True --dataset_name TIGER-Lab/MMEB-eval --subset_name ${SUBSET_LIST[$subset_id]} --dataset_split test --per_device_eval_batch_size $BATCH_SIZE --encode_output_path $output_path/eval --image_dir /mnt/disks/embedding/data/vlm2vec/MMEB-eval/image-tasks/"

    echo "Running on GPU $gpu_id: $cmd"
    eval $cmd  # Execute the command
}

echo "Processing $MODEL_NAME"
for subset_id in "${!SUBSET_LIST[@]}"; do
    gpu_id=${GPU_IDS[$gpu_index]}
    # Run the experiment on the current GPU
#            echo $gpu_id $subset_id
    run_experiment "$MODEL_NAME" "$OUTPUT_PATH" $gpu_id $subset_id &

    # Increment GPU index and reset after reaching the maximum number of GPUs
    gpu_index=$(( (gpu_index + 1) % NUM_GPUS ))

    # If we completed a batch (all GPUs have been used), wait for the batch to finish
    if [ $gpu_index -eq 0 ]; then
        echo "Waiting for the current batch of jobs to finish..."
        wait  # Wait for all background jobs to finish before starting the next batch
    fi
done

# Wait for any remaining background jobs before exiting
wait
echo "All batches are complete."
