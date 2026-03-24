#!/bin/bash
source /fsx/home/ruimeng/.bashrc
eval "$(/fsx/home/ruimeng/envs/miniconda3/bin/conda shell.bash hook)"
conda_env=/fsx/home/ruimeng/envs/vlm2vec
conda activate $conda_env
echo "conda location: $(which conda)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"


export LD_LIBRARY_PATH=/fsx/home/xyang/embed-env/bin/python:/usr/local/cuda-12.1/targets/x86_64-linux/include/:/fsx/home/ruimeng/.local/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export PATH=/fsx/home/ruimeng/envs/vlm2vec/bin:/fsx/home/ruimeng/envs/vlm2vec/lib/python3.10/site-packages:$PATH

#conda activate /fsx/home/xyang/embed-env
export HF_DATASETS_CACHE=/fsx/home/ruimeng/data/.hfdata_cache
export HF_HOME=/fsx/home/ruimeng/data/.hfmodel_cache/
export HUGGING_FACE_HUB_TOKEN=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

cd /fsx/home/ruimeng/project/VLM2Vec

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
IMAGE_RESOLUTION="high"
EXPERIMENTS_LIST=(
     # qwen unified data
    "/fsx/home/ruimeng/runs/mmeb/qwen2vl_7B-001-2.mmeb20_vidore1-v1-1.high-res.lora8.bs1024pergpu128.GCq4p4.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-1600"
#    "/fsx/home/ruimeng/runs/mmeb/qwen2vl_2B-002-1.mmeb20_vidore2_videohound8-v1.qwenresize.lora8.bs1024pergpu128.GCq8p8.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-1000"

#    "/fsx/home/ruimeng/runs/mmeb/qwen2vl_2B-001-2.mmeb20_vidore1-v1-1.high-res.lora8.bs1024pergpu128.GCq4p4.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-2000"
#    "/fsx/home/ruimeng/runs/mmeb/qwen2vl_7B-001-1.mmeb_vidore.high-res.lora8.bs512pergpu64.GCq1p1.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-1100"
#    "/fsx/home/ruimeng/runs/mmeb/qwen2vl_2B-001-1.mmeb20_vidore1-v1-0.high-res.lora8.bs1024pergpu128.GCq4p4.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-2000"
#    "/fsx/home/ruimeng/runs/mmeb/qwen2vl_2B-001-1.mmeb_vidore.high-res.lora8.mmeb20_sub100k.bs1024pergpu128.GCq4p4.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-600"
#    "/fsx/home/ruimeng/runs/mmeb/mmeb007-qwen2vl-2B-3.1-lateprocess-mid_res-flashattn-leftpadforreal.lora8.mmeb20_sub100k.bs1024pergpu128.GCq16p16.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-1700"
     # qwen late-process
#     "/fsx/home/ruimeng/runs/mmeb/mmeb006-qwen2vl-7B-3-lateprocess-high_res.lora8.mmeb20_sub100k.bs1024pergpu128.GCq2p2.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-1700/"  # bsz=8,78055 / 81559 MB
#     "/fsx/home/ruimeng/runs/mmeb/mmeb006-qwen2vl-7B-3-lateprocess-high_res.lora8.mmeb20_sub100k.bs1024pergpu128.GCq2p2.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-1000/"
#     "/fsx/home/ruimeng/runs/mmeb/mmeb006-qwen2vl-7B-2-lateprocess-high_res.lora8.mmeb20_sub100k.bs512pergpu64.GCq2p2.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-1000"  # bsz=3,67793 / 81559 MB
#     "/fsx/home/ruimeng/runs/mmeb/mmeb006-qwen2vl-2B-4-lateprocess-high_res.lora8.mmeb20_sub100k.bs1024pergpu128.GCq4p4.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-2000"  # bsz=3,56141 / 81559 MB, flashattn+bsz=16,70719 / 81559 MB
#     "/fsx/home/ruimeng/runs/mmeb/mmeb006-qwen2vl-7B-1-lateprocess-mid_res.lora8.mmeb20_sub100k.bs512pergpu64.GCq4p4.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-2000"
#     "/fsx/home/ruimeng/runs/mmeb/mmeb006-qwen2vl-2B-3.1-lateprocess-mid_res-flashattn-leftpadforreal.lora8.mmeb20_sub100k.bs256pergpu32.GCq16p16.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-2000"
#     "/fsx/home/ruimeng/runs/mmeb/mmeb006-qwen2vl-2B-3-lateprocess-mid_res-flashattn-leftpad.lora8.mmeb20_sub100k.bs256pergpu32.GCq8p8.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-2000"
#     "/fsx/home/ruimeng/runs/mmeb/mmeb006-qwen2vl-2B-2-lateprocess-mid_res.lora8.mmeb20_sub100k.bs256pergpu32.GCq8p8_OOM.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-1000"
#     "/fsx/home/ruimeng/runs/mmeb/mmeb006-qwen2vl-2B-1-lateprocess-low_res.lora8.mmeb20_sub100k.bs1024pergpu128.GCq32p32.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-2000"
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
#    echo $exp_dir
    echo $exp_basename
#    local output_path="/fsx/home/ruimeng/runs/mmeb/$exp_basename/$ckpt_name"
#    echo $output_dir
#    mkdir -p $output_path

    if [[ "$exp_basename" == *lora* ]]; then
        lora=" --lora"
        print_warning "Extracted lora: $lora"
    else
        lora=" "
        print_warning "lora not found in exp name"
    fi
    # Extract 'lenXXX' and 'cropXX' from the directory name using regex
    if [[ "$exp_basename" =~ len([0-9]+) ]]; then
        max_len_arg="--max_len ${BASH_REMATCH[1]}"
        echo "Extracted max_len: $max_len_arg"
    else
        print_warning "max_len not found in exp name, not set"
        max_len_arg=" "
#        return 1  # Exit the function with a non-zero status
    fi
    # Build the command string
    cmd="CUDA_VISIBLE_DEVICES=$gpu_id torchrun --nproc_per_node=1 --master_port=2229$gpu_id --max_restarts=0 eval_mmeb.py $lora --model_name $checkpoint_path --encode_output_path $checkpoint_path/eval $max_len_arg --pooling eos --normalize True --dataset_name EmbVision/MMEB_Test_1K_New --subset_name ${SUBSET_LIST[$subset_id]} --dataset_split test --image_resolution $IMAGE_RESOLUTION --per_device_eval_batch_size $BATCH_SIZE --image_dir /fsx/sfr/data/MMEB/MMEB_test/MMEB_Test_1K_New/images/"

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
