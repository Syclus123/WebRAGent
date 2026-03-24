export PYTHONPATH=../VLM2Vec/:$PYTHONPATH
export LD_LIBRARY_PATH=/fsx/home/xyang/embed-env/bin/python:/usr/local/cuda-12.1/targets/x86_64-linux/include/:/fsx/home/ruimeng/.local/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

GPU_ID=0
MODEL_PATH="TIGER-Lab/VLM2Vec-LLaVa-Next"
OUTPUT_PATH="/fsx/home/ruimeng/runs/test/hf-VLM2Vec-LLaVa-Next-lateprocess"
#TASKS="VisDial CIRR VisualNews_t2i VisualNews_i2t MSCOCO_t2i MSCOCO_i2t MSCOCO RefCOCO RefCOCO-Matching Visual7W-Pointing"
#TASKS="OK-VQA A-OKVQA DocVQA InfographicsVQA ChartQA"
TASKS="ImageNet-1K N24News HatefulMemes VOC2007 SUN397
      Place365 ImageNet-A ImageNet-R ObjectNet Country211
      OK-VQA A-OKVQA DocVQA InfographicsVQA ChartQA
      ScienceQA Visual7W VizWiz GQA TextVQA
      VisDial CIRR VisualNews_t2i VisualNews_i2t MSCOCO_t2i MSCOCO_i2t
      FashionIQ Wiki-SS-NQ
      NIGHTS WebQA OVEN EDIS
      MSCOCO RefCOCO RefCOCO-Matching Visual7W-Pointing"
cd /fsx/home/ruimeng/project/VLM2Vec
CUDA_VISIBLE_DEVICES=$GPU_ID torchrun --nproc_per_node=1 --master_port=22345 --max_restarts=0 eval.py --model_name $MODEL_PATH --image_dir /fsx/sfr/data/MMEB/MMEB_test/MMEB_Test_1K_New/images/ --encode_output_path $OUTPUT_PATH --pooling eos --normalize True --dataset_name TIGER-Lab/MMEB-eval --subset_name $TASKS --dataset_split test --per_device_eval_batch_size 8 --image_resolution high
