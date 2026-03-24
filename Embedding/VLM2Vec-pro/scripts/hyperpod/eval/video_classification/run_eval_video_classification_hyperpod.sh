export PYTHONPATH=/home/mingyi/AI-Projects/VLM2Vec-pro:$PYTHONPATH

CUDA_VISIBLE_DEVICES=2 python3 eval_video_cls.py \
--lora --model_name /fsx/home/ruimeng/runs/mmeb/qwen2vl_2B-002-6.mmeb20_vidore1_videohound2_mteb15-v2-cap100k-rerun.qwenresize.lora16.bs1024pergpu128-ib64-droplast.GCq8p8.NormTemp002.lr5e5.step5kwarm200.maxlen2k.8H100/checkpoint-5000/ --encode_output_path /fsx/home/ruimeng/runs/mmeb/qwen2vl_2B-002-6.mmeb20_vidore1_videohound2_mteb15-v2-cap100k-rerun.qwenresize.lora16.bs1024pergpu128-ib64-droplast.GCq8p8.NormTemp002.lr5e5.step5kwarm200.maxlen2k.8H100/checkpoint-5000/eval-video --pooling eos --normalize True --per_device_eval_batch_size 16 --dataloader_num_workers 8 --resize_use_processor true --dataset_config configs/data_configs/eval/video_cls.yaml

