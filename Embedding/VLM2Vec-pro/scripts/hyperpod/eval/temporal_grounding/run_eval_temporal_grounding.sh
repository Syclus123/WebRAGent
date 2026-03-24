export PYTHONPATH=/home/ziyan/VLM2Vec-pro:$PYTHONPATH

CUDA_VISIBLE_DEVICES=6 python eval_video_mret.py \
  --encode_output_path /home/ziyan/MMEB_Pro_Output \
  --model_name Qwen/Qwen2-VL-2B-Instruct \
  --checkpoint_path /home/ziyan/qwen2vl_2B-002-6.mmeb20_vidore1_videohound2_mteb15-v2-cap100k-rerun.qwenresize.lora16.bs1024pergpu128-ib64-droplast.GCq8p8.NormTemp002.lr5e5.step5kwarm200.maxlen2k.8H100_checkpoint-5000 \
  --lora True \
  --resize_use_processor true \
  --dataset_config configs/data_configs/darth_eval/test.yaml \
  --per_device_eval_batch_size 8
