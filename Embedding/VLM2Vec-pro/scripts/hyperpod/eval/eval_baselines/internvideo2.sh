#CUDA_VISIBLE_DEVICES=7 python eval.py \
#  --encode_output_path /home/ziyan/MMEB_Pro_Output/InternVideo2 \
#  --model_name OpenGVLab/InternVideo2-Stage2_6B \
#  --model_type internvideo2 \
#  --per_device_eval_batch_size 32 \
#  --dataset_config configs/data_configs/darth_eval/video_ret.yaml

CUDA_VISIBLE_DEVICES=0 python eval_mmeb_new.py \
  --encode_output_path /home/ziyan/MMEB_Pro_Output/InternVideo2 \
  --model_name OpenGVLab/InternVideo2-Stage2_6B \
  --model_type internvideo2 \
  --per_device_eval_batch_size 64 \
  --dataset_config configs/data_configs/darth_eval/mmeb_ret.yaml
