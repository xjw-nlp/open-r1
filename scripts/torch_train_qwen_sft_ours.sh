USED_GPUS='0'
NUM_GPUS=$(echo $USED_GPUS | tr ',' ' ' | wc -w)
export CUDA_VISIBLE_DEVICES=$USED_GPUS
export WANDB_MODE=offline

torchrun --nproc_per_node=${NUM_GPUS} \
     --nnodes="1" \
     --node_rank="0" \
     --master_addr="127.0.0.1" \
     --master_port="12345" \
     src/open_r1/sft_ours.py \
     --deepspeed recipes/accelerate_configs/zero2.json\
     --model_name_or_path /apdcephfs_qy3/share_1565115/shared_resource/model_base/DeepSeek-R1-Distill-Qwen-1.5B \
     --dataset_name /apdcephfs_qy3/share_1565115/shared_resource/data_base/OpenR1-Math-220k \
     --learning_rate 1.0e-5 \
     --num_train_epochs 1 \
     --max_seq_length 16384 \
     --per_device_train_batch_size 16 \
     --gradient_checkpointing \
     --bf16 \
     --output_dir data/Qwen2.5-1.5B-Open-R1-Distill-25041700