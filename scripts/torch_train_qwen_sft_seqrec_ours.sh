USED_GPUS='0,1,2,3,4,5,6,7'
NUM_GPUS=$(echo $USED_GPUS | tr ',' ' ' | wc -w)
export CUDA_VISIBLE_DEVICES=$USED_GPUS
export WANDB_MODE=offline
export CUDA_LAUNCH_BLOCKING=1

DATA_PATH=/apdcephfs_qy3/share_1565115/jonxie/workspace/open-r1/data
TIME_TAG=$(date +%Y%m%d%H%M%S)

torchrun --nproc_per_node=${NUM_GPUS} \
     --nnodes="1" \
     --node_rank="0" \
     --master_addr="127.0.0.1" \
     --master_port="12348" \
     src/open_r1/sft_seqrec_ours.py \
     --deepspeed recipes/accelerate_configs/zero2.json\
     --model_name_or_path /apdcephfs_qy3/share_1565115/shared_resource/model_base/Qwen2.5-1.5B \
     --dataset Instruments \
     --dataset_name Instruments \
     --data_path $DATA_PATH \
     --learning_rate 5.0e-4 \
     --num_train_epochs 30 \
     --max_seq_length 16384 \
     --per_device_train_batch_size 16 \
     --gradient_checkpointing \
     --bf16 \
     --tasks seqrec \
     --train_prompt_sample_num 1 \
     --train_data_sample_num 0 \
     --index_file .index.20250414.json \
     --output_dir model/Qwen2.5-1.5B-seqrec-sft-${TIME_TAG} \
     --only_train_response