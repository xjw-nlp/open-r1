export WANDB_MODE=offline
TIME_TAG=$(date +%Y%m%d%H%M%S)
MODEL_NAME=DeepSeek-R1-Distill-Qwen-1.5B
DATA_PATH=/apdcephfs_qy3/share_1565115/jonxie/workspace/open-r1/data
BASE_MODEL_PATH=/apdcephfs_qy3/share_1565115/shared_resource/model_base/${MODEL_NAME}
CKPT_PATH=/apdcephfs_qy3/share_1565115/jonxie/workspace/open-r1/model/DeepSeek-R1-Distill-Qwen-1.5B-seqrec-sft-20250509113214/checkpoint-36000

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=1 src/open_r1/grpo_seqrec_ours.py \
    --config recipes/${MODEL_NAME}/grpo/config_seqrec_ours.yaml \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name /apdcephfs_qy3/share_1565115/jonxie/workspace/open-r1/data/Instruments/Instruments_reasoning \
    --output_dir model/${MODEL_NAME}-seqrec-grpo-${TIME_TAG} \
    --push_to_hub false \
    --hub_model_id null \
    --hub_strategy end \
    --save_strategy steps \
    --save_steps 200 \
    --seqrec_data_path $DATA_PATH \
    --dataset Instruments \
    --index_file .index.20250414.json \
    --max_prompt_length 10240 \
    --num_train_epochs 10 \
    --save_total_limit 5

