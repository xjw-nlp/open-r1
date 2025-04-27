# export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
USED_GPUS='0,1,2,3,4,5,6,7'
NUM_GPUS=$(echo $USED_GPUS | tr ',' ' ' | wc -w)
export CUDA_VISIBLE_DEVICES=$USED_GPUS

DATASET=Instruments
DATA_PATH=./data
CKPT_PATH=/apdcephfs_qy3/share_1565115/jonxie/workspace/open-r1/model/Qwen2.5-1.5B-seqrec-sft-20250425124129/checkpoint-11500
RESULTS_FILE=./results/$DATASET/ddp.json
MODEL_NAME=Qwen2.5-1.5B
BASE_MODEL=/apdcephfs_qy3/share_1565115/shared_resource/model_base/${MODEL_NAME}

torchrun --nproc_per_node=${NUM_GPUS} --master_port=14324 src/open_r1/test_ddp.py \
    --ckpt_path ${CKPT_PATH} \
    --base_model $BASE_MODEL \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 1 \
    --num_beams 20 \
    --test_prompt_ids 0 \
    --index_file .index.20250414.json
    

