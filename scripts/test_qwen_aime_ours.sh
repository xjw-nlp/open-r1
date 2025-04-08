NUM_GPUS=4
MODEL=/apdcephfs_qy3/share_1565115/shared_resource/model_base/Qwen2.5-1.5B-Instruct
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
TASK=aime24
OUTPUT_NAME=DeepSeek-R1-Distill-Qwen-1.5B-baseline
OUTPUT_DIR=data/evals/$OUTPUT_NAME

export VLLM_WORKER_MULTIPROC_METHOD=spawn
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 