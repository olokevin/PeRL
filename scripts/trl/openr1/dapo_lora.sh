unset WANDB_DISABLED

# OUTPUT_DIR=outputs/debug

# MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# OUTPUT_DIR=outputs/grpo_lora_R1-Qwen-1.5B_$(date +%Y%m%d_%H%M%S)
# MAX_COMPLETION_LENGTH=16384
# RUN_NAME=1.5B-dapo-lora-128-$MAX_COMPLETION_LENGTH

DEVICE="${DEVICE:-7}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
LOSS_TYPE="${LOSS_TYPE:-dapo}" # grpo dapo

PEFT_TYPE="${PEFT_TYPE:-blocktt}" # blocktt lora full

LR="${LR:-1e-4}"
TBZ="${TBZ:-128}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-16384}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.4}"

lora_r=16
DECOMP_MODE=output_one_block
TRAIN_POSITION=small
S_MERGED_TO=frozen

if [[ "$PEFT_TYPE" == "full" ]]; then
    USE_PEFT=false
    NAME_SUFFIX=full
elif [[ "$PEFT_TYPE" == "lora" ]]; then
    USE_PEFT=true
    NAME_SUFFIX=$PEFT_TYPE-r${lora_r}
elif [[ "$PEFT_TYPE" == "blocktt" ]]; then
    USE_PEFT=true
    NAME_SUFFIX=$PEFT_TYPE-$TRAIN_POSITION-$S_MERGED_TO
else
    echo "Unsupported train mode: $PEFT_TYPE"
    exit 1
fi


OUTPUT_DIR=outputs/$MODEL_NAME/${LOSS_TYPE}/${PEFT_TYPE}/$TBZ-$MAX_COMPLETION_LENGTH-$LR-$NAME_SUFFIX-$(date +%Y%m%d_%H%M%S)
RUN_NAME=$MODEL_NAME-${LOSS_TYPE}-${PEFT_TYPE}-$TBZ-$MAX_COMPLETION_LENGTH-$LR-$NAME_SUFFIX

LOG_FILE=${OUTPUT_DIR}/output.log

mkdir -p ${OUTPUT_DIR}

MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-$((29500 + DEVICE))}

# CUDA_VISIBLE_DEVICES=6,7 ACCELERATE_LOG_LEVEL=info \
#     accelerate launch \
#     --main_process_port 29501 \
#     --config_file scripts/trl/accelerate/ds_zero2_2gpu.yaml \

CUDA_VISIBLE_DEVICES=${DEVICE} ACCELERATE_LOG_LEVEL=info \
    accelerate launch \
    --main_process_port ${MAIN_PROCESS_PORT} \
    --config_file scripts/trl/accelerate/single_gpu.yaml \
    run.py train \
    --config.training.gradient_accumulation_steps $TBZ \
    --config.training.per_device_train_batch_size 1 \
    --config.training.num_generations 8 \
    --config.training.learning_rate ${LR} \
    --config.training.run_name "${RUN_NAME}" \
    --config.training.vllm_gpu_memory_utilization ${VLLM_GPU_MEMORY_UTILIZATION} \
    --config.training.max_completion_length $MAX_COMPLETION_LENGTH \
    --config.common.seed 42 \
    --config.common.debug false \
    --config.model.model_name_or_path "${MODEL_NAME}" \
    --config.model.dtype "bfloat16" \
    --config.peft.task_type "CAUSAL_LM" \
    --config.peft.use_peft $USE_PEFT \
    --config.peft.type "${PEFT_TYPE}" \
    --config.peft.r ${lora_r} \
    --config.peft.lora_alpha 32 \
    --config.peft.lora_dropout 0.05 \
    --config.peft.blocktt_rank "full" \
    --config.peft.decomp_mode "$DECOMP_MODE" \
    --config.peft.train_position "$TRAIN_POSITION" \
    --config.peft.s_merged_to "$S_MERGED_TO" \
    --config.peft.total_step 1000 \
    --config.peft.target_modules '["q_proj","v_proj","k_proj","o_proj","up_proj","down_proj","gate_proj"]' \
    --config.training.beta 0.0 \
    --config.training.output_dir "${OUTPUT_DIR}" \
    --config.training.remove_unused_columns false \
    --config.training.num_train_epochs 1 \
    --config.training.warmup_ratio 0.0 \
    --config.training.max_prompt_length 512 \
    --config.training.logging_steps 1 \
    --config.training.save_strategy "steps" \
    --config.training.save_steps 64 \
    --config.training.max_steps 1024 \
    --config.training.use_vllm true \
    --config.training.top_entropy_quantile 1.0 \
    --config.training.epsilon_high 0.28 \
    --config.training.lr_scheduler_type "constant" \
    --config.training.lr_scheduler_kwargs.min_lr_rate 0.1 \
    --config.training.vllm_mode "colocate" \
    --config.training.use_liger_kernel false \
    --config.training.loss_type "${LOSS_TYPE}" \
    --config.training.report_to '["wandb"]' \
    --config.logging.trackio_space_id "Open-Tinker/Open-Tinker" \
    --config.logging.trackio_project "grpo-full-qwen3-4b" \
    --config.logging.wandb_project "grpo-full-qwen3-4b" \
    --config.dataset.dataset_name_or_path "open-r1/DAPO-Math-17k-Processed" \
    --config.dataset.example_numbers 1000000000 \
    &> ${LOG_FILE}