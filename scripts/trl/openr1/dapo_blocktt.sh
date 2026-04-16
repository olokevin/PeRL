# CUDA_VISIBLE_DEVICES=7 bash scripts/trl/openr1/dapo_blocktt.sh 2>&1 &

unset WANDB_DISABLED

# OUTPUT_DIR=outputs/debug

# MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# OUTPUT_DIR=outputs/grpo_lora_R1-Qwen-1.5B_$(date +%Y%m%d_%H%M%S)
# MAX_COMPLETION_LENGTH=16384
# RUN_NAME=1.5B-dapo-lora-128-$MAX_COMPLETION_LENGTH

MODEL_NAME=Qwen/Qwen3-1.7B
LR=1e-4
TBZ=128
MAX_COMPLETION_LENGTH=8192
NAME_SUFFIX=input-small-frozen
OUTPUT_DIR=outputs/qwen3_1.7b/dapo_blocktt/$TBZ-$MAX_COMPLETION_LENGTH-$LR-$NAME_SUFFIX-$(date +%Y%m%d_%H%M%S)
RUN_NAME=qwen3-1.7b-dapo-blocktt-$TBZ-$MAX_COMPLETION_LENGTH-$LR-$NAME_SUFFIX

LOG_FILE=${OUTPUT_DIR}/output.log

mkdir -p ${OUTPUT_DIR}

# CUDA_VISIBLE_DEVICES=6,7 ACCELERATE_LOG_LEVEL=info \
#     accelerate launch \
#     --main_process_port 29501 \
#     --config_file scripts/trl/accelerate/ds_zero2_2gpu.yaml \

ACCELERATE_LOG_LEVEL=info \
    accelerate launch \
    --main_process_port 29501 \
    --config_file scripts/trl/accelerate/single_gpu.yaml \
    run.py train \
    --config.training.gradient_accumulation_steps $TBZ \
    --config.training.per_device_train_batch_size 1 \
    --config.training.num_generations 8 \
    --config.training.learning_rate 1e-4 \
    --config.training.run_name "${RUN_NAME}" \
    --config.training.vllm_gpu_memory_utilization 0.3 \
    --config.training.max_completion_length $MAX_COMPLETION_LENGTH \
    --config.common.seed 42 \
    --config.common.debug false \
    --config.model.model_name_or_path "${MODEL_NAME}" \
    --config.model.dtype "bfloat16" \
    --config.peft.use_peft true \
    --config.peft.type "blocktt" \
    --config.peft.blocktt_rank "full" \
    --config.peft.decomp_mode "input_one_block" \
    --config.peft.train_position "small" \
    --config.peft.s_merged_to "frozen" \
    --config.peft.task_type "CAUSAL_LM" \
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
    --config.training.loss_type "dapo" \
    --config.training.report_to '["wandb"]' \
    --config.logging.trackio_space_id "Open-Tinker/Open-Tinker" \
    --config.logging.trackio_project "grpo-full-qwen3-4b" \
    --config.logging.wandb_project "grpo-full-qwen3-4b" \
    --config.dataset.dataset_name_or_path "open-r1/DAPO-Math-17k-Processed" \
    --config.dataset.example_numbers 1000000000 \
    &> ${LOG_FILE}

# CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info \
#     accelerate launch \
#     --main_process_port 29503 \
#     --config_file scripts/trl/accelerate/ds_zero2_4gpu.yaml \
#     run.py train \
#     --config.common.seed 42 \
#     --config.common.debug false \
#     --config.model.model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
#     --config.model.dtype "bfloat16" \
#     --config.peft.use_peft true \
#     --config.peft.type "blocktt" \
#     --config.peft.blocktt_rank "full" \
#     --config.peft.decomp_mode "input_one_block" \
#     --config.peft.train_position "small" \
#     --config.peft.s_merged_to "frozen" \
#     --config.peft.train_bias true \
#     --config.peft.target_modules '["q_proj","v_proj","k_proj","o_proj","up_proj","down_proj","gate_proj"]' \
#     --config.training.learning_rate 1e-5 \
#     --config.training.beta 0.0 \
#     --config.training.output_dir "${OUTPUT_DIR}" \
#     --config.training.run_name "${OUTPUT_DIR}" \
#     --config.training.remove_unused_columns false \
#     --config.training.gradient_accumulation_steps 8 \
#     --config.training.num_train_epochs 1 \
#     --config.training.max_completion_length 16384 \
#     --config.training.num_generations 8 \
#     --config.training.warmup_ratio 0.0 \
#     --config.training.max_prompt_length 512 \
#     --config.training.logging_steps 1 \
#     --config.training.per_device_train_batch_size 4 \
#     --config.training.save_strategy "steps" \
#     --config.training.save_steps 64 \
#     --config.training.max_steps 1024 \
#     --config.training.use_vllm true \
#     --config.training.top_entropy_quantile 1.0 \
#     --config.training.epsilon_high 0.28 \
#     --config.training.lr_scheduler_type "constant" \
#     --config.training.lr_scheduler_kwargs.min_lr_rate 0.1 \
#     --config.training.vllm_mode "colocate" \
#     --config.training.vllm_gpu_memory_utilization 0.4 \
#     --config.training.use_liger_kernel false \
#     --config.training.loss_type "dapo" \
#     --config.training.report_to '["wandb"]' \
#     --config.logging.trackio_space_id "Open-Tinker/Open-Tinker" \
#     --config.logging.trackio_project "grpo-blocktt-qwen2-5-1-5b" \
#     --config.logging.wandb_project "grpo-blocktt-qwen2-5-1-5b" \
#     --config.dataset.dataset_name_or_path "open-r1/DAPO-Math-17k-Processed" \
#     --config.dataset.example_numbers 1000000000 \
#     &> ${LOG_FILE}
