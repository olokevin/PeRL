### train
# DEVICE=7 bash scripts/trl/openr1/dapo_lora.sh 2>&1 &

### eval
CUDA_VISIBLE_DEVICES=7 \
  MAX_MODEL_LEN=8192 \
  MAX_NEW_TOKENS=3072 \
  CKPT=outputs/Qwen/Qwen3-1.7B/dapo/lora/128-3072-1e-4-lora-r16-20260410_114837/checkpoint-320 \
  bash scripts/trl/openr1/eval_ckpt.sh