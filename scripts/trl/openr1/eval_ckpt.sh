#!/usr/bin/env bash
set -euo pipefail

# Example:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/trl/openr1/eval_ckpt.sh
#   CKPT=outputs/my_run/checkpoint-320 DATASET=aime2024,math500 bash scripts/trl/openr1/eval_ckpt.sh
#   CKPT=/path/to/full/checkpoint CKPT_TYPE=full bash scripts/trl/openr1/eval_ckpt.sh

CKPT="${CKPT:-outputs/Qwen/Qwen3-1.7B/dapo/lora/128-3072-1e-4-lora-r16-20260410_114837}"
CKPT_STEP="${CKPT_STEP:-latest}"
CKPT_TYPE="${CKPT_TYPE:-auto}" # auto|full|lora|blocktt
MODEL="${MODEL:-}"             # optional override for adapter checkpoints

DATASET="${DATASET:-aime2024,math500,gsm8k}"
DATASET_ROOT="${DATASET_ROOT:-datasets}"
PROMPT_FORMAT="${PROMPT_FORMAT:-lighteval}"
ROLLOUT_N="${ROLLOUT_N:-1}"

DTYPE="${DTYPE:-bfloat16}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"

NUM_GPUS="${NUM_GPUS:-1}"
DP_SIZE="${DP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-1}"
SERVE_PORT="${SERVE_PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.60}"
MAX_NUM_REQUEST="${MAX_NUM_REQUEST:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
SEED="${SEED:-42}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-3600}"
MAX_SAMPLES="${MAX_SAMPLES:-}" # optional

RUN_TAG="${RUN_TAG:-$(basename "${CKPT%/}")}" 
RESULT_DIR="${RESULT_DIR:-outputs/eval/${RUN_TAG}-$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-${RESULT_DIR}/eval.log}"

mkdir -p "${RESULT_DIR}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

echo "[eval] CKPT=${CKPT}"
echo "[eval] CKPT_STEP=${CKPT_STEP} CKPT_TYPE=${CKPT_TYPE}"
echo "[eval] DATASET=${DATASET} ROLLOUT_N=${ROLLOUT_N}"
echo "[eval] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[eval] nvidia-smi memory snapshot:"
  nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader || true
fi
echo "[eval] RESULT_DIR=${RESULT_DIR}"

CMD=(
  python -m perl.eval
  --result-dir "${RESULT_DIR}"
  --ckpt "${CKPT}"
  --ckpt-step "${CKPT_STEP}"
  --ckpt-type "${CKPT_TYPE}"
  --dataset "${DATASET}"
  --dataset-root "${DATASET_ROOT}"
  --prompt-format "${PROMPT_FORMAT}"
  --rollout-n "${ROLLOUT_N}"
  --serve-port "${SERVE_PORT}"
  --dp-size "${DP_SIZE}"
  --tp-size "${TP_SIZE}"
  --num-gpus "${NUM_GPUS}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-num-request "${MAX_NUM_REQUEST}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --seed "${SEED}"
  --request-timeout "${REQUEST_TIMEOUT}"
  --dtype "${DTYPE}"
  --max-model-len "${MAX_MODEL_LEN}"
)

if [[ -n "${MODEL}" ]]; then
  CMD+=(--model "${MODEL}")
fi

if [[ "${TRUST_REMOTE_CODE}" == "true" ]]; then
  CMD+=(--trust-remote-code)
fi

if [[ -n "${MAX_SAMPLES}" ]]; then
  CMD+=(--max-samples "${MAX_SAMPLES}")
fi

if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=(${VLLM_EXTRA_ARGS})
  CMD+=("${EXTRA_ARR[@]}")
fi

PYTHONPATH=. "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
