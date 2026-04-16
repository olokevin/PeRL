import argparse
import asyncio
import atexit
import json
import logging
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import aiohttp
from datasets import load_dataset


PROMPT_TEMPLATES = {
    "lighteval": """{problem} Please reason step by step, and put your final answer within \\boxed{{}}.""",
    "open-r1": """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.

{problem}

Remember to put your answer on its own line after "Answer:".
""".strip(),
}

DATASETS = {
    "aime2024": ("HuggingFaceH4/aime_2024", "train"),
    "aime2025": ("yentinglin/aime_2025", "train"),
    "amc2023": ("zwhe99/amc23", "test"),
    "math500": ("HuggingFaceH4/MATH-500", "test"),
    "minerva": ("math-ai/minervamath", "test"),
    "hmmt2025": ("FlagEval/HMMT_2025", "train"),
}

LOCAL_DATASET_ALIASES = {
    "aime2024": "aime24.jsonl",
    "math500": "math.jsonl",
    "gsm8k": "gsm8k.jsonl",
}

CHECKPOINT_NAME_PATTERN = re.compile(r"^checkpoint-(\d+)$")
WEIGHT_FILE_NAMES = {
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
}


@dataclass
class ResolvedCheckpoint:
    requested_ckpt: Path
    checkpoint_dir: Path
    checkpoint_step: int | None
    checkpoint_type: str
    base_model_name_or_path: str | None


class StageContext:
    def __init__(self, logger: logging.Logger, stage: str, description: str) -> None:
        self.logger = logger
        self.stage = stage
        self.description = description
        self.start = 0.0

    def __enter__(self):
        self.start = time.time()
        self.logger.info("[%s] %s - start", self.stage, self.description)
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.time() - self.start
        if exc is None:
            self.logger.info("[%s] %s - done (%.2fs)", self.stage, self.description, elapsed)
            return False
        self.logger.exception("[%s] %s - failed after %.2fs: %s", self.stage, self.description, elapsed, exc)
        return False


def setup_logging(result_dir: Path) -> logging.Logger:
    result_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("perl.eval")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(result_dir / "eval.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _checkpoint_dirs(path: Path) -> List[Tuple[int, Path]]:
    if not path.exists() or not path.is_dir():
        return []

    candidates: List[Tuple[int, Path]] = []
    for child in path.iterdir():
        if not child.is_dir():
            continue
        m = CHECKPOINT_NAME_PATTERN.match(child.name)
        if m:
            candidates.append((int(m.group(1)), child))

    candidates.sort(key=lambda x: x[0])
    return candidates


def _resolve_checkpoint_dir(ckpt_path: Path, ckpt_step: str) -> Tuple[Path, int | None]:
    if not ckpt_path.exists():
        raise ValueError(f"Checkpoint path does not exist: {ckpt_path}")
    if not ckpt_path.is_dir():
        raise ValueError(f"Checkpoint path must be a directory: {ckpt_path}")

    m = CHECKPOINT_NAME_PATTERN.match(ckpt_path.name)
    if m:
        return ckpt_path, int(m.group(1))

    dirs = _checkpoint_dirs(ckpt_path)
    if not dirs:
        return ckpt_path, None

    if ckpt_step == "latest":
        return dirs[-1][1], dirs[-1][0]

    try:
        target_step = int(ckpt_step)
    except ValueError as exc:
        raise ValueError(f"--ckpt-step must be 'latest' or an integer, got: {ckpt_step}") from exc

    for step, checkpoint_dir in dirs:
        if step == target_step:
            return checkpoint_dir, step

    available = ", ".join(str(step) for step, _ in dirs)
    raise ValueError(
        f"Requested checkpoint step {target_step} not found under {ckpt_path}. "
        f"Available steps: {available}"
    )


def _has_full_weights(path: Path) -> bool:
    if any((path / name).exists() for name in WEIGHT_FILE_NAMES):
        return True
    if list(path.glob("pytorch_model-*.bin")):
        return True
    return bool(list(path.glob("model-*.safetensors")))


def _infer_adapter_type(adapter_config: Dict[str, Any]) -> str:
    peft_type = str(adapter_config.get("peft_type", "")).upper()
    if peft_type == "BLOCKTT":
        return "blocktt"
    return "lora"


def _looks_like_legacy_blocktt(checkpoint_dir: Path) -> bool:
    output_log = checkpoint_dir.parent / "output.log"
    if output_log.exists():
        try:
            with output_log.open("r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            if "Unknown PEFT type passed: BLOCKTT" in text or "applying blocktt" in text.lower():
                return True
        except Exception:
            pass

    readme = checkpoint_dir / "README.md"
    if readme.exists():
        try:
            content = readme.read_text(encoding="utf-8", errors="ignore").lower()
            if "base_model" in content and "library_name: peft" in content:
                return True
        except Exception:
            pass

    return False


def resolve_checkpoint_spec(
    ckpt_path: Path,
    ckpt_step: str,
    ckpt_type: str = "auto",
    model_override: str | None = None,
) -> ResolvedCheckpoint:
    if ckpt_type not in {"auto", "full", "lora", "blocktt"}:
        raise ValueError(f"Unsupported --ckpt-type: {ckpt_type}")

    checkpoint_dir, checkpoint_step = _resolve_checkpoint_dir(ckpt_path, ckpt_step)

    adapter_config_path = checkpoint_dir / "adapter_config.json"
    adapter_model_safetensors = checkpoint_dir / "adapter_model.safetensors"
    adapter_model_bin = checkpoint_dir / "adapter_model.bin"

    has_adapter_config = adapter_config_path.exists()
    has_adapter_weights = adapter_model_safetensors.exists() or adapter_model_bin.exists()
    has_full = (checkpoint_dir / "config.json").exists() and _has_full_weights(checkpoint_dir)

    if has_adapter_config and not has_adapter_weights:
        raise ValueError(
            "Adapter checkpoint is incomplete. Missing adapter weight file in "
            f"{checkpoint_dir}. Required one of: adapter_model.safetensors, adapter_model.bin"
        )

    if has_adapter_config and has_adapter_weights:
        adapter_config = _load_json(adapter_config_path)
        inferred_type = _infer_adapter_type(adapter_config)
        if ckpt_type != "auto" and ckpt_type != inferred_type:
            raise ValueError(
                f"--ckpt-type={ckpt_type} conflicts with adapter peft_type={adapter_config.get('peft_type')} "
                f"at {checkpoint_dir}"
            )

        base_model = model_override or adapter_config.get("base_model_name_or_path")
        if not base_model:
            raise ValueError(
                "Base model could not be resolved for adapter checkpoint. "
                "Provide --model or ensure adapter_config.json has base_model_name_or_path."
            )

        return ResolvedCheckpoint(
            requested_ckpt=ckpt_path,
            checkpoint_dir=checkpoint_dir,
            checkpoint_step=checkpoint_step,
            checkpoint_type=inferred_type,
            base_model_name_or_path=str(base_model),
        )

    if has_full:
        if ckpt_type in {"lora", "blocktt"}:
            raise ValueError(
                f"--ckpt-type={ckpt_type} conflicts with full checkpoint layout at {checkpoint_dir}"
            )
        return ResolvedCheckpoint(
            requested_ckpt=ckpt_path,
            checkpoint_dir=checkpoint_dir,
            checkpoint_step=checkpoint_step,
            checkpoint_type="full",
            base_model_name_or_path=None,
        )

    if _looks_like_legacy_blocktt(checkpoint_dir):
        raise ValueError(
            "Detected a legacy/failed BlockTT checkpoint that cannot be evaluated: "
            f"{checkpoint_dir}. Missing required files adapter_config.json + adapter_model.safetensors. "
            "This usually comes from the historical save failure (`Unknown PEFT type passed: BLOCKTT`). "
            "Use a newly saved BlockTT adapter checkpoint or a fully merged model checkpoint."
        )

    raise ValueError(
        f"Unsupported checkpoint layout at {checkpoint_dir}. Expected either:\n"
        "1) full checkpoint: config.json + model weights, or\n"
        "2) adapter checkpoint: adapter_config.json + adapter_model.safetensors"
    )


def _resolve_torch_dtype(dtype: str):
    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "auto": None,
    }
    if dtype.lower() not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype.lower()]


def merge_model_if_needed(
    args: argparse.Namespace,
    result_dir: Path,
    logger: logging.Logger,
) -> Tuple[str, ResolvedCheckpoint]:
    spec = resolve_checkpoint_spec(
        ckpt_path=Path(args.ckpt),
        ckpt_step=args.ckpt_step,
        ckpt_type=args.ckpt_type,
        model_override=args.model if args.model else None,
    )

    logger.info(
        "Resolved checkpoint: dir=%s step=%s type=%s",
        spec.checkpoint_dir,
        spec.checkpoint_step,
        spec.checkpoint_type,
    )

    if spec.checkpoint_type == "full":
        return str(spec.checkpoint_dir), spec

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if spec.checkpoint_type == "blocktt":
        from perl.lora.blocktt import register_blocktt_method

        register_blocktt_method()

    merged_model_dir = result_dir / "model"
    if merged_model_dir.exists():
        shutil.rmtree(merged_model_dir)
    merged_model_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = _resolve_torch_dtype(args.dtype)

    logger.info("Loading base model for merge: %s", spec.base_model_name_or_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        spec.base_model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )

    logger.info("Loading adapter checkpoint: %s", spec.checkpoint_dir)
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(spec.checkpoint_dir),
        is_trainable=False,
    )

    logger.info("Merging adapter into base model")
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(str(merged_model_dir), safe_serialization=True)

    tokenizer_source = (
        str(spec.checkpoint_dir)
        if (spec.checkpoint_dir / "tokenizer_config.json").exists()
        else spec.base_model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.save_pretrained(str(merged_model_dir))

    logger.info("Merged model written to %s", merged_model_dir)
    return str(merged_model_dir), spec


def extract_vllm_args(unknown: List[str]) -> Tuple[List[str], List[str]]:
    # Pass unknown args directly to vLLM server command.
    return list(unknown), []


def _build_vllm_command(model_path: str, port: int, args: argparse.Namespace, vllm_args: List[str]) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--port",
        str(port),
        "--served-model-name",
        args.served_model_name,
        "--tensor-parallel-size",
        str(args.tp_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--api-key",
        args.api_key,
    ]

    if args.trust_remote_code:
        command.append("--trust-remote-code")

    command.extend(vllm_args)
    return command


def _resolve_visible_device_ids(available_gpu_count: int) -> List[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return [str(i) for i in range(available_gpu_count)]
    ids = [token.strip() for token in raw.split(",") if token.strip()]
    return ids


def start_vllm_processes(
    model_path: str,
    args: argparse.Namespace,
    vllm_args: List[str],
    logger: logging.Logger,
) -> Tuple[List[subprocess.Popen], List[int]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for vLLM evaluation, but no GPU was detected.")

    available_gpu_count = torch.cuda.device_count()
    if args.num_gpus > available_gpu_count:
        raise RuntimeError(
            f"Insufficient GPUs: requested num_gpus={args.num_gpus}, available={available_gpu_count}"
        )

    required_gpu_count = args.dp_size * args.tp_size
    if args.num_gpus < required_gpu_count:
        raise RuntimeError(
            f"num_gpus({args.num_gpus}) must be >= dp_size * tp_size ({required_gpu_count})"
        )
    if available_gpu_count < required_gpu_count:
        raise RuntimeError(
            f"Available GPUs ({available_gpu_count}) < required dp_size * tp_size ({required_gpu_count})"
        )

    visible_device_ids = _resolve_visible_device_ids(available_gpu_count)
    if len(visible_device_ids) < required_gpu_count:
        raise RuntimeError(
            f"Visible CUDA devices ({visible_device_ids}) are fewer than required dp_size * tp_size ({required_gpu_count})"
        )

    processes: List[subprocess.Popen] = []
    ports: List[int] = []

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.dp_size):
        port = args.serve_port + i
        ports.append(port)

        gpu_start = i * args.tp_size
        gpu_ids = visible_device_ids[gpu_start : gpu_start + args.tp_size]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

        command = _build_vllm_command(model_path, port, args, vllm_args)
        logger.info("Starting vLLM on port=%d GPUs=%s", port, env["CUDA_VISIBLE_DEVICES"])

        log_path = result_dir / f"vllm_{port}.log"
        log_file = open(log_path, "w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        process._log_file = log_file  # type: ignore[attr-defined]
        process._log_path = log_path  # type: ignore[attr-defined]
        processes.append(process)

    return processes, ports


def stop_vllm_processes(processes: List[subprocess.Popen], logger: logging.Logger) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()

    deadline = time.time() + 15
    for process in processes:
        if process.poll() is not None:
            continue
        timeout = max(0.0, deadline - time.time())
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()

    for process in processes:
        log_file = getattr(process, "_log_file", None)
        if log_file is not None:
            try:
                log_file.close()
            except Exception:
                logger.warning("Failed to close vLLM log file handle")


def wait_for_vllm_ready(
    port: int,
    proc: subprocess.Popen,
    timeout: float,
    logger: logging.Logger,
    api_key: str,
) -> bool:
    url = f"http://127.0.0.1:{port}/v1/models"
    start = time.time()

    while time.time() - start < timeout:
        if proc.poll() is not None:
            log_path = getattr(proc, "_log_path", None)
            if log_path is not None:
                logger.error(
                    "vLLM process on port %d exited early with code %s. Check log: %s",
                    port,
                    proc.returncode,
                    log_path,
                )
            else:
                logger.error("vLLM process on port %d exited early with code %s", port, proc.returncode)
            return False

        try:
            request = urllib.request.Request(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            with urllib.request.urlopen(request, timeout=2.0) as resp:
                if resp.status == 200:
                    logger.info("vLLM ready on port %d", port)
                    return True
        except Exception:
            pass

        time.sleep(1)

    logger.error("Timed out waiting for vLLM ready on port %d", port)
    return False


async def generate_with_vllm_async(
    session: aiohttp.ClientSession,
    prompt: str,
    port: int,
    args: argparse.Namespace,
) -> str:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    request_max_tokens = min(args.max_new_tokens, max(1, args.max_model_len - 128))
    payload: Dict[str, Any] = {
        "model": args.served_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": request_max_tokens,
    }

    if args.seed is not None:
        payload["seed"] = int(args.seed)

    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    headers = {"Authorization": f"Bearer {args.api_key}"}

    async with session.post(url, json=payload, headers=headers, timeout=timeout) as resp:
        text = await resp.text()
        if resp.status != 200:
            if resp.status == 400 and "max_tokens" in text:
                # vLLM error format includes:
                # "... request has <prompt_tokens> input tokens (<req> > <ctx> - <prompt_tokens>)"
                m = re.search(r"\((\d+)\s*>\s*(\d+)\s*-\s*(\d+)\)", text)
                if m:
                    model_ctx = int(m.group(2))
                    prompt_tokens = int(m.group(3))
                    retry_max_tokens = max(1, model_ctx - prompt_tokens - 16)
                    retry_payload = dict(payload)
                    retry_payload["max_tokens"] = retry_max_tokens
                    async with session.post(
                        url,
                        json=retry_payload,
                        headers=headers,
                        timeout=timeout,
                    ) as retry_resp:
                        retry_text = await retry_resp.text()
                        if retry_resp.status == 200:
                            data = json.loads(retry_text)
                            choices = data.get("choices", [])
                            if not choices:
                                return ""
                            message = choices[0].get("message", {})
                            return str(message.get("content", ""))
                        raise RuntimeError(
                            f"vLLM request failed after retry status={retry_resp.status} body={retry_text[:500]}"
                        )
            raise RuntimeError(f"vLLM request failed status={resp.status} body={text[:500]}")

    data = json.loads(text)
    choices = data.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    return str(message.get("content", ""))


def _extract_candidate_answer(response: str) -> Tuple[str | None, float]:
    from perl.rm.math_verifier import extract_boxed_answer

    boxed = extract_boxed_answer(response)
    if boxed is not None:
        return boxed, 1.0

    matches = re.findall(r"(?im)^answer\s*:\s*(.+?)\s*$", response)
    if matches:
        return matches[-1].strip(), 1.0

    return None, 0.0


def grade_answer_perl(response: str, ground_truth: str) -> Tuple[float, float]:
    from perl.rm.math_verifier import compute_score

    candidate, format_score = _extract_candidate_answer(response)
    if not candidate:
        return 0.0, format_score

    try:
        score = compute_score(candidate, ground_truth)
    except Exception:
        score = 0.0
    return score, format_score


def _try_load_local_jsonl_dataset(dataset_name: str, dataset_root: Path):
    candidates: List[Path] = []
    if dataset_name in LOCAL_DATASET_ALIASES:
        candidates.append(dataset_root / LOCAL_DATASET_ALIASES[dataset_name])
    candidates.append(dataset_root / f"{dataset_name}.jsonl")

    for path in candidates:
        if path.exists():
            rows: List[Dict[str, Any]] = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
            return rows
    return None


def load_dataset_from_hf(dataset_name: str, dataset_root: Path):
    local_dataset = _try_load_local_jsonl_dataset(dataset_name, dataset_root)
    if local_dataset is not None:
        return local_dataset

    if dataset_name in DATASETS:
        hf_name, split = DATASETS[dataset_name]
        return load_dataset(hf_name, split=split)
    raise ValueError(
        f"Unsupported dataset: {dataset_name}. "
        f"No local jsonl found in {dataset_root} and no built-in HF mapping."
    )


def prepare_prompt(
    dataset_name: str, sample: Dict[str, Any], prompt_template: str
) -> str:
    """Construct model input prompt based on sample, modify as needed."""
    if "problem" in sample:
        problem = sample["problem"]
    elif "question" in sample:
        problem = sample["question"]
    elif "prompt" in sample:
        problem = sample["prompt"]
    else:
        raise ValueError(f"Unsupported sample format: {sample}")
    return prompt_template.format(problem=problem)


def score_response(
    dataset_name: str, response: str, sample: Dict[str, Any]
) -> Tuple[float, float]:
    """
    Returns:
      - score: float, score of the response
      - format_score: float, score of the response format
    """
    if "answer" in sample:
        ground_truth = sample["answer"]
    elif "label" in sample:
        ground_truth = sample["label"]
    else:
        raise ValueError(f"Unsupported sample format: {sample}")
    return grade_answer_perl(response, str(ground_truth))


def parse_args() -> Tuple[argparse.Namespace, List[str], List[str]]:
    parser = argparse.ArgumentParser(
        description="Checkpoint-first evaluation script (full / LoRA / BlockTT)."
    )
    parser.add_argument(
        "--result-dir",
        required=True,
        help="Directory for intermediate processes and result output.",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Checkpoint path. Can be a run directory or checkpoint-* directory.",
    )
    parser.add_argument(
        "--ckpt-step",
        default="latest",
        help="Checkpoint step when --ckpt is a run directory. Use 'latest' (default) or an integer.",
    )
    parser.add_argument(
        "--ckpt-type",
        choices=["auto", "full", "lora", "blocktt"],
        default="auto",
        help="Force checkpoint type detection result.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Optional base model override for adapter checkpoints.",
    )
    parser.add_argument(
        "--dataset",
        default="aime2024",
        help="Dataset abbreviation to evaluate, comma separated (e.g., aime2024).",
    )
    parser.add_argument(
        "--dataset-root",
        default="datasets",
        help="Local dataset root directory. If <dataset>.jsonl exists here, it is used first.",
    )
    parser.add_argument(
        "--prompt-format",
        default="lighteval",
        help="Prompt format template to use.",
    )
    parser.add_argument(
        "--rollout-n",
        type=int,
        default=1,
        help="Number of rollouts to generate per sample.",
    )
    parser.add_argument(
        "--serve-port", type=int, default=8000, help="First vLLM backend port number."
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=1,
        help="Number of data parallel backends (start multiple vLLMs).",
    )
    parser.add_argument(
        "--tp-size", type=int, default=1, help="Tensor parallel size passed to vLLM."
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Verify needed GPU count before running, error if insufficient.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.4,
        help="GPU memory utilization limit passed to vLLM (0~1), controls memory usage per card.",
    )
    parser.add_argument(
        "--seed", type=float, default=None, help="Generation random seed."
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Generation temperature."
    )
    parser.add_argument("--top-p", type=float, default=1.0, help="Generation top-p.")
    parser.add_argument(
        "--max-new-tokens", type=int, default=2048, help="Generation length."
    )
    parser.add_argument(
        "--dtype", default="auto", help="Model dtype, used during merging and vLLM."
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="Whether to trust remote code."
    )
    parser.add_argument(
        "--served-model-name", default="eval-model", help="Model name exposed by vLLM."
    )
    parser.add_argument(
        "--api-key", default="dummy", help="API Key for OpenAI compatible interface."
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=3600.0,
        help="Timeout for a single request.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Context length used to cap completion tokens for requests.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="For debugging, limit number of evaluation samples.",
    )
    parser.add_argument(
        "--max-num-request",
        type=int,
        default=None,
        help="Max number of concurrent requests per data parallel (DP) vLLM backend.",
    )

    args, unknown = parser.parse_known_args()

    if args.max_num_request is None:
        args.max_num_request = args.dp_size
    else:
        assert args.max_num_request > 0
        assert args.max_num_request % args.dp_size == 0, (
            f"args.max_num_request({args.max_num_request}) must be divisible by args.dp_size({args.dp_size})"
        )

    vllm_args, leftover = extract_vllm_args(unknown)
    return args, vllm_args, leftover


class ProgressVisualizer:
    def __init__(
        self,
        filepath: Path,
        problem_n: int,
        rollout_n: int,
        completed: Set[Tuple[int, int]],
    ) -> None:
        self.filepath = filepath
        self.problem_n = problem_n
        self.rollout_n = rollout_n
        # Row: rollout_id, Col: problem_id
        self.grid = [["." for _ in range(problem_n)] for _ in range(rollout_n)]
        for pid, rid in completed:
            if 0 <= rid < rollout_n and 0 <= pid < problem_n:
                self.grid[rid][pid] = "X"
        self.lock = asyncio.Lock()
        self._write_sync()

    def _write_sync(self) -> None:
        try:
            with self.filepath.open("w", encoding="utf-8") as f:
                for row in self.grid:
                    f.write("".join(row) + "\n")
        except Exception:
            pass

    async def update(self, problem_id: int, rollout_id: int) -> None:
        if 0 <= rollout_id < self.rollout_n and 0 <= problem_id < self.problem_n:
            async with self.lock:
                if self.grid[rollout_id][problem_id] != "X":
                    self.grid[rollout_id][problem_id] = "X"
                    await asyncio.get_running_loop().run_in_executor(
                        None, self._write_sync
                    )

    def cleanup(self) -> None:
        try:
            if self.filepath.exists():
                self.filepath.unlink()
        except Exception:
            pass


def _iter_dataset_samples(ds, max_samples: int | None):
    if max_samples is None:
        for idx, sample in enumerate(ds):
            yield idx, sample
        return

    for idx, sample in enumerate(ds):
        if idx >= max_samples:
            break
        yield idx, sample


async def generate_responses(
    args: argparse.Namespace,
    dataset_name: str,
    rollout_n: int,
    ports: List[int],
    logger: logging.Logger,
    semaphores: Dict[int, asyncio.Semaphore],
) -> None:
    """
    Asynchronously generate responses and save to output.jsonl.
    Implementation: Read existing output.jsonl to build cache, only generate missing entries.
    Generated results are appended to output.jsonl in real-time.
    """
    dataset_dir = Path(args.result_dir) / dataset_name
    output_file = dataset_dir / "output.jsonl"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_file.touch(exist_ok=True)

    with StageContext(logger, f"C.1[{dataset_name}]", "Reading cached output"):
        generated_results: List[Dict[str, Any]] = []
        cache: Set[Tuple[int, int]] = set()

        if output_file.exists():
            with output_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if (
                            "problem_id" in data
                            and "rollout_id" in data
                            and "response" in data
                            and data["response"] != ""
                            and int(data["rollout_id"]) < rollout_n
                        ):
                            generated_results.append(data)
                            cache.add((data["problem_id"], data["rollout_id"]))
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON line in output.jsonl, skipped.")

        logger.info("Loaded cache entries: %d", len(generated_results))

    with StageContext(logger, f"C.2[{dataset_name}]", "Preparing generation tasks"):
        ds = load_dataset_from_hf(dataset_name, Path(args.dataset_root))
        # max_concurrent_per_dp and semaphores are now handled externally and passed in

        tasks_to_process: List[Tuple[int, int, str, int]] = []
        ports_cycle = len(ports)

        prompt_template = PROMPT_TEMPLATES[args.prompt_format]

        problem_n = len(ds)
        if args.max_samples is not None:
            problem_n = min(problem_n, args.max_samples)

        for idx, sample in _iter_dataset_samples(ds, args.max_samples):
            prompt = prepare_prompt(dataset_name, sample, prompt_template)
            for rollout_id in range(rollout_n):
                if (idx, rollout_id) in cache:
                    continue
                port_idx = (idx * rollout_n + rollout_id) % ports_cycle
                tasks_to_process.append((idx, rollout_id, prompt, port_idx))

        logger.info("New requests to generate: %d", len(tasks_to_process))

        visualizer = ProgressVisualizer(dataset_dir / "process.txt", problem_n, rollout_n, cache)

        if not tasks_to_process:
            logger.info("All requests exist in cache, no generation needed.")
            visualizer.cleanup()
            return

    with StageContext(logger, f"C.3[{dataset_name}]", "Parallel Generation"):
        file_lock = asyncio.Lock()

        async def generate_one_task(
            problem_id: int,
            rollout_id: int,
            prompt: str,
            port_idx: int,
            session: aiohttp.ClientSession,
        ) -> None:
            port = ports[port_idx]
            semaphore = semaphores[port]
            response = ""

            async with semaphore:
                try:
                    response = await generate_with_vllm_async(session, prompt, port, args)
                except Exception as exc:
                    logger.error(
                        "Generation failed problem=%06d rollout=%03d port=%d: %s",
                        problem_id,
                        rollout_id,
                        port,
                        exc,
                    )
                    return

            record = {
                "problem_id": problem_id,
                "rollout_id": rollout_id,
                "response": response,
            }

            generated_results.append(record)

            async with file_lock:
                with output_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            await visualizer.update(problem_id, rollout_id)

        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                generate_one_task(pid, rid, pmt, pidx, session)
                for pid, rid, pmt, pidx in tasks_to_process
            ]
            await asyncio.gather(*tasks)
            visualizer.cleanup()

        logger.info(
            "Dataset %s generation complete, results saved to %s",
            dataset_name,
            output_file,
        )
        expected = len(tasks_to_process) + len(cache)
        actual = len(generated_results)
        if actual < expected:
            raise RuntimeError(
                f"Generation incomplete for dataset={dataset_name}: "
                f"expected {expected} entries but got {actual}. "
                "Please reduce max_new_tokens / max_model_len or inspect vLLM errors."
            )


def evaluate_dataset_results(
    args: argparse.Namespace,
    dataset_name: str,
    rollout_n: int,
    logger: logging.Logger,
) -> Dict[str, Dict[int, float]]:
    """
    Evaluation stage: Read output.jsonl, score and generate result.jsonl, return stats metrics.
    """
    dataset_dir = Path(args.result_dir) / dataset_name
    output_file = dataset_dir / "output.jsonl"
    result_file = dataset_dir / "result.jsonl"
    result_json_file = dataset_dir / "result.json"

    with StageContext(logger, f"D.1[{dataset_name}]", "Loading model output"):
        if not output_file.exists():
            raise ValueError(f"output.jsonl not found, cannot evaluate: {dataset_name}")

        outputs_map: Dict[int, List[Tuple[int, str]]] = {}
        with output_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if "problem_id" in d and "rollout_id" in d:
                        outputs_map.setdefault(d["problem_id"], []).append(
                            (d["rollout_id"], d.get("response", ""))
                        )
                except json.JSONDecodeError:
                    pass

    with StageContext(logger, f"D.2[{dataset_name}]", "Loading original dataset"):
        ds = load_dataset_from_hf(dataset_name, Path(args.dataset_root))

    with StageContext(logger, f"D.3[{dataset_name}]", "Parallel Evaluation & Metrics"):
        raw_stats_list: List[Dict[str, Any]] = []

        prompt_template = PROMPT_TEMPLATES[args.prompt_format]
        with result_file.open("w", encoding="utf-8") as rf:
            for idx, sample in _iter_dataset_samples(ds, args.max_samples):
                problem_id = idx
                prompt = prepare_prompt(dataset_name, sample, prompt_template)

                rollouts = outputs_map.get(problem_id, [])
                # Sort by rollout_id
                rollouts.sort(key=lambda x: x[0])
                rollout_dict = {r[0]: r[1] for r in rollouts}

                responses: List[str] = []
                scores: List[float] = []
                format_scores: List[float] = []

                for rid in range(rollout_n):
                    if rid not in rollout_dict:
                        raise ValueError(
                            f"Missing result: problem_id={problem_id} rollout_id={rid}. Please check if generation requests failed."
                        )
                    resp = rollout_dict.get(rid, "")
                    responses.append(resp)

                    if resp:
                        score, format_score = score_response(dataset_name, resp, sample)
                    else:
                        score, format_score = 0.0, 0.0
                    scores.append(score)
                    format_scores.append(format_score)

                if scores:
                    avg_val = statistics.mean(scores)
                    max_val = max(scores)
                    min_val = min(scores)
                    try:
                        std_val = statistics.stdev(scores)
                    except statistics.StatisticsError:
                        std_val = 0.0
                else:
                    avg_val = max_val = min_val = std_val = 0.0

                format_score_avg = statistics.mean(format_scores) if format_scores else 0.0

                record = {
                    "problem_id": problem_id,
                    "prompt": prompt,
                    "responses": responses,
                    "scores": scores,
                    "avg": avg_val,
                    "max": max_val,
                    "min": min_val,
                    "std": std_val,
                    "format_score_avg": format_score_avg,
                    "data_source": dataset_name,
                }
                rf.write(json.dumps(record, ensure_ascii=False) + "\n")

                raw_stats_list.append(
                    {
                        "problem_id": problem_id,
                        "avg": avg_val,
                        "max": max_val,
                        "min": min_val,
                        "std": std_val,
                        "format_score_avg": format_score_avg,
                    }
                )

    with StageContext(logger, f"D.4[{dataset_name}]", "Summarizing and writing files"):
        if raw_stats_list:
            summary = {
                "avg": statistics.mean(x["avg"] for x in raw_stats_list),
                "max": statistics.mean(x["max"] for x in raw_stats_list),
                "min": statistics.mean(x["min"] for x in raw_stats_list),
                "std": statistics.mean(x["std"] for x in raw_stats_list),
                "format_score_avg": statistics.mean(x["format_score_avg"] for x in raw_stats_list),
            }
        else:
            summary = {
                "avg": 0.0,
                "max": 0.0,
                "min": 0.0,
                "std": 0.0,
                "format_score_avg": 0.0,
            }

        response_example = []
        if 0 in outputs_map and outputs_map[0]:
            response_example = [outputs_map[0][0]]

        final_json = {
            "data_source": dataset_name,
            "rollout_n": rollout_n,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": summary,
            "raw": raw_stats_list,
            "response_example": response_example,
        }

        with result_json_file.open("w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=2, ensure_ascii=False)

        logger.info(
            "Evaluation complete, results written to %s and %s",
            result_file,
            result_json_file,
        )

    return {"summary": summary}


async def main() -> None:
    args, vllm_args, leftover = parse_args()
    logger = setup_logging(Path(args.result_dir))
    if leftover:
        logger.warning(
            "Detected unrecognized arguments (will be ignored): %s", leftover
        )

    with StageContext(logger, "A", "Resolve Checkpoint and Prepare Model"):
        model_path, resolved_ckpt = merge_model_if_needed(args, Path(args.result_dir), logger)

    with StageContext(logger, "B", "Start vLLM Backends"):
        processes, ports = start_vllm_processes(model_path, args, vllm_args, logger)
        atexit.register(stop_vllm_processes, processes, logger)

        def handle_signal(signum, frame):  # noqa: ANN001
            logger.warning(
                "Received signal %d, preparing to clean up and exit.", signum
            )
            stop_vllm_processes(processes, logger)
            sys.exit(1)

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        for proc, port in zip(processes, ports):
            if not wait_for_vllm_ready(
                port,
                proc,
                timeout=300,
                logger=logger,
                api_key=args.api_key,
            ):
                stop_vllm_processes(processes, logger)
                sys.exit(1)

    # Initialize global semaphores
    dp_size = max(1, args.dp_size)
    max_concurrent_per_dp = max(1, args.max_num_request // dp_size)
    semaphores = {port: asyncio.Semaphore(max_concurrent_per_dp) for port in ports}
    logger.info(
        "Global concurrency control initialized: Max concurrency per DP process=%d",
        max_concurrent_per_dp,
    )

    async def process_dataset_task(
        args: argparse.Namespace,
        dataset_name: str,
        rollout_n: int,
        ports: List[int],
        logger: logging.Logger,
        semaphores: Dict[int, asyncio.Semaphore],
    ) -> None:
        with StageContext(
            logger, f"C[{dataset_name}]", "Dataset Generation (Cache/Gen)"
        ):
            await generate_responses(
                args, dataset_name, rollout_n, ports, logger, semaphores
            )

        with StageContext(logger, f"D[{dataset_name}]", "Evaluation & Statistics"):
            await asyncio.to_thread(
                evaluate_dataset_results, args, dataset_name, rollout_n, logger
            )

    datasets_to_run = [item.strip() for item in args.dataset.split(",") if item.strip()]
    tasks = []

    for task_abbr in datasets_to_run:
        if "@" in task_abbr:
            dataset_name = task_abbr.split("@")[0]
            rollout_n = int(task_abbr.split("@")[1])
        else:
            dataset_name = task_abbr
            rollout_n = args.rollout_n

        tasks.append(
            asyncio.create_task(
                process_dataset_task(
                    args, dataset_name, rollout_n, ports, logger, semaphores
                )
            )
        )

    if tasks:
        logger.info(
            "Submitted %d dataset tasks concurrently, starting execution...", len(tasks)
        )
        await asyncio.gather(*tasks)
    else:
        logger.warning("No dataset tasks to execute.")

    stop_vllm_processes(processes, logger)
    logger.info("All evaluation processes completed.")

    if resolved_ckpt.checkpoint_type in {"lora", "blocktt"}:
        merged_model_dir = Path(args.result_dir) / "model"
        if merged_model_dir.exists():
            logger.info("Deleting merged model directory: %s", merged_model_dir)
            try:
                shutil.rmtree(merged_model_dir)
                logger.info("Merged model directory deleted.")
            except Exception as e:
                logger.warning("Failed to delete merged model directory: %s", e)


if __name__ == "__main__":
    asyncio.run(main())
