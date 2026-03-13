# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PeRL (Parameter-Efficient Reinforcement Learning) is a framework for combining PEFT methods with RL training. It evaluates 15+ PEFT methods (LoRA, DoRA, MiSS, AdaLoRA, etc.) using GRPO on DeepSeek-R1-Distill models for mathematical reasoning tasks.

## Build & Run Commands

```bash
# Environment setup
python -m venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install flash-attn --no-cache-dir --no-build-isolation  # optional

# Fetch submodules (veRL, MikaEval)
git submodule update --init --recursive

# Sanity check: verify trainable param counts across PEFT methods
python perl/test.py

# Training (standard LoRA, 1.5B, 4 GPUs)
bash scripts/trl/openr1/dapo_lora.sh

# Smoke test: clone a script and reduce --config.training.max_steps
```

Prefer adapting an existing script under `scripts/` instead of assembling `accelerate launch` commands manually.

## Architecture

**Entry point:** `run.py` uses Fire CLI to dispatch to `perl/train.py` (training) or `perl/eval.py` (evaluation). All config is passed via `--config.*` flags mapped to dataclasses in `perl/config/config.py`.

**Training flow:** Parse config → load tokenizer → load dataset (pluggable loaders in `perl/data/`) → load model → apply PEFT via `apply_peft()` dispatcher → run TRL's `GRPOTrainer` → save checkpoints.

**PEFT adapter dispatch** (`perl/lora/adapter.py`): `PEFT_TYPE_TO_FUNCTION_MAPPING` dict maps peft type strings to functions. Each returns `(optimizer, model)`. Some methods (LoRA+, LoRA-FA) use custom optimizers with differentiated learning rates.

**Dataset loaders** (`perl/data/__init__.py`): Pattern-matched on dataset name (contains "r1" → OpenR1, "tinyzero" → TinyZero, etc.). Each returns `{train_dataset, test_dataset, reward_functions, reward_weights}`.

**Reward model server** (`perl/rm/`): FastAPI async server using SGLang backend for math answer extraction and verification.

**Evaluation** (`perl/eval.py`): Multi-stage pipeline — merge adapter → start vLLM → generate → evaluate. Supports AIME, AMC, MATH-500, Minerva, HMMT benchmarks.

**External modules:** `modules/verl` (advanced RL framework) and `modules/eval` (MikaEval) are git submodules.

## Conventions

- Python: PEP 8, 4-space indent. `snake_case` for files/functions/variables, `PascalCase` for dataclasses.
- Shell scripts: uppercase env vars (`OUTPUT_DIR`, `LOG_FILE`), deterministic paths under `outputs/`.
- New CLI options must map cleanly to `--config.*` arguments via typed dataclass fields.
- Commit prefixes: `[feat]`, `[rebuild]`, `[scripts]`, `[debug]`, `[doc]`, `[data]`.
- PRs must include: what/why, exact run commands, hardware context (GPU count, accelerate config), and linked tracking (WandB/issue).
- When changing training logic, include a reproducible command and observed result.
