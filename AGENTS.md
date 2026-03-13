# Repository Guidelines

## Project Structure & Module Organization
`perl/` contains the core training code:
- `perl/train.py` and `run.py` are the main training entry points.
- `perl/lora/` implements PEFT adapters (LoRA, DoRA, MiSS, etc.).
- `perl/trainer/` contains GRPO trainer integrations.
- `perl/data/` defines dataset loaders and reward wiring.
- `perl/rm/` contains reward-model server components.

`scripts/trl/` and `scripts/slime/` provide runnable experiment scripts.  
`modules/verl` and `modules/eval` are git submodules for external training/eval stacks.  
`env/requirements_hard.txt` stores pinned environments; `requirements.txt` is the lightweight default.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create/activate local environment.
- `uv pip install -r requirements.txt`: install core dependencies.
- `uv pip install flash-attn --no-cache-dir --no-build-isolation`: optional acceleration package.
- `bash scripts/trl/openr1/dapo_lora.sh`: run a standard LoRA GRPO training job.
- `git submodule update --init --recursive`: fetch required submodule code.

Prefer adapting an existing script under `scripts/` instead of assembling long `accelerate launch` commands manually.

## Coding Style & Naming Conventions
Use Python with 4-space indentation and PEP 8-style spacing.  
Use `snake_case` for files, functions, and variables; use `PascalCase` for dataclasses/config objects (see `perl/config/config.py`).  
Keep config fields explicit and typed; new CLI-configurable options should map cleanly to `--config.*` arguments.

Shell scripts should use uppercase env vars (for example `OUTPUT_DIR`, `LOG_FILE`) and deterministic paths under `outputs/`.

## Testing Guidelines
This repository currently emphasizes script-level validation over a full unit-test suite.
- `python perl/test.py`: sanity-check trainable parameter counts across PEFT methods.
- Run a short smoke training by cloning a script in `scripts/trl/openr1/` and reducing `--config.training.max_steps`.

When changing training logic, include at least one reproducible command and observed result (log snippet or metric).

## Commit & Pull Request Guidelines
Recent history uses bracketed prefixes such as `[feat]`, `[rebuild]`, `[scripts]`, `[debug]`, `[doc]`, and `[data]`. Follow this pattern and keep subjects short and imperative.

PRs should include:
- What changed and why.
- Exact run command(s) used for validation.
- Hardware/runtime context (GPU count, accelerate config).
- Linked issue or experiment tracking link (for example, WandB run).
