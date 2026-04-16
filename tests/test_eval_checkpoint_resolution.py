import json
from pathlib import Path

import pytest

from perl.eval import resolve_checkpoint_spec


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_resolve_latest_checkpoint_from_run_dir(tmp_path: Path):
    run_dir = tmp_path / "run"

    ckpt64 = run_dir / "checkpoint-64"
    ckpt320 = run_dir / "checkpoint-320"

    _write_json(
        ckpt64 / "adapter_config.json",
        {"peft_type": "LORA", "base_model_name_or_path": "Qwen/Qwen3-1.7B"},
    )
    (ckpt64 / "adapter_model.safetensors").write_text("x", encoding="utf-8")

    _write_json(
        ckpt320 / "adapter_config.json",
        {"peft_type": "LORA", "base_model_name_or_path": "Qwen/Qwen3-1.7B"},
    )
    (ckpt320 / "adapter_model.safetensors").write_text("x", encoding="utf-8")

    spec = resolve_checkpoint_spec(run_dir, ckpt_step="latest")

    assert spec.checkpoint_dir.name == "checkpoint-320"
    assert spec.checkpoint_step == 320
    assert spec.checkpoint_type == "lora"


def test_resolve_specific_checkpoint_step(tmp_path: Path):
    run_dir = tmp_path / "run"
    ckpt64 = run_dir / "checkpoint-64"

    _write_json(
        ckpt64 / "adapter_config.json",
        {"peft_type": "LORA", "base_model_name_or_path": "Qwen/Qwen3-1.7B"},
    )
    (ckpt64 / "adapter_model.safetensors").write_text("x", encoding="utf-8")

    spec = resolve_checkpoint_spec(run_dir, ckpt_step="64")
    assert spec.checkpoint_dir.name == "checkpoint-64"
    assert spec.checkpoint_step == 64



def test_detect_full_checkpoint(tmp_path: Path):
    ckpt = tmp_path / "checkpoint-128"
    _write_json(ckpt / "config.json", {"architectures": ["Qwen3ForCausalLM"]})
    (ckpt / "model.safetensors").write_text("x", encoding="utf-8")

    spec = resolve_checkpoint_spec(ckpt, ckpt_step="latest")

    assert spec.checkpoint_type == "full"
    assert spec.base_model_name_or_path is None


def test_detect_blocktt_checkpoint(tmp_path: Path):
    ckpt = tmp_path / "checkpoint-64"
    _write_json(
        ckpt / "adapter_config.json",
        {"peft_type": "BLOCKTT", "base_model_name_or_path": "Qwen/Qwen3-1.7B"},
    )
    (ckpt / "adapter_model.safetensors").write_text("x", encoding="utf-8")

    spec = resolve_checkpoint_spec(ckpt, ckpt_step="latest")

    assert spec.checkpoint_type == "blocktt"


def test_model_override_takes_precedence_for_adapter(tmp_path: Path):
    ckpt = tmp_path / "checkpoint-64"
    _write_json(
        ckpt / "adapter_config.json",
        {"peft_type": "LORA", "base_model_name_or_path": "Qwen/FromConfig"},
    )
    (ckpt / "adapter_model.safetensors").write_text("x", encoding="utf-8")

    spec = resolve_checkpoint_spec(
        ckpt,
        ckpt_step="latest",
        model_override="Qwen/OverrideModel",
    )

    assert spec.base_model_name_or_path == "Qwen/OverrideModel"


def test_fail_when_adapter_missing_base_model_and_override(tmp_path: Path):
    ckpt = tmp_path / "checkpoint-64"
    _write_json(ckpt / "adapter_config.json", {"peft_type": "LORA"})
    (ckpt / "adapter_model.safetensors").write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="Base model could not be resolved"):
        resolve_checkpoint_spec(ckpt, ckpt_step="latest")


def test_fail_fast_on_legacy_blocktt_layout(tmp_path: Path):
    run_dir = tmp_path / "blocktt_run"
    ckpt = run_dir / "checkpoint-64"
    ckpt.mkdir(parents=True)
    (ckpt / "README.md").write_text("library_name: peft\nbase_model: Qwen/Qwen3-1.7B", encoding="utf-8")

    (run_dir / "output.log").write_text(
        "ValueError: Unknown PEFT type passed: BLOCKTT",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="legacy/failed BlockTT checkpoint"):
        resolve_checkpoint_spec(run_dir, ckpt_step="latest")
