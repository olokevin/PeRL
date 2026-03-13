# BlockTT Migration to PeRL — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate BlockTT (Block Tensor-Train) fine-tuning as a new PEFT method in PeRL, enabling fair comparison with existing methods under the same training infrastructure.

**Architecture:** Port BTTLayer from `lora-without-regret/btt_layer.py`, wrap it as a HF PEFT-compatible tuner (like SliceFine), register it in PeRL's PEFT dispatcher, and add config fields + training script.

**Tech Stack:** PyTorch, HuggingFace PEFT (`BaseTuner`, `BaseTunerLayer`), TRL GRPOTrainer, DeepSpeed ZeRO-2

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `perl/lora/blocktt.py` | Create | BTTLayer, BlockTTConfig, BlockTTModel tuner, `register_blocktt_method()` |
| `perl/lora/adapter.py` | Modify | Import + register BlockTT, add `apply_blocktt()` function |
| `perl/config/config.py` | Modify | Add 5 optional BlockTT fields to `PeftConfig` dataclass |
| `perl/train.py` | Modify | Fix dispatcher to use `apply_peft()` instead of hardcoded `apply_lora()` |
| `scripts/trl/openr1/dapo_blocktt.sh` | Create | Training script with BlockTT defaults |
| `tests/test_blocktt.py` | Create | Unit tests for BTTLayer, merge/unmerge round-trip, config |

---

## Chunk 1: Core BTTLayer and BlockTT PEFT Integration

### Task 1: Create `perl/lora/blocktt.py` — BTTLayer class

**Files:**
- Create: `perl/lora/blocktt.py`

- [ ] **Step 1: Write the BTTLayer class**

Port the BTTLayer from `/home/yequan/Project/lora/lora-without-regret/btt_layer.py`. This is the core neural network module that replaces `nn.Linear` with block tensor-train factorization.

Include these helper functions from the source:
- `_closest_factor_pair(d)` — finds closest factor pair for block decomposition
- `_resolve_blocktt_trainable_sides(left_size, right_size, train_position)` — determines which core to train
- `resolve_blocktt_s_merged_to(...)` — resolves singular value merge strategy

```python
# perl/lora/blocktt.py
# copyright (c) 2025, mikastars39.org
# All rights reserved.
# This source code is licensed under the Apache-2.0 License.
# See the LICENSE file in the root directory for details.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union, List

import torch
import torch.nn as nn
import numpy as np

from peft.config import PeftConfig
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer


VALID_S_MERGED_TO = {"frozen", "trainable", "output", "input", "split", "keep"}


def _closest_factor_pair(d):
    root = int(d ** 0.5)
    best_a = 1
    best_b = d
    best_diff = best_b - best_a
    for a in range(1, root + 1):
        if d % a == 0:
            b = d // a
            diff = abs(b - a)
            if diff < best_diff:
                best_a, best_b, best_diff = a, b, diff
    return best_a, best_b


def _resolve_blocktt_trainable_sides(left_size, right_size, train_position):
    if train_position not in {"small", "large", "both"}:
        raise ValueError("BlockTT train_position must be one of: small, large, both")
    if train_position == "both":
        return True, True
    if train_position == "small":
        train_left = left_size <= right_size
        return train_left, not train_left
    train_left = left_size >= right_size
    return train_left, not train_left


def resolve_blocktt_s_merged_to(train_position, s_merged_to=None, left_size=None, right_size=None):
    if s_merged_to is None:
        if train_position == "both":
            return "split"
        s_merged_to = "frozen"
    if s_merged_to not in VALID_S_MERGED_TO:
        raise ValueError("s_merged_to must be one of: frozen, trainable, output, input, split, keep")
    if s_merged_to in {"output", "input", "split", "keep"}:
        return s_merged_to
    if left_size is None or right_size is None:
        raise ValueError("left_size and right_size are required for frozen/trainable aliases")
    train_left, train_right = _resolve_blocktt_trainable_sides(
        left_size=left_size, right_size=right_size, train_position=train_position,
    )
    if train_left and train_right:
        raise ValueError(
            "BlockTT s_merged_to frozen/trainable is invalid when both cores are trainable. "
            "Use output, input, or split."
        )
    if s_merged_to == "trainable":
        return "output" if train_left else "input"
    return "input" if train_left else "output"


class BTTLayer(nn.Module, BaseTunerLayer):
    """
    Block Tensor-Train layer replacing nn.Linear.
    Uses dual inheritance (nn.Module + BaseTunerLayer) for PEFT compatibility.

    Parameter shapes:
      btt_r: (n, b, m * rank)
      btt_l: (m, rank * n, a)
    """

    adapter_layer_names = ("btt_r", "btt_l")
    other_param_names = ("btt_s",)

    def __init__(
        self,
        in_features,
        out_features,
        rank,
        bias=True,
        decomp_mode="input_one_block",
    ):
        nn.Module.__init__(self)
        BaseTunerLayer.__init__(self)

        mode_aliases = {
            "input_block": "input_one_block",
            "output_block": "output_one_block",
        }
        decomp_mode = mode_aliases.get(decomp_mode, decomp_mode)

        self.in_features = in_features
        self.out_features = out_features
        self.decomp_mode = decomp_mode

        out_blocks, out_block_size = _closest_factor_pair(out_features)
        in_blocks, in_block_size = _closest_factor_pair(in_features)

        if decomp_mode == "square":
            m, a, n, b = out_blocks, out_block_size, in_blocks, in_block_size
        elif decomp_mode == "output_one_block":
            m, a, n, b = 1, out_features, in_blocks, in_block_size
        elif decomp_mode == "input_one_block":
            m, a, n, b = out_blocks, out_block_size, 1, in_features
        else:
            raise ValueError("decomp_mode must be one of: square, output_one_block, input_one_block")

        self.b, self.n, self.m, self.a = b, n, m, a

        if isinstance(rank, str):
            if rank != "full":
                raise ValueError("rank as string must be 'full'")
            resolved_rank = min(self.b, self.a)
        elif isinstance(rank, int):
            if rank <= 0:
                raise ValueError("rank as integer must be > 0")
            resolved_rank = rank
        else:
            raise TypeError("rank must be an int or 'full'")

        self.rank = resolved_rank

        # Initialize with placeholder values — will be overwritten by init_from_linear_weight
        self.btt_r = nn.Parameter(torch.zeros(self.n, self.b, self.m * self.rank))
        self.btt_l = nn.Parameter(torch.zeros(self.m, self.rank * self.n, self.a))
        self.register_parameter("btt_s", None)

        if not bias:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(torch.zeros(out_features))

    @torch.no_grad()
    def init_from_linear_weight(self, weight, bias=None, s_merged_to=None, train_position="small"):
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"Linear weight shape must be {(self.out_features, self.in_features)}, got {tuple(weight.shape)}"
            )

        param_dtype = weight.dtype
        blocks = weight.reshape(self.m, self.a, self.n, self.b)
        blocks = blocks.permute(0, 2, 1, 3).reshape(self.m * self.n, self.a, self.b)
        svd_dtype = torch.float32 if param_dtype in (torch.float16, torch.bfloat16) else param_dtype
        U, S, Vh = torch.linalg.svd(blocks.to(dtype=svd_dtype), full_matrices=False)

        max_svd_rank = min(self.a, self.b)
        use_rank = min(self.rank, max_svd_rank)

        core_l = torch.zeros(self.m * self.n, self.a, self.rank, device=weight.device, dtype=param_dtype)
        core_r = torch.zeros(self.m * self.n, self.rank, self.b, device=weight.device, dtype=param_dtype)

        merge_target = resolve_blocktt_s_merged_to(
            train_position=train_position, s_merged_to=s_merged_to,
            left_size=self.btt_l.numel(), right_size=self.btt_r.numel(),
        )
        u_used = U[:, :, :use_rank].to(dtype=param_dtype)
        vh_used = Vh[:, :use_rank, :].to(dtype=param_dtype)
        s_used = torch.clamp(S[:, :use_rank], min=0).to(dtype=param_dtype)

        if merge_target == "keep":
            core_l[:, :, :use_rank] = u_used
            core_r[:, :use_rank, :] = vh_used
            s_keep = torch.zeros(self.m * self.n, self.rank, device=weight.device, dtype=param_dtype)
            s_keep[:, :use_rank] = s_used
            self.btt_s = nn.Parameter(s_keep.reshape(self.m, self.n, self.rank), requires_grad=False)
        elif merge_target == "split":
            sqrt_s = torch.sqrt(s_used)
            core_l[:, :, :use_rank] = u_used * sqrt_s.unsqueeze(1)
            core_r[:, :use_rank, :] = sqrt_s.unsqueeze(-1) * vh_used
        elif merge_target == "output":
            core_l[:, :, :use_rank] = u_used * s_used.unsqueeze(1)
            core_r[:, :use_rank, :] = vh_used
        else:  # "input"
            core_l[:, :, :use_rank] = u_used
            core_r[:, :use_rank, :] = s_used.unsqueeze(-1) * vh_used
        if merge_target != "keep":
            self.btt_s = None

        core_l = core_l.reshape(self.m, self.n, self.a, self.rank)
        core_r = core_r.reshape(self.m, self.n, self.rank, self.b)
        packed_l = core_l.permute(0, 1, 3, 2).reshape(self.m, self.rank * self.n, self.a)
        packed_r = core_r.permute(1, 3, 0, 2).reshape(self.n, self.b, self.m * self.rank)

        self.btt_l.data.copy_(packed_l)
        self.btt_r.data.copy_(packed_r)

        if bias is not None:
            if self.bias is None:
                raise ValueError("BTTLayer has no bias but bias tensor was provided")
            self.bias.data.copy_(bias.to(device=self.bias.device, dtype=self.bias.dtype))
        elif self.bias is not None:
            raise ValueError("BTTLayer has bias but no source bias was provided")

    @torch.no_grad()
    def materialize_dense_weight(self):
        r = self.btt_r.reshape(self.n, self.b, self.m, self.rank).permute(2, 0, 3, 1)
        l = self.btt_l.reshape(self.m, self.n, self.rank, self.a)
        if self.btt_s is not None:
            l = l * self.btt_s.unsqueeze(-1)
        w_blocks = torch.einsum("mnra,mnrb->mnab", l, r)
        return w_blocks.permute(0, 2, 1, 3).reshape(self.out_features, self.in_features)

    def forward(self, x):
        if x.shape[-1] != self.in_features:
            raise ValueError(f"BTTLayer expected last dim {self.in_features}, got {x.shape[-1]}")

        orig_shape = x.shape
        x = x.reshape(-1, self.n, self.b)
        batch_n = x.shape[0]
        x_t = x.transpose(0, 1).contiguous()

        # (n, B, b) @ (n, b, m*r) -> (n, B, m*r)
        inner = torch.bmm(x_t, self.btt_r)
        inner = inner.reshape(self.n, batch_n, self.m, self.rank)
        inner = inner.permute(2, 1, 0, 3).contiguous()  # (m, B, n, r)

        # (m, B, n*r) @ (m, n*r, a) -> (m, B, a)
        btt_l = self.btt_l
        if self.btt_s is not None:
            btt_l = (
                self.btt_l.reshape(self.m, self.n, self.rank, self.a)
                * self.btt_s.unsqueeze(-1)
            ).reshape(self.m, self.rank * self.n, self.a)
        out = torch.bmm(inner.reshape(self.m, batch_n, self.rank * self.n), btt_l)
        out = out.permute(1, 0, 2).contiguous().reshape(*orig_shape[:-1], self.out_features)

        if self.bias is not None:
            out += self.bias
        return out

    def extra_repr(self):
        return (
            f"mode: {self.decomp_mode}, "
            f"blocks: ({self.m}x{self.n}), block_size: ({self.a}x{self.b}), "
            f"rank: {self.rank}, "
            f"btt_r: {self.btt_r.shape}, btt_l: {self.btt_l.shape}, "
            f"bias: {self.bias.shape if self.bias is not None else False}"
        )
```

- [ ] **Step 2: Commit BTTLayer**

```bash
git add perl/lora/blocktt.py
git commit -m "[feat] add BTTLayer core module for BlockTT PEFT integration"
```

---

### Task 2: Add BlockTTConfig and BlockTTModel to `perl/lora/blocktt.py`

**Files:**
- Modify: `perl/lora/blocktt.py` (append after BTTLayer class)

- [ ] **Step 3: Write failing test for BlockTTConfig and BlockTTModel**

Create `tests/test_blocktt.py` with tests that verify:
1. BlockTTConfig creates with correct defaults
2. BlockTTModel can replace target modules in a small model
3. Merge/unmerge round-trip preserves forward pass output

```python
# tests/test_blocktt.py
import torch
import torch.nn as nn
import pytest
from transformers import AutoConfig, AutoModelForCausalLM


def _make_tiny_model():
    """Create a small Qwen2 model (2 layers) for fast testing."""
    config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    config.num_hidden_layers = 2
    config.hidden_size = 128
    config.intermediate_size = 256
    config.num_attention_heads = 4
    config.num_key_value_heads = 4
    return AutoModelForCausalLM.from_config(config).to(torch.float32)


class TestBTTLayer:
    def test_forward_shape(self):
        from perl.lora.blocktt import BTTLayer

        layer = BTTLayer(in_features=64, out_features=128, rank="full", bias=True)
        weight = torch.randn(128, 64)
        bias = torch.randn(128)
        layer.init_from_linear_weight(weight, bias, s_merged_to="split", train_position="small")
        x = torch.randn(2, 64)
        out = layer(x)
        assert out.shape == (2, 128)

    def test_materialize_matches_original_float32(self):
        from perl.lora.blocktt import BTTLayer

        weight = torch.randn(128, 64, dtype=torch.float32)
        layer = BTTLayer(in_features=64, out_features=128, rank="full", bias=False)
        layer.init_from_linear_weight(weight, s_merged_to="split", train_position="small")
        reconstructed = layer.materialize_dense_weight()
        assert torch.allclose(weight, reconstructed, atol=1e-5), (
            f"Max diff: {(weight - reconstructed).abs().max().item()}"
        )

    def test_materialize_matches_original_bfloat16(self):
        from perl.lora.blocktt import BTTLayer

        weight = torch.randn(128, 64, dtype=torch.bfloat16)
        layer = BTTLayer(in_features=64, out_features=128, rank="full", bias=False)
        layer.to(dtype=torch.bfloat16)
        layer.init_from_linear_weight(weight, s_merged_to="split", train_position="small")
        reconstructed = layer.materialize_dense_weight()
        assert torch.allclose(weight.float(), reconstructed.float(), atol=1e-3), (
            f"Max diff: {(weight.float() - reconstructed.float()).abs().max().item()}"
        )

    def test_merge_unmerge_roundtrip_float32(self):
        from perl.lora.blocktt import BTTLayer

        weight = torch.randn(128, 64, dtype=torch.float32)
        bias = torch.randn(128, dtype=torch.float32)
        layer = BTTLayer(in_features=64, out_features=128, rank="full", bias=True)
        layer.init_from_linear_weight(weight, bias, s_merged_to="split", train_position="small")

        x = torch.randn(4, 64)
        out_before = layer(x)

        # Merge: materialize dense weight
        dense = layer.materialize_dense_weight()
        layer.register_parameter("weight", nn.Parameter(dense, requires_grad=False))
        layer.merged_adapters.append("default")

        # Unmerge: remove dense weight
        delattr(layer, "weight")
        layer.merged_adapters.clear()

        out_after = layer(x)
        assert torch.allclose(out_before, out_after, atol=1e-5), (
            f"Max diff: {(out_before - out_after).abs().max().item()}"
        )

    def test_merge_unmerge_roundtrip_bfloat16(self):
        from perl.lora.blocktt import BTTLayer

        weight = torch.randn(128, 64, dtype=torch.bfloat16)
        bias = torch.randn(128, dtype=torch.bfloat16)
        layer = BTTLayer(in_features=64, out_features=128, rank="full", bias=True)
        layer.to(dtype=torch.bfloat16)
        layer.init_from_linear_weight(weight, bias, s_merged_to="split", train_position="small")

        x = torch.randn(4, 64, dtype=torch.bfloat16)
        out_before = layer(x)

        dense = layer.materialize_dense_weight()
        layer.register_parameter("weight", nn.Parameter(dense, requires_grad=False))
        layer.merged_adapters.append("default")

        delattr(layer, "weight")
        layer.merged_adapters.clear()

        out_after = layer(x)
        assert torch.allclose(out_before.float(), out_after.float(), atol=1e-3), (
            f"Max diff: {(out_before.float() - out_after.float()).abs().max().item()}"
        )


class TestBlockTTConfig:
    def test_defaults(self):
        from perl.lora.blocktt import BlockTTConfig

        config = BlockTTConfig()
        assert config.peft_type == "BLOCKTT"
        assert config.decomp_mode == "input_one_block"
        assert config.train_position == "small"
        assert config.s_merged_to == "frozen"
        assert config.blocktt_rank == "full"
        assert config.train_bias is True

    def test_target_modules_set_conversion(self):
        from perl.lora.blocktt import BlockTTConfig

        config = BlockTTConfig(target_modules=["q_proj", "v_proj"])
        assert isinstance(config.target_modules, set)
        assert config.target_modules == {"q_proj", "v_proj"}


class TestBlockTTModel:
    def test_get_peft_model(self):
        from perl.lora.blocktt import BlockTTConfig, register_blocktt_method
        from peft import get_peft_model

        register_blocktt_method()
        model = _make_tiny_model()

        config = BlockTTConfig(
            blocktt_rank="full",
            target_modules=["q_proj", "v_proj"],
            train_position="small",
            s_merged_to="frozen",
        )
        peft_model = get_peft_model(model, config)

        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
        assert trainable > 0, "No trainable parameters found"
        assert trainable < total, "All parameters are trainable — trainability config failed"

    def test_merge_unmerge_via_tuner(self):
        """Test merge/unmerge through BlockTTModel methods (not raw BTTLayer)."""
        from perl.lora.blocktt import BlockTTConfig, BTTLayer, register_blocktt_method
        from peft import get_peft_model

        register_blocktt_method()
        model = _make_tiny_model()

        config = BlockTTConfig(
            blocktt_rank="full",
            target_modules=["q_proj", "v_proj"],
            train_position="small",
            s_merged_to="split",
        )
        peft_model = get_peft_model(model, config)

        # Verify forward pass is identical before and after merge/unmerge
        x = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            out_before = peft_model(x).logits

        peft_model.merge_adapter()
        peft_model.unmerge_adapter()

        with torch.no_grad():
            out_after = peft_model(x).logits

        assert torch.allclose(out_before, out_after, atol=1e-5), (
            f"Max diff: {(out_before - out_after).abs().max().item()}"
        )

    def test_merged_named_parameters_prefix(self):
        """After merge, materialized .weight has no btt_ prefix; BTT cores still have btt_ prefix."""
        from perl.lora.blocktt import BlockTTConfig, BTTLayer, register_blocktt_method
        from peft import get_peft_model

        register_blocktt_method()
        model = _make_tiny_model()

        config = BlockTTConfig(
            blocktt_rank="full",
            target_modules=["q_proj"],
            train_position="small",
            s_merged_to="split",
        )
        peft_model = get_peft_model(model, config)
        peft_model.merge_adapter()

        param_names = [n for n, _ in peft_model.named_parameters()]
        # Materialized weight should exist without btt_ prefix
        weight_params = [n for n in param_names if n.endswith(".weight") and "q_proj" in n]
        assert len(weight_params) > 0, "No materialized .weight found after merge"

        # BTT core params should still have btt_ in name
        btt_params = [n for n in param_names if "btt_r" in n or "btt_l" in n]
        assert len(btt_params) > 0, "BTT core params missing after merge"

        peft_model.unmerge_adapter()
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
cd /home/yequan/Project/lora/PeRL && python -m pytest tests/test_blocktt.py -v
```

Expected: FAIL — `BlockTTConfig` and `BlockTTModel` not yet defined.

- [ ] **Step 5: Write BlockTTConfig class**

Append to `perl/lora/blocktt.py`:

```python
@dataclass
class BlockTTConfig(PeftConfig):
    decomp_mode: str = field(default="input_one_block", metadata={"help": "Decomposition mode: square, input_one_block, output_one_block"})
    train_position: str = field(default="small", metadata={"help": "Which core to train: small, large, both"})
    s_merged_to: str = field(default="frozen", metadata={"help": "SVD singular value merge strategy"})
    blocktt_rank: Union[str, int] = field(default="full", metadata={"help": "TT rank: 'full' or positive integer"})
    train_bias: bool = field(default=True, metadata={"help": "Whether to train bias parameters"})
    target_modules: Optional[Union[List[str], str]] = field(default=None, metadata={"help": "Modules to replace"})

    def __post_init__(self):
        self.peft_type = "BLOCKTT"
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list)
            else self.target_modules
        )
```

- [ ] **Step 6: Write BlockTTModel tuner class**

Append to `perl/lora/blocktt.py`:

```python
class BlockTTModel(BaseTuner):
    prefix: str = "btt_"

    def __init__(self, model, config, adapter_name="default"):
        super().__init__(model, config, adapter_name)
        self._configure_trainability(config, adapter_name)

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        """Required by BaseTuner. Returns config unchanged since BlockTT has no model-dependent config."""
        return peft_config

    def _configure_trainability(self, config, adapter_name):
        """Set requires_grad based on train_position and train_bias."""
        peft_config = config[adapter_name] if isinstance(config, dict) else config

        # Freeze all params first
        for p in self.model.parameters():
            p.requires_grad = False

        for module in self.model.modules():
            if not isinstance(module, BTTLayer):
                continue

            left_size = module.btt_l.numel()
            right_size = module.btt_r.numel()
            train_left, train_right = _resolve_blocktt_trainable_sides(
                left_size=left_size,
                right_size=right_size,
                train_position=peft_config.train_position,
            )
            module.btt_l.requires_grad = train_left
            module.btt_r.requires_grad = train_right
            if module.btt_s is not None:
                module.btt_s.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = peft_config.train_bias

    def _create_and_replace(self, peft_config, adapter_name, target, target_name, parent, current_key, **kwargs):
        if isinstance(target, BTTLayer):
            return  # already replaced

        if not isinstance(target, nn.Linear):
            return

        # Parse rank
        rank = peft_config.blocktt_rank
        if isinstance(rank, str) and rank != "full":
            rank = int(rank)

        btt_layer = BTTLayer(
            in_features=target.in_features,
            out_features=target.out_features,
            rank=rank,
            bias=(target.bias is not None),
            decomp_mode=peft_config.decomp_mode,
        ).to(device=target.weight.device, dtype=target.weight.dtype)

        btt_layer.init_from_linear_weight(
            target.weight.data,
            target.bias.data if target.bias is not None else None,
            s_merged_to=peft_config.s_merged_to,
            train_position=peft_config.train_position,
        )

        self._replace_module(parent, target_name, btt_layer, target)

    def merge_adapter(self, adapter_names=None):
        for module in self.model.modules():
            if not isinstance(module, BTTLayer):
                continue
            if hasattr(module, "weight"):
                continue  # already merged
            dense = module.materialize_dense_weight()
            module.register_parameter("weight", nn.Parameter(dense, requires_grad=False))
            module.merged_adapters.append("default")

    def unmerge_adapter(self):
        peft_config = self.peft_config[self.active_adapter]
        for module in self.model.modules():
            if not isinstance(module, BTTLayer):
                continue
            if not hasattr(module, "weight") or module.weight is None:
                continue
            delattr(module, "weight")
            module.merged_adapters.clear()

            # Re-enable gradients on trainable cores
            left_size = module.btt_l.numel()
            right_size = module.btt_r.numel()
            train_left, train_right = _resolve_blocktt_trainable_sides(
                left_size=left_size,
                right_size=right_size,
                train_position=peft_config.train_position,
            )
            module.btt_l.requires_grad = train_left
            module.btt_r.requires_grad = train_right
            if module.bias is not None:
                module.bias.requires_grad = peft_config.train_bias

    def enable_adapter_layers(self):
        for module in self.model.modules():
            if isinstance(module, BTTLayer):
                module._disable_adapters = False

    def disable_adapter_layers(self):
        for module in self.model.modules():
            if isinstance(module, BTTLayer):
                module._disable_adapters = True
```

- [ ] **Step 7: Write register_blocktt_method()**

Append to `perl/lora/blocktt.py`:

```python
def register_blocktt_method():
    import peft.mapping
    peft.mapping.PEFT_TYPE_TO_CONFIG_MAPPING["BLOCKTT"] = BlockTTConfig
    peft.mapping.PEFT_TYPE_TO_TUNER_MAPPING["BLOCKTT"] = BlockTTModel
```

- [ ] **Step 8: Run BTTLayer and Config tests**

```bash
cd /home/yequan/Project/lora/PeRL && python -m pytest tests/test_blocktt.py::TestBTTLayer tests/test_blocktt.py::TestBlockTTConfig -v
```

Expected: PASS for BTTLayer and BlockTTConfig tests.

- [ ] **Step 9: Commit BlockTTConfig, BlockTTModel, and registration**

```bash
git add perl/lora/blocktt.py tests/test_blocktt.py
git commit -m "[feat] add BlockTTConfig, BlockTTModel tuner, and registration for PEFT integration"
```

---

## Chunk 2: PeRL Integration (adapter, config, train, script)

### Task 3: Add BlockTT fields to PeftConfig

**Files:**
- Modify: `perl/config/config.py:18-30`

- [ ] **Step 10: Add BlockTT-specific fields to PeftConfig**

Add these optional fields after the existing `target_modules` field in the `PeftConfig` dataclass:

```python
# BlockTT-specific fields (only used when type="blocktt")
decomp_mode: str = "input_one_block"
train_position: str = "small"
s_merged_to: str = "frozen"
blocktt_rank: str = "full"
train_bias: bool = True
```

The full modified dataclass should look like:

```python
@dataclass
class PeftConfig:
    """PEFT configuration settings"""
    type: str = None
    use_peft: bool = True
    task_type: str = "CAUSAL_LM"
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    total_step: int = 1000
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"]
    )
    # BlockTT-specific fields (only used when type="blocktt")
    decomp_mode: str = "input_one_block"
    train_position: str = "small"
    s_merged_to: str = "frozen"
    blocktt_rank: str = "full"
    train_bias: bool = True
```

- [ ] **Step 11: Commit config changes**

```bash
git add perl/config/config.py
git commit -m "[feat] add BlockTT config fields to PeftConfig dataclass"
```

---

### Task 4: Register BlockTT in adapter dispatcher

**Files:**
- Modify: `perl/lora/adapter.py:1-8` (add import + registration)
- Modify: `perl/lora/adapter.py:193-208` (add to mapping)

- [ ] **Step 12: Add import and registration at top of adapter.py**

After the existing SliceFine import/registration (lines 6-7), add:

```python
from .blocktt import register_blocktt_method
register_blocktt_method()
```

- [ ] **Step 13: Add apply_blocktt function**

Add before the mapping dict (before line 193):

```python
def apply_blocktt(model, args):
    from .blocktt import BlockTTConfig
    from peft import get_peft_model

    raw_rank = getattr(args.peft, "blocktt_rank", "full")
    blocktt_rank = raw_rank if raw_rank == "full" else int(raw_rank)

    config = BlockTTConfig(
        task_type=args.peft.task_type,
        decomp_mode=getattr(args.peft, "decomp_mode", "input_one_block"),
        train_position=getattr(args.peft, "train_position", "small"),
        s_merged_to=getattr(args.peft, "s_merged_to", "frozen"),
        blocktt_rank=blocktt_rank,
        train_bias=getattr(args.peft, "train_bias", True),
        target_modules=args.peft.target_modules,
    )
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    if sum(p.numel() for p in peft_model.parameters() if p.requires_grad) == 0:
        raise ValueError("BlockTT: no trainable parameters found. Check train_position and target_modules.")
    return None, peft_model
```

- [ ] **Step 14: Add blocktt to PEFT_TYPE_TO_FUNCTION_MAPPING**

Add to the mapping dict:

```python
"blocktt": apply_blocktt,
```

- [ ] **Step 15: Commit adapter changes**

```bash
git add perl/lora/adapter.py
git commit -m "[feat] register BlockTT in PEFT adapter dispatcher"
```

---

### Task 5: Fix train.py dispatcher

**Files:**
- Modify: `perl/train.py:95-100`

- [ ] **Step 16: Replace hardcoded apply_lora with apply_peft dispatcher**

Change lines 96-100 from:

```python
    if args.peft.use_peft:
        logger.info(f"Detected PEFT configuration, configuring lora")
        from perl.lora.adapter import apply_lora
        optimizer, model = apply_lora(model, args)
        logger.info(f"Lora configured successfully")
```

To:

```python
    if args.peft.use_peft:
        logger.info(f"Detected PEFT configuration, applying {args.peft.type}")
        from perl.lora.adapter import apply_peft
        optimizer, model = apply_peft(model, args)
        logger.info(f"PEFT ({args.peft.type}) configured successfully")
```

- [ ] **Step 17: Commit train.py fix**

```bash
git add perl/train.py
git commit -m "[feat] use apply_peft dispatcher instead of hardcoded apply_lora in train.py"
```

---

### Task 6: Create training script

**Files:**
- Create: `scripts/trl/openr1/dapo_blocktt.sh`

- [ ] **Step 18: Write the BlockTT training script**

Model on `dapo_dora.sh`, replacing PEFT-specific args:

```bash
unset WANDB_DISABLED
OUTPUT_DIR=outputs/grpo_blocktt_qwen2_5_1_5b_$(date +%Y%m%d_%H%M%S)
LOG_FILE=${OUTPUT_DIR}/output.log

mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info \
    accelerate launch \
    --main_process_port 29503 \
    --config_file scripts/trl/accelerate/ds_zero2_4gpu.yaml \
    run.py train \
    --config.common.seed 42 \
    --config.common.debug false \
    --config.model.model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --config.model.dtype "bfloat16" \
    --config.peft.use_peft true \
    --config.peft.type "blocktt" \
    --config.peft.blocktt_rank "full" \
    --config.peft.decomp_mode "input_one_block" \
    --config.peft.train_position "small" \
    --config.peft.s_merged_to "frozen" \
    --config.peft.train_bias true \
    --config.peft.target_modules '["q_proj","v_proj","k_proj","o_proj","up_proj","down_proj","gate_proj"]' \
    --config.training.learning_rate 1e-5 \
    --config.training.beta 0.0 \
    --config.training.output_dir "${OUTPUT_DIR}" \
    --config.training.run_name "${OUTPUT_DIR}" \
    --config.training.remove_unused_columns false \
    --config.training.gradient_accumulation_steps 8 \
    --config.training.num_train_epochs 1 \
    --config.training.max_completion_length 16384 \
    --config.training.num_generations 8 \
    --config.training.warmup_ratio 0.0 \
    --config.training.max_prompt_length 512 \
    --config.training.logging_steps 1 \
    --config.training.per_device_train_batch_size 4 \
    --config.training.save_strategy "steps" \
    --config.training.save_steps 64 \
    --config.training.max_steps 1024 \
    --config.training.use_vllm true \
    --config.training.top_entropy_quantile 1.0 \
    --config.training.epsilon_high 0.28 \
    --config.training.lr_scheduler_type "constant" \
    --config.training.lr_scheduler_kwargs.min_lr_rate 0.1 \
    --config.training.vllm_mode "colocate" \
    --config.training.vllm_gpu_memory_utilization 0.4 \
    --config.training.use_liger_kernel false \
    --config.training.loss_type "dapo" \
    --config.training.report_to '["wandb"]' \
    --config.logging.trackio_space_id "Open-Tinker/Open-Tinker" \
    --config.logging.trackio_project "grpo-blocktt-qwen2-5-1-5b" \
    --config.logging.wandb_project "grpo-blocktt-qwen2-5-1-5b" \
    --config.dataset.dataset_name_or_path "open-r1/DAPO-Math-17k-Processed" \
    --config.dataset.example_numbers 1000000000 \
    &> ${LOG_FILE}
```

- [ ] **Step 19: Make script executable and commit**

```bash
chmod +x scripts/trl/openr1/dapo_blocktt.sh
git add scripts/trl/openr1/dapo_blocktt.sh
git commit -m "[scripts] add BlockTT training script for DeepSeek-R1-Distill-Qwen-1.5B"
```

---

## Chunk 3: Verification

### Task 7: Run parameter count test

**Files:**
- Read: `perl/test.py` (no modifications needed — auto-includes new methods via mapping)

- [ ] **Step 20: Run the parameter count test**

```bash
cd /home/yequan/Project/lora/PeRL && python perl/test.py
```

Expected: BlockTT appears in output with non-zero trainable parameters and a valid percentage.

- [ ] **Step 21: Run full test suite**

```bash
cd /home/yequan/Project/lora/PeRL && python -m pytest tests/test_blocktt.py -v
```

Expected: All tests pass.

- [ ] **Step 22: Run smoke test (4 steps)**

```bash
# Copy the script, override max_steps to 4 for quick validation
OUTPUT_DIR=outputs/blocktt_smoke_test
mkdir -p ${OUTPUT_DIR}
CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info \
    accelerate launch \
    --main_process_port 29503 \
    --config_file scripts/trl/accelerate/ds_zero2_4gpu.yaml \
    run.py train \
    --config.common.seed 42 \
    --config.common.debug true \
    --config.model.model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --config.model.dtype "bfloat16" \
    --config.peft.use_peft true \
    --config.peft.type "blocktt" \
    --config.peft.blocktt_rank "full" \
    --config.peft.decomp_mode "input_one_block" \
    --config.peft.train_position "small" \
    --config.peft.s_merged_to "frozen" \
    --config.peft.train_bias true \
    --config.peft.target_modules '["q_proj","v_proj","k_proj","o_proj","up_proj","down_proj","gate_proj"]' \
    --config.training.learning_rate 1e-5 \
    --config.training.output_dir "${OUTPUT_DIR}" \
    --config.training.max_steps 4 \
    --config.training.save_strategy "no" \
    --config.training.use_vllm true \
    --config.training.vllm_mode "colocate" \
    --config.training.vllm_gpu_memory_utilization 0.4 \
    --config.dataset.dataset_name_or_path "open-r1/DAPO-Math-17k-Processed"
```

Expected: Training completes 4 steps without errors. Verify:
1. "PEFT (blocktt) configured successfully" in logs
2. Non-zero loss values
3. No CUDA errors or shape mismatches
