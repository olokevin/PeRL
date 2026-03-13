# BlockTT Migration to PeRL — Design Spec

## Summary

Integrate the BlockTT (Block Tensor-Train) fine-tuning method from `lora-without-regret` into PeRL as a new PEFT method, enabling fair comparison with existing methods (LoRA, DoRA, MiSS, etc.) under the same training infrastructure.

## Background

BlockTT decomposes weight matrices into a 2-core block tensor-train structure. Instead of adding low-rank adapters on top of frozen weights (like LoRA), it replaces `nn.Linear` layers entirely with `BTTLayer` modules that store factorized cores (`btt_r`, `btt_l`, optional `btt_s`).

Source implementation: `/home/yequan/Project/lora/lora-without-regret/btt_layer.py`

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Integration pattern | Pattern C (custom PEFT tuner, like SliceFine) | BlockTT replaces layers rather than wrapping them; needs custom tuner registration |
| vLLM rollout handling | PEFT-compatible merge/unmerge | Keeps all methods using TRL's GRPOTrainer; fairer comparison, less code duplication |
| Config approach | Add optional fields to existing `PeftConfig` | Follows existing pattern (SliceFine uses `getattr` fallback); consistent `--config.peft.*` CLI |
| Default learning rate | 1e-5 (PeRL standard) | Consistency across methods; users tune separately |
| Default blocktt_rank | `"full"` (lossless) | Preserves BlockTT's known-good default; users specify rank explicitly |
| Custom optimizer | None | Uses PeRL's standard optimizer for fair comparison |

## Architecture

### New File: `perl/lora/blocktt.py`

Port of `btt_layer.py` from `lora-without-regret`, restructured as a HF PEFT-compatible tuner.

#### BTTLayer

Neural network module replacing `nn.Linear`. Uses dual inheritance: extends both `nn.Module` and HF PEFT's `BaseTunerLayer` (following SliceFine's pattern) to get adapter lifecycle methods (`merged_adapters`, `active_adapters`, `_disable_adapters`, `check_adapters_to_merge()`).

**Class attributes:**
- `adapter_layer_names = ("btt_r", "btt_l")` — tells PEFT which attributes are adapter-specific (note: these are raw `nn.Parameter` objects, not `nn.ModuleDict` like SliceFine; this is fine since `BaseTunerLayer` only uses these names for device movement and adapter identification)
- `other_param_names = ("btt_s",)` — frozen metadata parameters

Core parameters:

- `btt_r: Parameter` — right core, shape `(n, b, m * rank)`
- `btt_l: Parameter` — left core, shape `(m, rank * n, a)`
- `btt_s: Parameter | None` — singular values, shape `(m, n, rank)`, frozen unless `s_merged_to="keep"`
- `bias: Parameter | None` — bias vector

Where `n, b` = input block count and size; `m, a` = output block count and size; `rank` = TT rank.

**Forward pass:** Two `torch.bmm` contractions:
```
x → reshape(n, B, b) → bmm(btt_r) → reshape(m, B, n*r) → bmm(btt_l) → reshape(B, out_features)
```

**`materialize_dense_weight()`:** Reconstructs full dense weight via `einsum("mnra,mnrb->mnab", l, r)` then reshapes to `(out_features, in_features)`.

**Initialization:** SVD-based from the original dense weight of the replaced `nn.Linear` layer. Singular values distributed according to `s_merged_to` strategy. The `BTTLayer` is constructed and initialized from the original `nn.Linear`'s weight BEFORE calling `_replace_module()`, ensuring the full weight is available for SVD decomposition. This is safe under DeepSpeed ZeRO-2 (used by PeRL's scripts) since ZeRO-2 partitions optimizer states, not parameters.

#### BlockTTConfig

PEFT config class (extends HF PEFT's `PeftConfig`):

```python
@dataclass
class BlockTTConfig(PeftConfig):
    decomp_mode: str = "input_one_block"    # or "output_one_block"
    train_position: str = "small"           # "small", "large", or "both"
    s_merged_to: str = "frozen"             # "frozen", "trainable", "output", "input", "split", "keep"
    blocktt_rank: Union[str, int] = "full"   # "full" or positive integer (parsed in apply_blocktt)
    train_bias: bool = True
    target_modules: list[str] | None = None

    def __post_init__(self):
        self.peft_type = "BLOCKTT"
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list)
            else self.target_modules
        )
```

Registered as peft type `"BLOCKTT"` via `register_blocktt_method()`.

#### BlockTTModel

Custom tuner class (extends HF PEFT's `BaseTuner`):

**Class attribute:** `prefix = "btt_"` — matches BTT parameter names (`btt_r`, `btt_l`, `btt_s`). This is critical: TRL's `_move_model_to_vllm()` skips any parameter whose name contains the prefix, so BTT core parameters are automatically excluded from vLLM sync while the materialized `.weight` (no `btt_` in name) is sent through.

- **`_create_and_replace()`**: For each target module, constructs a `BTTLayer` initialized from the original `nn.Linear`'s weight via SVD, then calls `_replace_module()` to swap it in. Configures trainability based on `train_position`.

- **`merge_adapter()`**: For each `BTTLayer`:
  1. Call `materialize_dense_weight()` to reconstruct the full dense matrix
  2. Store result as `module.weight` (`nn.Parameter`, `requires_grad=False`)
  3. Track merged state via `self.merged_adapters` (from `BaseTunerLayer`)

  Note: BTT core parameters (`btt_r`, `btt_l`, `btt_s`) do NOT need to be hidden — they remain in `named_parameters()` but TRL's prefix-based skip filter (`if self.model.prefix in name: continue`) automatically excludes them. The materialized `.weight` and existing `.bias` have no `btt_` prefix, so they pass through to vLLM correctly.

  Implementation detail: The materialized weight is registered via `module.register_parameter("weight", nn.Parameter(dense, requires_grad=False))` so it appears in `named_parameters()` and `state_dict()`. Checkpointing should only occur in unmerged state to avoid saving redundant dense weights.

- **`unmerge_adapter()`**: Reverse of merge:
  1. Remove the materialized weight via `delattr(module, "weight")` (safe since it was dynamically added during merge)
  2. Clear merged state tracking via `self.merged_adapters.discard(adapter_name)`
  3. Re-enable gradients on trainable cores

- **`enable_adapter_layers()` / `disable_adapter_layers()`**: Delegates to `BaseTunerLayer` infrastructure for adapter lifecycle control.

#### register_blocktt_method()

Registers into HF PEFT's mapping dicts at import time:
```python
peft.mapping.PEFT_TYPE_TO_CONFIG_MAPPING["BLOCKTT"] = BlockTTConfig
peft.mapping.PEFT_TYPE_TO_TUNER_MAPPING["BLOCKTT"] = BlockTTModel
```

### Modified File: `perl/lora/adapter.py`

```python
from .blocktt import register_blocktt_method
register_blocktt_method()

def apply_blocktt(model, args):
    from .blocktt import BlockTTConfig
    from peft import get_peft_model

    # Parse blocktt_rank: "full" stays as string, otherwise convert to int
    raw_rank = getattr(args.peft, "blocktt_rank", "full")
    blocktt_rank = raw_rank if raw_rank == "full" else int(raw_rank)

    config = BlockTTConfig(
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

PEFT_TYPE_TO_FUNCTION_MAPPING["blocktt"] = apply_blocktt
```

### Modified File: `perl/config/config.py`

Add optional fields to `PeftConfig`:

```python
@dataclass
class PeftConfig:
    # ... existing fields ...
    decomp_mode: str = "input_one_block"
    train_position: str = "small"
    s_merged_to: str = "frozen"
    blocktt_rank: str = "full"
    train_bias: bool = True
```

### New File: `scripts/trl/openr1/dapo_blocktt.sh`

Training script following existing pattern (modeled on `dapo_dora.sh`):

```bash
OUTPUT_DIR=outputs/blocktt
accelerate launch \
  --main_process_port 29501 \
  --config_file scripts/trl/accelerate/ds_zero2_4gpu.yaml \
  run.py train \
  --config.common.seed 42 \
  --config.model.model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
  --config.peft.type "blocktt" \
  --config.peft.blocktt_rank "full" \
  --config.peft.decomp_mode "input_one_block" \
  --config.peft.train_position "small" \
  --config.peft.s_merged_to "frozen" \
  --config.training.learning_rate 1e-5 \
  --config.training.max_steps 1024 \
  --config.dataset.dataset_name_or_path "open-r1/DAPO-Math-17k-Processed" \
  # ... remaining standard args matching other scripts ...
```

## Merge/Unmerge Lifecycle (Critical Path)

This is the most complex part. Here's the exact sequence during each training step:

```
Training step N:
  1. GRPOTrainer calls _move_model_to_vllm()
  2.   → model.merge_adapter()
  3.     → For each BTTLayer: materialize_dense_weight() → store as .weight
  4.   → Iterate model.named_parameters()
  5.     → TRL sees "{module}.weight" with shape (out_features, in_features) ✓
  6.     → TRL cleans PEFT prefixes from names
  7.     → TRL calls vllm_model.load_weights([(name, param.data)])
  8.   → model.unmerge_adapter()
  9.     → For each BTTLayer: remove .weight, restore BTT cores
  10. GRPOTrainer generates rollouts via vLLM (using materialized weights)
  11. GRPOTrainer computes loss and backpropagates
  12.   → Gradients flow through trainable BTT cores only
  13. Optimizer updates trainable cores
```

## Compatibility Notes

- **Gradient checkpointing**: Works transparently — BTTLayer replaces `nn.Linear` at the module level, so transformer-block-level checkpointing recomputes the BTT forward pass as expected.
- **Checkpoint saving/loading**: `register_blocktt_method()` runs at import time in `adapter.py`, so PEFT's `save_pretrained()`/`load_pretrained()` will find the correct config/tuner. The `prefix = "btt_"` ensures PEFT identifies adapter parameters correctly.
- **DeepSpeed ZeRO-2**: Compatible. Parameters are not partitioned under ZeRO-2, so the full weight is available during SVD initialization. ZeRO-3 is NOT supported (would require parameter gathering).

## Scope Exclusions

- **No Muon optimizer support** — uses standard AdamW via PeRL's trainer
- **No `lr_act` / gating** — `lr_act=False`, no activation between cores
- **No fractional rank** — only integer or `"full"` string
- **No custom optimizer** — `apply_blocktt()` returns `(None, model)`

## Testing Strategy

1. **Parameter count test**: Add `"blocktt"` to `perl/test.py` — verify trainable parameter counting works
2. **Smoke test**: Run `dapo_blocktt.sh` with `max_steps=4` — verify training loop completes without errors
3. **Merge/unmerge round-trip**: Verify that `merge_adapter()` → `unmerge_adapter()` produces identical forward pass outputs (use `torch.allclose` with `atol=1e-5` for float32, `atol=1e-3` for bfloat16, since merge uses einsum while forward uses bmm)

## File Summary

| File | Action | Description |
|------|--------|-------------|
| `perl/lora/blocktt.py` | Create | BTTLayer, BlockTTConfig, BlockTTModel, register function |
| `perl/lora/adapter.py` | Modify | Add import, registration, and `apply_blocktt()` |
| `perl/config/config.py` | Modify | Add 5 optional BlockTT fields to PeftConfig |
| `scripts/trl/openr1/dapo_blocktt.sh` | Create | Training script with BlockTT defaults |
| `perl/test.py` | Modify | Include "blocktt" in parameter count test |
