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

Neural network module replacing `nn.Linear`. Core parameters:

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

**Initialization:** SVD-based from the original dense weight of the replaced `nn.Linear` layer. Singular values distributed according to `s_merged_to` strategy.

#### BlockTTConfig

PEFT config class (extends HF PEFT's `PeftConfig`):

```python
@dataclass
class BlockTTConfig(PeftConfig):
    decomp_mode: str = "input_one_block"    # or "output_one_block"
    train_position: str = "small"           # "small", "large", or "both"
    s_merged_to: str = "frozen"             # "frozen", "trainable", "output", "input", "split", "keep"
    blocktt_rank: str = "full"              # "full" or positive integer
    train_bias: bool = True
    target_modules: list[str] | None = None
```

Registered as peft type `"BLOCKTT"` via `register_blocktt_method()`.

#### BlockTTModel

Custom tuner class (extends HF PEFT's `BaseTuner`):

- **`_create_and_replace()`**: For each target module, replaces `nn.Linear` with `BTTLayer`. Initializes from the original weight via SVD decomposition. Configures trainability based on `train_position`.

- **`merge_adapter()`**: For each `BTTLayer`:
  1. Call `materialize_dense_weight()` to reconstruct the full dense matrix
  2. Store result as `module.weight` (`nn.Parameter`, `requires_grad=False`)
  3. Also expose `module.bias` if present
  4. Hide BTT core parameters from `named_parameters()` iteration

  After merge, TRL's `_move_model_to_vllm()` sees standard `{module_name}.weight` parameters with shapes matching vLLM's expectations.

- **`unmerge_adapter()`**: Reverse of merge:
  1. Remove the materialized `module.weight`
  2. Restore BTT cores as the active parameters
  3. Re-enable gradients on trainable cores

- **`enable_adapter_layers()` / `disable_adapter_layers()`**: Standard PEFT hooks for controlling trainability.

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
    config = BlockTTConfig(
        decomp_mode=getattr(args.peft, "decomp_mode", "input_one_block"),
        train_position=getattr(args.peft, "train_position", "small"),
        s_merged_to=getattr(args.peft, "s_merged_to", "frozen"),
        blocktt_rank=getattr(args.peft, "blocktt_rank", "full"),
        train_bias=getattr(args.peft, "train_bias", True),
        target_modules=args.peft.target_modules,
    )
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
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

## Scope Exclusions

- **No Muon optimizer support** — uses standard AdamW via PeRL's trainer
- **No `lr_act` / gating** — `lr_act=False`, no activation between cores
- **No fractional rank** — only integer or `"full"` string
- **No custom optimizer** — `apply_blocktt()` returns `(None, model)`

## Testing Strategy

1. **Parameter count test**: Add `"blocktt"` to `perl/test.py` — verify trainable parameter counting works
2. **Smoke test**: Run `dapo_blocktt.sh` with `max_steps=4` — verify training loop completes without errors
3. **Merge/unmerge round-trip**: Verify that `merge_adapter()` → `unmerge_adapter()` produces identical forward pass outputs (no weight corruption)

## File Summary

| File | Action | Description |
|------|--------|-------------|
| `perl/lora/blocktt.py` | Create | BTTLayer, BlockTTConfig, BlockTTModel, register function |
| `perl/lora/adapter.py` | Modify | Add import, registration, and `apply_blocktt()` |
| `perl/config/config.py` | Modify | Add 5 optional BlockTT fields to PeftConfig |
| `scripts/trl/openr1/dapo_blocktt.sh` | Create | Training script with BlockTT defaults |
| `perl/test.py` | Modify | Include "blocktt" in parameter count test |
