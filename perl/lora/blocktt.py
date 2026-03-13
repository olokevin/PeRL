"""BlockTT (Block Tensor-Train) PEFT integration for PeRL.

Port of BTTLayer from lora-without-regret/btt_layer.py, restructured as a
HuggingFace PEFT-compatible custom tuner (following the SliceFine pattern).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn

from peft.config import PeftConfig
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_adapters_to_merge


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

VALID_S_MERGED_TO = {"frozen", "trainable", "output", "input", "split", "keep"}


def _closest_factor_pair(d: int) -> tuple[int, int]:
    """Return the factor pair (a, b) of *d* with the smallest difference."""
    root = int(d ** 0.5)
    best_a, best_b = 1, d
    best_diff = best_b - best_a
    for a in range(1, root + 1):
        if d % a == 0:
            b = d // a
            diff = abs(b - a)
            if diff < best_diff:
                best_a, best_b, best_diff = a, b, diff
    return best_a, best_b


def _resolve_blocktt_trainable_sides(
    left_size: int,
    right_size: int,
    train_position: str,
) -> tuple[bool, bool]:
    """Decide which core(s) to train.  Tie-breaks favour the left core."""
    if train_position not in {"small", "large", "both"}:
        raise ValueError("BlockTT train_position must be one of: small, large, both")

    if train_position == "both":
        return True, True
    if train_position == "small":
        train_left = left_size <= right_size
        return train_left, not train_left
    # large
    train_left = left_size >= right_size
    return train_left, not train_left


def resolve_blocktt_s_merged_to(
    train_position: str,
    s_merged_to: str | None = None,
    left_size: int | None = None,
    right_size: int | None = None,
) -> str:
    """Resolve the ``s_merged_to`` strategy to a concrete target."""
    if s_merged_to is None:
        if train_position == "both":
            return "split"
        s_merged_to = "frozen"

    if s_merged_to not in VALID_S_MERGED_TO:
        raise ValueError(
            "s_merged_to must be one of: frozen, trainable, output, input, split, keep"
        )

    if s_merged_to in {"output", "input", "split", "keep"}:
        return s_merged_to

    if left_size is None or right_size is None:
        raise ValueError("left_size and right_size are required for frozen/trainable aliases")

    train_left, train_right = _resolve_blocktt_trainable_sides(
        left_size=left_size,
        right_size=right_size,
        train_position=train_position,
    )
    if train_left and train_right:
        raise ValueError(
            "BlockTT s_merged_to frozen/trainable is invalid when both cores are trainable. "
            "Use output, input, or split."
        )

    if s_merged_to == "trainable":
        return "output" if train_left else "input"
    # frozen
    return "input" if train_left else "output"


# ---------------------------------------------------------------------------
# BTTLayer
# ---------------------------------------------------------------------------

class BTTLayer(nn.Module, BaseTunerLayer):
    """PEFT-compatible Block Tensor-Train layer replacing ``nn.Linear``.

    Parameter shapes (canonical 2-core layout):
        btt_r : (n, b, m * rank)
        btt_l : (m, rank * n, a)
        btt_s : (m, n, rank)  [optional, frozen]
    """

    adapter_layer_names = ("btt_r", "btt_l")
    other_param_names = ("btt_s",)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int | str,
        bias: bool = True,
        decomp_mode: str = "square",
    ) -> None:
        nn.Module.__init__(self)
        BaseTunerLayer.__init__(self)

        # BaseTunerLayer's `bias` property delegates to get_base_layer().bias.
        # We store a lightweight nn.Module as base_layer so that chain terminates
        # without recursion.  The real bias lives on this container.
        _base = nn.Module()
        if bias:
            _base.register_parameter("bias", nn.Parameter(torch.zeros(out_features)))
        else:
            _base.register_parameter("bias", None)
        self.base_layer = _base

        mode_aliases = {
            "input_block": "input_one_block",
            "output_block": "output_one_block",
        }
        decomp_mode = mode_aliases.get(decomp_mode, decomp_mode)

        self.in_features = in_features
        self.out_features = out_features
        self.decomp_mode = decomp_mode

        # Compute block structure
        out_blocks, out_block_size = _closest_factor_pair(out_features)
        in_blocks, in_block_size = _closest_factor_pair(in_features)

        if decomp_mode == "square":
            m, a = out_blocks, out_block_size
            n, b = in_blocks, in_block_size
        elif decomp_mode == "output_one_block":
            m, a = 1, out_features
            n, b = in_blocks, in_block_size
        elif decomp_mode == "input_one_block":
            m, a = out_blocks, out_block_size
            n, b = 1, in_features
        else:
            raise ValueError(
                "decomp_mode must be one of: square, output_one_block, input_one_block"
            )

        self.b, self.n, self.m, self.a = b, n, m, a

        # Resolve rank
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

        # Default-initialise parameters (will be overwritten by init_from_linear_weight)
        self.btt_r = nn.Parameter(torch.zeros(self.n, self.b, self.m * self.rank))
        self.btt_l = nn.Parameter(torch.zeros(self.m, self.rank * self.n, self.a))
        self.register_parameter("btt_s", None)

        # BaseTunerLayer bookkeeping
        self._disable_adapters = False
        self.merged_adapters: list[str] = []

    # ---- SVD-based initialisation from a dense weight -------------------

    @torch.no_grad()
    def init_from_linear_weight(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        s_merged_to: str | None = None,
        train_position: str = "small",
    ) -> None:
        """Initialise cores from a dense ``(out_features, in_features)`` weight via SVD."""
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"Weight shape must be {(self.out_features, self.in_features)}, "
                f"got {tuple(weight.shape)}"
            )

        param_dtype = weight.dtype
        # Dense weight -> block matrix batch (m*n, a, b)
        blocks = weight.reshape(self.m, self.a, self.n, self.b)
        blocks = blocks.permute(0, 2, 1, 3).reshape(self.m * self.n, self.a, self.b)

        svd_dtype = (
            torch.float32
            if param_dtype in (torch.float16, torch.bfloat16)
            else param_dtype
        )
        U, S, Vh = torch.linalg.svd(blocks.to(dtype=svd_dtype), full_matrices=False)

        max_svd_rank = min(self.a, self.b)
        use_rank = min(self.rank, max_svd_rank)

        core_l = torch.zeros(
            self.m * self.n, self.a, self.rank,
            device=weight.device, dtype=param_dtype,
        )
        core_r = torch.zeros(
            self.m * self.n, self.rank, self.b,
            device=weight.device, dtype=param_dtype,
        )

        merge_target = resolve_blocktt_s_merged_to(
            train_position=train_position,
            s_merged_to=s_merged_to,
            left_size=self.m * self.rank * self.n * self.a,
            right_size=self.n * self.b * self.m * self.rank,
        )

        u_used = U[:, :, :use_rank].to(dtype=param_dtype)
        vh_used = Vh[:, :use_rank, :].to(dtype=param_dtype)
        s_used = torch.clamp(S[:, :use_rank], min=0).to(dtype=param_dtype)

        if merge_target == "keep":
            core_l[:, :, :use_rank] = u_used
            core_r[:, :use_rank, :] = vh_used
            s_keep = torch.zeros(
                self.m * self.n, self.rank,
                device=weight.device, dtype=param_dtype,
            )
            s_keep[:, :use_rank] = s_used
            self.btt_s = nn.Parameter(
                s_keep.reshape(self.m, self.n, self.rank), requires_grad=False,
            )
        elif merge_target == "split":
            sqrt_s = torch.sqrt(s_used)
            core_l[:, :, :use_rank] = u_used * sqrt_s.unsqueeze(1)
            core_r[:, :use_rank, :] = sqrt_s.unsqueeze(-1) * vh_used
        elif merge_target == "output":
            core_l[:, :, :use_rank] = u_used * s_used.unsqueeze(1)
            core_r[:, :use_rank, :] = vh_used
        else:  # input
            core_l[:, :, :use_rank] = u_used
            core_r[:, :use_rank, :] = s_used.unsqueeze(-1) * vh_used

        if merge_target != "keep":
            self.btt_s = None

        # Pack into canonical storage layout
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
            self.bias.data.zero_()

    # ---- Dense reconstruction -------------------------------------------

    @torch.no_grad()
    def materialize_dense_weight(self) -> torch.Tensor:
        """Reconstruct the full ``(out_features, in_features)`` dense weight."""
        r = self.btt_r.reshape(self.n, self.b, self.m, self.rank).permute(2, 0, 3, 1)
        l = self.btt_l.reshape(self.m, self.n, self.rank, self.a)
        if self.btt_s is not None:
            l = l * self.btt_s.unsqueeze(-1)
        w_blocks = torch.einsum("mnra,mnrb->mnab", l, r)
        return w_blocks.permute(0, 2, 1, 3).reshape(self.out_features, self.in_features)

    # ---- Forward --------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"BTTLayer expected last dim {self.in_features}, got {x.shape[-1]}"
            )

        orig_shape = x.shape
        x = x.reshape(-1, self.n, self.b)
        batch_n = x.shape[0]
        x_t = x.transpose(0, 1).contiguous()  # (n, B, b)

        # Step 1: (n, B, b) @ (n, b, m*r) -> (n, B, m*r)
        inner = torch.bmm(x_t, self.btt_r)
        inner = inner.reshape(self.n, batch_n, self.m, self.rank)
        inner = inner.permute(2, 1, 0, 3).contiguous()  # (m, B, n, r)

        # Step 2: (m, B, n*r) @ (m, n*r, a) -> (m, B, a)
        btt_l = self.btt_l
        if self.btt_s is not None:
            btt_l = (
                self.btt_l.reshape(self.m, self.n, self.rank, self.a)
                * self.btt_s.unsqueeze(-1)
            ).reshape(self.m, self.rank * self.n, self.a)

        out = torch.bmm(
            inner.reshape(self.m, batch_n, self.rank * self.n),
            btt_l,
        )
        out = out.permute(1, 0, 2).contiguous().reshape(*orig_shape[:-1], self.out_features)

        if self.bias is not None:
            out = out + self.bias

        return out

    def set_adapter(self, adapter_names, inference_mode=False):
        """Override: trainability is managed by BlockTTModel._configure_trainability."""
        pass

    def extra_repr(self) -> str:
        return (
            f"mode={self.decomp_mode}, "
            f"blocks=({self.m}x{self.n}), "
            f"block_size=({self.a}x{self.b}), "
            f"rank={self.rank}, "
            f"btt_r={tuple(self.btt_r.shape)}, "
            f"btt_l={tuple(self.btt_l.shape)}, "
            f"bias={self.bias is not None}"
        )


# ---------------------------------------------------------------------------
# BlockTTConfig
# ---------------------------------------------------------------------------

@dataclass
class BlockTTConfig(PeftConfig):
    decomp_mode: str = field(
        default="input_one_block",
        metadata={"help": "Block decomposition mode: square, input_one_block, output_one_block"},
    )
    train_position: str = field(
        default="small",
        metadata={"help": "Which core to train: small, large, both"},
    )
    s_merged_to: str = field(
        default="frozen",
        metadata={"help": "Singular value merge strategy: frozen, trainable, output, input, split, keep"},
    )
    blocktt_rank: Union[str, int] = field(
        default="full",
        metadata={"help": "TT rank: 'full' or a positive integer"},
    )
    train_bias: bool = field(
        default=True,
        metadata={"help": "Whether to train bias parameters"},
    )
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex to target"},
    )

    def __post_init__(self):
        self.peft_type = "BLOCKTT"
        self.target_modules = (
            set(self.target_modules)
            if isinstance(self.target_modules, list)
            else self.target_modules
        )


# ---------------------------------------------------------------------------
# BlockTTModel (custom PEFT tuner)
# ---------------------------------------------------------------------------

class BlockTTModel(BaseTuner):
    """PEFT tuner that replaces target ``nn.Linear`` modules with ``BTTLayer``."""

    prefix: str = "btt_"
    tuner_layer_cls = BTTLayer

    def __init__(self, model, config, adapter_name="default"):
        super().__init__(model, config, adapter_name)
        self._configure_trainability(config, adapter_name)

    @staticmethod
    def _prepare_adapter_config(peft_config: PeftConfig, model_config: dict) -> PeftConfig:
        return peft_config

    def _configure_trainability(self, config, adapter_name: str) -> None:
        """Freeze everything then selectively enable gradients on BTT cores."""
        if isinstance(config, dict):
            cfg = config.get(adapter_name, config)
        else:
            cfg = config

        # Freeze all parameters first
        for p in self.model.parameters():
            p.requires_grad = False

        train_position = getattr(cfg, "train_position", "small")
        train_bias = getattr(cfg, "train_bias", True)

        for module in self.model.modules():
            if not isinstance(module, BTTLayer):
                continue

            left_size = module.btt_l.numel()
            right_size = module.btt_r.numel()

            train_left, train_right = _resolve_blocktt_trainable_sides(
                left_size=left_size,
                right_size=right_size,
                train_position=train_position,
            )

            module.btt_l.requires_grad = train_left
            module.btt_r.requires_grad = train_right
            if module.btt_s is not None:
                module.btt_s.requires_grad = False

            if module.bias is not None:
                module.bias.requires_grad = train_bias

    def _create_and_replace(
        self,
        peft_config: BlockTTConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
        **kwargs,
    ) -> None:
        if isinstance(target, BTTLayer):
            # Already a BTTLayer; skip
            return

        if not isinstance(target, nn.Linear):
            return

        rank = peft_config.blocktt_rank
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

    def merge_adapter(self, adapter_names: Optional[List[str]] = None) -> None:
        """Materialize dense weights for vLLM sync."""
        for module in self.model.modules():
            if not isinstance(module, BTTLayer):
                continue
            dense = module.materialize_dense_weight()
            module.register_parameter(
                "weight", nn.Parameter(dense, requires_grad=False),
            )
            if adapter_names:
                module.merged_adapters.extend(adapter_names)
            else:
                module.merged_adapters.append("default")

    def unmerge_adapter(self) -> None:
        """Remove materialized weights and re-enable trainable core gradients."""
        for module in self.model.modules():
            if not isinstance(module, BTTLayer):
                continue
            if hasattr(module, "weight") and module.weight is not None:
                delattr(module, "weight")
            module.merged_adapters.clear()

        # Re-enable gradients on trainable cores
        config = self.peft_config.get("default", next(iter(self.peft_config.values())))
        self._configure_trainability(config, "default")

    def enable_adapter_layers(self) -> None:
        for module in self.model.modules():
            if isinstance(module, BTTLayer):
                module._disable_adapters = False

    def disable_adapter_layers(self) -> None:
        for module in self.model.modules():
            if isinstance(module, BTTLayer):
                module._disable_adapters = True


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_blocktt_method():
    """Register BlockTT as a PEFT method in HF PEFT's mapping dicts."""
    import peft.mapping

    peft.mapping.PEFT_TYPE_TO_CONFIG_MAPPING["BLOCKTT"] = BlockTTConfig
    peft.mapping.PEFT_TYPE_TO_TUNER_MAPPING["BLOCKTT"] = BlockTTModel
