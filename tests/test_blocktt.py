"""Tests for the BlockTT PEFT integration module."""
import pytest
import torch
import torch.nn as nn

from perl.lora.blocktt import (
    BTTLayer,
    BlockTTConfig,
    BlockTTModel,
    _closest_factor_pair,
    _resolve_blocktt_trainable_sides,
    register_blocktt_method,
    resolve_blocktt_s_merged_to,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_model():
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    config.num_hidden_layers = 2
    config.hidden_size = 128
    config.intermediate_size = 256
    config.num_attention_heads = 4
    config.num_key_value_heads = 4
    return AutoModelForCausalLM.from_config(config).to(torch.float32)


# ---------------------------------------------------------------------------
# TestBTTLayer
# ---------------------------------------------------------------------------

class TestBTTLayer:
    """Unit tests for the BTTLayer class."""

    def _make_layer(self, in_f=128, out_f=256, rank="full", decomp_mode="input_one_block", dtype=torch.float32):
        """Helper: create a BTTLayer initialized from a random linear weight."""
        linear = nn.Linear(in_f, out_f, bias=True).to(dtype)
        layer = BTTLayer(
            in_features=in_f,
            out_features=out_f,
            rank=rank,
            bias=True,
            decomp_mode=decomp_mode,
        ).to(dtype)
        layer.init_from_linear_weight(
            linear.weight.data,
            linear.bias.data,
            s_merged_to="split",
            train_position="small",
        )
        return layer, linear

    def test_forward_shape(self):
        layer, _ = self._make_layer()
        x = torch.randn(2, 5, 128)
        out = layer(x)
        assert out.shape == (2, 5, 256)

    def test_materialize_matches_original_float32(self):
        layer, linear = self._make_layer(dtype=torch.float32)
        dense = layer.materialize_dense_weight()
        # With full rank and float32, the SVD reconstruction should be very close
        assert torch.allclose(dense, linear.weight.data, atol=1e-5), (
            f"Max diff: {(dense - linear.weight.data).abs().max().item()}"
        )

    def test_materialize_matches_original_bfloat16(self):
        layer, linear = self._make_layer(dtype=torch.bfloat16)
        dense = layer.materialize_dense_weight()
        # bfloat16 SVD is done in float32 then cast back; precision loss expected
        assert torch.allclose(
            dense.float(), linear.weight.data.float(), atol=2e-3
        ), f"Max diff: {(dense.float() - linear.weight.data.float()).abs().max().item()}"

    def test_merge_unmerge_roundtrip_float32(self):
        layer, _ = self._make_layer(dtype=torch.float32)
        x = torch.randn(2, 5, 128)
        out_forward = layer(x)
        dense = layer.materialize_dense_weight()
        out_dense = x @ dense.t()
        if layer.bias is not None:
            out_dense = out_dense + layer.bias
        assert torch.allclose(out_forward, out_dense, atol=1e-5), (
            f"Max diff: {(out_forward - out_dense).abs().max().item()}"
        )

    def test_merge_unmerge_roundtrip_bfloat16(self):
        layer, _ = self._make_layer(dtype=torch.bfloat16)
        x = torch.randn(2, 5, 128, dtype=torch.bfloat16)
        out_forward = layer(x)
        dense = layer.materialize_dense_weight()
        out_dense = x @ dense.t()
        if layer.bias is not None:
            out_dense = out_dense + layer.bias
        # bmm vs matmul accumulation order differs; bfloat16 has limited mantissa
        assert torch.allclose(out_forward, out_dense, atol=1e-2), (
            f"Max diff: {(out_forward - out_dense).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# TestBlockTTConfig
# ---------------------------------------------------------------------------

class TestBlockTTConfig:

    def test_defaults(self):
        cfg = BlockTTConfig(target_modules=["q_proj"])
        assert cfg.peft_type == "BLOCKTT"
        assert cfg.decomp_mode == "input_one_block"
        assert cfg.train_position == "small"
        assert cfg.s_merged_to == "frozen"
        assert cfg.blocktt_rank == "full"
        assert cfg.train_bias is True

    def test_target_modules_set_conversion(self):
        cfg = BlockTTConfig(target_modules=["q_proj", "v_proj", "q_proj"])
        assert isinstance(cfg.target_modules, set)
        assert cfg.target_modules == {"q_proj", "v_proj"}


# ---------------------------------------------------------------------------
# TestBlockTTModel
# ---------------------------------------------------------------------------

class TestBlockTTModel:

    @pytest.fixture(autouse=True)
    def _register(self):
        register_blocktt_method()

    def test_get_peft_model(self):
        from peft import get_peft_model

        model = _make_tiny_model()
        config = BlockTTConfig(
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            blocktt_rank="full",
            decomp_mode="input_one_block",
        )
        peft_model = get_peft_model(model, config)

        # Should have some trainable parameters
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
        assert trainable > 0
        assert trainable < total

        # Verify BTTLayers exist in the model
        btt_layers = [m for m in peft_model.modules() if isinstance(m, BTTLayer)]
        assert len(btt_layers) > 0

    def test_merge_unmerge_via_tuner(self):
        from peft import get_peft_model

        model = _make_tiny_model()
        config = BlockTTConfig(
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            blocktt_rank="full",
            decomp_mode="input_one_block",
        )
        peft_model = get_peft_model(model, config)

        # Capture trainable state before merge
        trainable_before = {
            n for n, p in peft_model.named_parameters() if p.requires_grad
        }
        assert len(trainable_before) > 0

        # Merge
        peft_model.merge_adapter()

        # All BTTLayers should have .weight after merge
        for module in peft_model.modules():
            if isinstance(module, BTTLayer):
                assert hasattr(module, "weight") and module.weight is not None
                assert len(module.merged_adapters) > 0

        # Unmerge
        peft_model.unmerge_adapter()

        # .weight should be gone, cores should be trainable again
        for module in peft_model.modules():
            if isinstance(module, BTTLayer):
                assert not hasattr(module, "weight") or module.weight is None
                assert len(module.merged_adapters) == 0

        # Trainable params should be restored
        trainable_after = {
            n for n, p in peft_model.named_parameters() if p.requires_grad
        }
        assert trainable_after == trainable_before

    def test_merged_named_parameters_prefix(self):
        from peft import get_peft_model

        model = _make_tiny_model()
        config = BlockTTConfig(
            target_modules=["q_proj", "k_proj"],
            blocktt_rank="full",
            decomp_mode="input_one_block",
        )
        peft_model = get_peft_model(model, config)
        peft_model.merge_adapter()

        names = dict(peft_model.named_parameters())

        # Materialized .weight should NOT have btt_ prefix
        weight_names = [n for n in names if n.endswith(".weight")]
        for wn in weight_names:
            assert "btt_" not in wn, f"Merged weight has btt_ prefix: {wn}"

        # BTT core params should still have btt_ prefix
        core_names = [n for n in names if "btt_r" in n or "btt_l" in n]
        assert len(core_names) > 0, "Expected BTT core parameters to still exist"
        for cn in core_names:
            assert "btt_" in cn

        peft_model.unmerge_adapter()
