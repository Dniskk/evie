"""Tests for transformer architecture."""

import pytest
import torch
import torch.nn as nn

from evie.models.transformer import (
    PositionalEncoding,
    TransformerBlock,
    TransformerDecoder,
    create_causal_mask,
)


class TestPositionalEncoding:
    """Tests for PositionalEncoding."""

    def test_positional_encoding_shape(self) -> None:
        dim = 128
        max_seq = 512
        pos_enc = PositionalEncoding(dim, max_seq)

        x = torch.randn(4, 256, dim)
        output = pos_enc(x)

        assert output.shape == (4, 256, dim)

    def test_positional_encoding_uniqueness(self) -> None:
        dim = 64
        pos_enc = PositionalEncoding(dim, 1024)

        x = torch.randn(1, 10, dim)
        output = pos_enc(x)

        pos1 = output[0, 0, :].clone().detach()
        pos2 = output[0, 1, :].clone().detach()
        pos3 = output[0, 2, :].clone().detach()

        assert not torch.allclose(pos1, pos2)
        assert not torch.allclose(pos2, pos3)

    def test_positional_encoding_dropout(self) -> None:
        dim = 128
        pos_enc = PositionalEncoding(dim, 512, dropout=0.5)
        pos_enc.eval()

        x = torch.randn(2, 100, dim)
        output1 = pos_enc(x)
        output2 = pos_enc(x)

        assert torch.allclose(output1, output2)

    def test_positional_encoding_odd_dimension(self) -> None:
        # Test that odd dimensions are handled correctly
        dim = 63  # Odd dimension
        max_seq = 256
        pos_enc = PositionalEncoding(dim, max_seq)

        x = torch.randn(2, 100, dim)
        output = pos_enc(x)

        assert output.shape == (2, 100, dim)
        # Verify no NaN values
        assert not torch.isnan(output).any()


class TestTransformerBlock:
    """Tests for TransformerBlock."""

    def test_transformer_block_output_shape(self) -> None:
        dim = 128
        num_heads = 8
        hidden_dim = 512
        block = TransformerBlock(dim, num_heads, hidden_dim)

        x = torch.randn(4, 32, dim)
        output = block(x)

        assert output.shape == (4, 32, dim)

    def test_transformer_block_with_mask(self) -> None:
        dim = 64
        num_heads = 4
        hidden_dim = 256
        block = TransformerBlock(dim, num_heads, hidden_dim)

        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, dim)
        mask = torch.ones(batch_size, seq_len, seq_len)

        output = block(x, mask)
        assert output.shape == (batch_size, seq_len, dim)

    def test_transformer_block_residual_connection(self) -> None:
        dim = 128
        num_heads = 8
        hidden_dim = 512
        block = TransformerBlock(dim, num_heads, hidden_dim)
        block.eval()

        x = torch.randn(1, 1, dim)

        with torch.no_grad():
            output = block(x)

        assert output.shape == x.shape
        assert not torch.allclose(output, x)

    def test_transformer_block_causal_mask(self) -> None:
        dim = 64
        num_heads = 4
        hidden_dim = 256
        block = TransformerBlock(dim, num_heads, hidden_dim)
        block.eval()

        seq_len = 8
        x = torch.randn(1, seq_len, dim)

        # Test with causal mask
        causal_mask = create_causal_mask(seq_len)
        output = block(x, causal_mask)
        assert output.shape == (1, seq_len, dim)


class TestTransformerDecoder:
    """Tests for TransformerDecoder."""

    def test_decoder_output_shape(self) -> None:
        vocab_size = 1000
        dim = 128
        num_heads = 8
        num_layers = 2
        hidden_dim = 512

        decoder = TransformerDecoder(
            vocab_size,
            dim,
            num_heads,
            num_layers,
            hidden_dim,
        )

        batch_size = 4
        seq_len = 32
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        output = decoder(x)

        assert output.shape == (batch_size, seq_len, vocab_size)

    def test_decoder_with_different_configs(self) -> None:
        configs = [
            {"vocab_size": 256, "dim": 64, "num_heads": 4, "num_layers": 1},
            {
                "vocab_size": 5000,
                "dim": 256,
                "num_heads": 16,
                "num_layers": 4,
            },
            {"vocab_size": 10000, "dim": 512, "num_heads": 8, "num_layers": 6},
        ]

        for config in configs:
            decoder = TransformerDecoder(
                vocab_size=config["vocab_size"],
                dim=config["dim"],
                num_heads=config["num_heads"],
                num_layers=config["num_layers"],
                hidden_dim=config["dim"] * 4,
            )

            x = torch.randint(0, config["vocab_size"], (2, 16))
            output = decoder(x)

            assert output.shape == (2, 16, config["vocab_size"])

    def test_decoder_gradient_flow(self) -> None:
        vocab_size = 1000
        dim = 128
        num_heads = 8
        num_layers = 2
        hidden_dim = 512

        decoder = TransformerDecoder(
            vocab_size,
            dim,
            num_heads,
            num_layers,
            hidden_dim,
        )
        decoder.train()

        x = torch.randint(0, vocab_size, (2, 16))
        output = decoder(x)

        loss = output.sum()
        loss.backward()

        for param in decoder.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_decoder_inference_deterministic(self) -> None:
        vocab_size = 256
        dim = 64
        num_heads = 4
        num_layers = 2
        hidden_dim = 256

        decoder = TransformerDecoder(
            vocab_size,
            dim,
            num_heads,
            num_layers,
            hidden_dim,
        )
        decoder.eval()

        x = torch.randint(0, vocab_size, (1, 8))

        with torch.no_grad():
            output1 = decoder(x)
            output2 = decoder(x)

        assert torch.allclose(output1, output2)

    def test_decoder_vocab_probability_output(self) -> None:
        vocab_size = 100
        dim = 64
        num_heads = 4
        num_layers = 1
        hidden_dim = 256

        decoder = TransformerDecoder(
            vocab_size,
            dim,
            num_heads,
            num_layers,
            hidden_dim,
        )
        decoder.eval()

        x = torch.randint(0, vocab_size, (2, 4))

        with torch.no_grad():
            logits = decoder(x)

        assert logits.shape == (2, 4, vocab_size)
        assert logits.dtype == torch.float32

    def test_decoder_with_causal_mask(self) -> None:
        vocab_size = 256
        dim = 64
        num_heads = 4
        num_layers = 2
        hidden_dim = 256

        decoder = TransformerDecoder(
            vocab_size,
            dim,
            num_heads,
            num_layers,
            hidden_dim,
        )
        decoder.eval()

        batch_size = 2
        seq_len = 16
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Use causal mask for autoregressive decoding
        causal_mask = create_causal_mask(seq_len)

        with torch.no_grad():
            output = decoder(x, causal_mask)

        assert output.shape == (batch_size, seq_len, vocab_size)

    def test_decoder_invalid_dim_num_heads(self) -> None:
        with pytest.raises(ValueError, match="dim.*must be divisible by num_heads"):
            TransformerDecoder(
                vocab_size=100,
                dim=100,
                num_heads=7,  # 100 not divisible by 7
                num_layers=2,
                hidden_dim=256,
            )

    def test_decoder_invalid_num_layers(self) -> None:
        with pytest.raises(ValueError, match="num_layers must be at least 1"):
            TransformerDecoder(
                vocab_size=100,
                dim=64,
                num_heads=4,
                num_layers=0,  # Invalid
                hidden_dim=256,
            )

    def test_decoder_embedding_scaling(self) -> None:
        vocab_size = 100
        dim = 64
        num_heads = 4
        num_layers = 1
        hidden_dim = 256

        decoder = TransformerDecoder(
            vocab_size,
            dim,
            num_heads,
            num_layers,
            hidden_dim,
        )

        # Verify embedding scale is set correctly
        assert decoder.embedding_scale == (dim**0.5)


class TestCausalMask:
    """Tests for causal mask utility."""

    def test_causal_mask_shape(self) -> None:
        seq_len = 10
        mask = create_causal_mask(seq_len)
        assert mask.shape == (seq_len, seq_len)

    def test_causal_mask_lower_triangular(self) -> None:
        seq_len = 5
        mask = create_causal_mask(seq_len)

        # Check that diagonal and below are 1
        for i in range(seq_len):
            for j in range(seq_len):
                if i >= j:
                    assert mask[i, j] == 1
                else:
                    assert mask[i, j] == 0

    def test_causal_mask_device(self) -> None:
        seq_len = 8
        mask_cpu = create_causal_mask(seq_len, device=torch.device("cpu"))
        assert mask_cpu.device.type == "cpu"

        if torch.cuda.is_available():
            mask_cuda = create_causal_mask(seq_len, device=torch.device("cuda"))
            assert mask_cuda.device.type == "cuda"


class TestIntegration:
    """Integration tests for the transformer architecture."""

    def test_full_forward_pass(self) -> None:
        vocab_size = 2000
        dim = 256
        num_heads = 8
        num_layers = 4
        hidden_dim = 1024

        model = TransformerDecoder(
            vocab_size,
            dim,
            num_heads,
            num_layers,
            hidden_dim,
            max_seq_length=512,
            dropout=0.1,
        )
        model.eval()

        batch_size = 8
        seq_len = 64
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, seq_len, vocab_size)
        assert not torch.isnan(output).any()

    def test_model_parameter_count(self) -> None:
        model = TransformerDecoder(
            vocab_size=1000,
            dim=256,
            num_heads=8,
            num_layers=4,
            hidden_dim=1024,
        )

        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0

    def test_model_as_nn_module(self) -> None:
        model = TransformerDecoder(
            vocab_size=512,
            dim=128,
            num_heads=8,
            num_layers=2,
            hidden_dim=512,
        )

        assert isinstance(model, nn.Module)
        assert hasattr(model, "parameters")
        assert hasattr(model, "forward")

    def test_device_compatibility(self) -> None:
        if torch.cuda.is_available():
            model = TransformerDecoder(
                vocab_size=256,
                dim=64,
                num_heads=4,
                num_layers=1,
                hidden_dim=256,
            )
            model = model.cuda()

            x = torch.randint(0, 256, (2, 16), device="cuda")
            output = model(x)

            assert output.device.type == "cuda"

    def test_model_uses_pytorch_components(self) -> None:
        """Verify model uses PyTorch built-in components."""
        model = TransformerDecoder(
            vocab_size=100,
            dim=64,
            num_heads=4,
            num_layers=2,
            hidden_dim=256,
        )

        # Check that embedding is torch.nn.Embedding
        assert isinstance(model.embedding, nn.Embedding)

        # Check that norm is torch.nn.LayerNorm
        assert isinstance(model.norm, nn.LayerNorm)

        # Check that transformer blocks use PyTorch components
        for layer in model.layers:
            assert isinstance(layer, TransformerBlock)
            assert isinstance(layer.norm1, nn.LayerNorm)
            assert isinstance(layer.norm2, nn.LayerNorm)
            assert isinstance(layer.attention, nn.MultiheadAttention)
            assert isinstance(layer.ffn, nn.Sequential)
