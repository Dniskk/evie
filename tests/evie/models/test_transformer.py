"""Tests for transformer architecture."""

import torch
import torch.nn as nn

from evie.models.transformer import (
    Embedding,
    FeedForward,
    LayerNorm,
    MultiHeadAttention,
    PositionalEncoding,
    TransformerBlock,
    TransformerDecoder,
)


class TestEmbedding:
    """Tests for Embedding layer."""

    def test_embedding_output_shape(self) -> None:
        vocab_size = 1000
        dim = 128
        batch_size = 4
        seq_len = 32

        embedding = Embedding(vocab_size, dim)
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        output = embedding(x)

        assert output.shape == (batch_size, seq_len, dim)

    def test_embedding_scaling(self) -> None:
        vocab_size = 100
        dim = 64
        embedding = Embedding(vocab_size, dim)

        x = torch.zeros((1, 1), dtype=torch.long)
        output = embedding(x)

        expected_scale = (dim**0.5)
        actual_scale = output.abs().max().item()
        assert actual_scale > 0


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


class TestLayerNorm:
    """Tests for LayerNorm."""

    def test_layer_norm_shape(self) -> None:
        dim = 128
        layer_norm = LayerNorm(dim)

        x = torch.randn(4, 32, dim)
        output = layer_norm(x)

        assert output.shape == x.shape

    def test_layer_norm_mean_variance(self) -> None:
        dim = 64
        layer_norm = LayerNorm(dim)
        layer_norm.weight.data.fill_(1.0)
        layer_norm.bias.data.fill_(0.0)

        x = torch.randn(2, 16, dim)
        output = layer_norm(x)

        mean = output.mean(dim=-1)
        var = output.var(dim=-1, unbiased=False)

        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-5)


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention."""

    def test_attention_output_shape(self) -> None:
        dim = 128
        num_heads = 8
        attention = MultiHeadAttention(dim, num_heads)

        batch_size = 4
        seq_len = 32
        x = torch.randn(batch_size, seq_len, dim)
        output = attention(x)

        assert output.shape == (batch_size, seq_len, dim)

    def test_attention_with_mask(self) -> None:
        dim = 64
        num_heads = 4
        attention = MultiHeadAttention(dim, num_heads)

        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, dim)

        mask = torch.ones(batch_size, seq_len, seq_len)
        output = attention(x, mask)

        assert output.shape == (batch_size, seq_len, dim)

    def test_attention_dropout_training(self) -> None:
        dim = 128
        num_heads = 8
        attention = MultiHeadAttention(dim, num_heads, dropout=0.5)
        attention.train()

        x = torch.randn(1, 32, dim)
        output1 = attention(x)
        output2 = attention(x)

        assert not torch.allclose(output1, output2)

    def test_attention_dropout_eval(self) -> None:
        dim = 128
        num_heads = 8
        attention = MultiHeadAttention(dim, num_heads, dropout=0.5)
        attention.eval()

        x = torch.randn(1, 32, dim)
        output1 = attention(x)
        output2 = attention(x)

        assert torch.allclose(output1, output2)

    def test_attention_gradient_flow(self) -> None:
        dim = 64
        num_heads = 4
        attention = MultiHeadAttention(dim, num_heads)
        attention.train()

        x = torch.randn(2, 8, dim, requires_grad=True)
        output = attention(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestFeedForward:
    """Tests for FeedForward."""

    def test_feedforward_output_shape(self) -> None:
        dim = 128
        hidden_dim = 512
        ffn = FeedForward(dim, hidden_dim)

        x = torch.randn(4, 32, dim)
        output = ffn(x)

        assert output.shape == (4, 32, dim)

    def test_feedforward_different_shapes(self) -> None:
        dim = 64
        hidden_dim = 256
        ffn = FeedForward(dim, hidden_dim)

        x = torch.randn(1, 100, dim)
        output = ffn(x)
        assert output.shape == (1, 100, dim)

        x = torch.randn(8, 16, dim)
        output = ffn(x)
        assert output.shape == (8, 16, dim)


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
