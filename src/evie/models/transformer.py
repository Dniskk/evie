"""Transformer architecture implementation for evie LLM.

This module provides a foundational transformer architecture with multi-head
self-attention, feedforward networks, and positional encoding for language modeling.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    """Embedding layer for token and position embeddings.

    Applies sqrt(dim) scaling factor as in "Attention is All You Need".

    Args:
        vocab_size: Size of the vocabulary.
        dim: Embedding dimension.
    """

    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.dim = dim
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embedding weights with normal distribution."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_length).

        Returns:
            Embedded tensor of shape (batch_size, seq_length, dim).
        """
        return self.embedding(x) * math.sqrt(self.dim)


class PositionalEncoding(nn.Module):
    """Positional encoding using sinusoidal functions.

    Generates positional encodings using sine and cosine functions as described
    in "Attention is All You Need".

    Args:
        dim: Embedding dimension.
        max_seq_length: Maximum sequence length. Defaults to 2048.
        dropout: Dropout probability. Defaults to 0.1.
    """

    def __init__(
        self,
        dim: int,
        max_seq_length: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

        pe = torch.zeros(max_seq_length, dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float)
            * -(math.log(10000.0) / dim)
        )

        # Apply sine to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices (1, 3, 5, ...)
        # Handle odd dimensions: if dim is odd, div_term has one extra element for sine
        if dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to embeddings.

        Args:
            x: Input tensor of shape (batch_size, seq_length, dim).

        Returns:
            Tensor with positional encoding added, shape (batch_size, seq_length, dim).
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class LayerNorm(nn.Module):
    """Layer normalization with learnable affine parameters.

    Args:
        dim: Dimension of the input.
        eps: Small value to prevent division by zero. Defaults to 1e-6.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input.

        Args:
            x: Input tensor of any shape.

        Returns:
            Normalized tensor of the same shape.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return normalized * self.weight + self.bias


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism.

    Implements scaled dot-product attention with multiple heads to attend to
    different subspaces.

    Args:
        dim: Model dimension.
        num_heads: Number of attention heads.
        dropout: Attention dropout probability. Defaults to 0.1.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize attention weights with Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, seq_length, dim).
            mask: Optional attention mask. Can be:
                  - (seq_length, seq_length): Same mask for all batch items and heads
                  - (batch_size, seq_length, seq_length): Per-batch mask
                  - (batch_size, num_heads, seq_length, seq_length): Full mask
                  Positions with value 0 are masked out (set to -inf before softmax).
                  Defaults to None.

        Returns:
            Output tensor of shape (batch_size, seq_length, dim).
        """
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x).reshape(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # Handle different mask shapes through broadcasting
            # mask shape can be (S, S), (B, S, S), or (B, H, S, S)
            # scores shape is (B, H, S, S)
            if mask.dim() == 2:
                # (S, S) -> (1, 1, S, S) for broadcasting
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                # (B, S, S) -> (B, 1, S, S) for broadcasting
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size,
            seq_len,
            self.dim,
        )

        output = self.proj(attn_output)
        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network.

    Two linear transformations with ReLU activation in between.

    Args:
        dim: Input/output dimension.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout probability. Defaults to 0.1.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(p=dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize feedforward weights with Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward network.

        Args:
            x: Input tensor of any shape with last dimension = dim.

        Returns:
            Output tensor of the same shape.
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward layers.

    Combines multi-head attention, layer normalization, and feed-forward networks
    with residual connections.

    Args:
        dim: Model dimension.
        num_heads: Number of attention heads.
        hidden_dim: Hidden dimension for feed-forward network.
        dropout: Dropout probability. Defaults to 0.1.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attention = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_length, dim).
            mask: Optional attention mask. Defaults to None.

        Returns:
            Output tensor of shape (batch_size, seq_length, dim).
        """
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerDecoder(nn.Module):
    """Transformer decoder stack.

    Stacks multiple transformer blocks to form a complete transformer decoder.
    Uses pre-normalization (LayerNorm before attention and FFN) for training stability.

    Args:
        vocab_size: Size of the vocabulary.
        dim: Model dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer blocks.
        hidden_dim: Hidden dimension for feed-forward networks.
        max_seq_length: Maximum sequence length. Defaults to 2048.
        dropout: Dropout probability. Defaults to 0.1.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_heads: int,
        num_layers: int,
        hidden_dim: int,
        max_seq_length: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            msg = f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            raise ValueError(msg)
        if num_layers < 1:
            msg = f"num_layers must be at least 1, got {num_layers}"
            raise ValueError(msg)

        self.embedding = Embedding(vocab_size, dim)
        self.pos_encoding = PositionalEncoding(dim, max_seq_length, dropout)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(dim, num_heads, hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = LayerNorm(dim)
        self.output_proj = nn.Linear(dim, vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize output projection weights."""
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through transformer decoder.

        Args:
            x: Input token indices of shape (batch_size, seq_length).
            mask: Optional attention mask. See MultiHeadAttention for supported shapes.
                  For causal (autoregressive) decoding, use create_causal_mask().
                  Defaults to None.

        Returns:
            Logits of shape (batch_size, seq_length, vocab_size).
        """
        x = self.embedding(x)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        x = self.output_proj(x)
        return x


def create_causal_mask(seq_length: int, device: torch.device | None = None) -> torch.Tensor:
    """Create a causal (lower triangular) attention mask for autoregressive decoding.

    Args:
        seq_length: Length of the sequence.
        device: Device to create the mask on. Defaults to CPU.

    Returns:
        Boolean mask of shape (seq_length, seq_length) where mask[i, j] = 1 if i >= j,
        else 0. This ensures each position can only attend to itself and previous positions.
    """
    mask = torch.tril(torch.ones(seq_length, seq_length, device=device))
    return mask
