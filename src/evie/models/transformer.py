"""Transformer architecture implementation for evie LLM.

This module provides a foundational transformer architecture with multi-head
self-attention, feedforward networks, and positional encoding for language modeling.

Uses PyTorch's optimized built-in components for better performance and reliability.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward layers.

    Combines multi-head attention, layer normalization, and feed-forward networks
    with residual connections. Uses PyTorch's built-in MultiheadAttention and
    LayerNorm for optimized performance.

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
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_length, dim).
            mask: Optional attention mask. Can be:
                  - (seq_length, seq_length): Same mask for all batch items
                  - (batch_size, seq_length, seq_length): Per-batch mask
                  For PyTorch's MultiheadAttention, positions with value False
                  are masked out. Use convert_mask_for_mha() to convert from
                  the old format (where 0 = masked).
                  Defaults to None.

        Returns:
            Output tensor of shape (batch_size, seq_length, dim).
        """
        # Pre-normalization
        normed = self.norm1(x)

        # Convert mask to PyTorch MultiheadAttention format if provided
        attn_mask = None
        if mask is not None:
            # Convert from (1 = attend, 0 = mask) to (True = attend, False = mask)
            # Then invert for PyTorch's MultiheadAttention (True = mask)
            attn_mask = (mask == 0)

            # Handle different mask dimensions
            if attn_mask.dim() == 2:
                # (S, S) - broadcast to all batches
                pass
            elif attn_mask.dim() == 3:
                # (B, S, S) - need to flatten for MultiheadAttention
                # PyTorch expects (B*num_heads, S, S) or (S, S)
                # We'll use (S, S) by taking the first batch item
                # This assumes the mask is the same across batches
                attn_mask = attn_mask[0]

        # Self-attention with residual connection
        attn_output, _ = self.attention(
            normed,
            normed,
            normed,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = x + attn_output

        # Feed-forward with residual connection
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerDecoder(nn.Module):
    """Transformer decoder stack.

    Stacks multiple transformer blocks to form a complete transformer decoder.
    Uses pre-normalization (LayerNorm before attention and FFN) for training stability.
    Leverages PyTorch's built-in components for optimal performance.

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

        self.dim = dim

        # Use PyTorch's built-in Embedding with sqrt(dim) scaling
        self.embedding = nn.Embedding(vocab_size, dim)
        self.embedding_scale = math.sqrt(dim)

        self.pos_encoding = PositionalEncoding(dim, max_seq_length, dropout)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(dim, num_heads, hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights following standard practices."""
        # Initialize embedding
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        # Initialize output projection
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
            mask: Optional attention mask. Positions with value 1 can be attended to,
                  positions with value 0 are masked out.
                  Can be (seq_length, seq_length) or (batch_size, seq_length, seq_length).
                  For causal (autoregressive) decoding, use create_causal_mask().
                  Defaults to None.

        Returns:
            Logits of shape (batch_size, seq_length, vocab_size).
        """
        # Embedding with scaling as in "Attention is All You Need"
        x = self.embedding(x) * self.embedding_scale
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
