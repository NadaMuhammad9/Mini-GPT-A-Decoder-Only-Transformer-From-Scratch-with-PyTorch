"""
Transformer Block Module

Implements a Decoder-Only Transformer Block.

Each block consists of:
- Layer Normalization.
- Causal Multi-Head Self-Attention.
- Residual Connections.
- Feed-Forward Neural Network.
- GELU Activation Function.

Uses a Pre-Layer Normalization architecture.
"""

import torch.nn as nn
from .attention import CausalSelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)

        self.attention = CausalSelfAttention(
            embed_dim, num_heads, dropout
        )

        self.ln2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x
