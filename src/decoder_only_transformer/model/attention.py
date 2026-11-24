"""
Attention Module

Implements Causal Multi-Head Self-Attention from scratch using PyTorch.

The causal attention mask prevents each token from attending to
future tokens, ensuring autoregressive language modeling.

This allows the model to predict the next token based only on
previous tokens and the current token.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(
            B, T, self.num_heads, self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B, T, self.num_heads, self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B, T, self.num_heads, self.head_dim
        ).transpose(1, 2)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        mask = torch.tril(
            torch.ones(T, T, device=x.device, dtype=torch.bool)
        )

        scores = scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        output = weights @ v

        output = output.transpose(1, 2).contiguous()
        output = output.view(B, T, C)

        return self.proj(output)
