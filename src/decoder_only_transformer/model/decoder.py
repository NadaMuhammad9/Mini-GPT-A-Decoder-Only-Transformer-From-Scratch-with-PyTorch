"""
Decoder Module

Implements the complete Decoder-Only Transformer architecture.

The model combines:
- Token and positional embeddings.
- Multiple Transformer Blocks.
- Final Layer Normalization.
- Language Modeling Head.

The model produces logits over the vocabulary for next-token prediction
and supports autoregressive text generation.
""" 

import torch
import torch.nn as nn

from .embeddings import Embeddings
from .block import TransformerBlock

class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        block_size=8,
        dropout=0.1
    ):
        super().__init__()

        self.block_size = block_size

        self.embeddings = Embeddings(
            vocab_size, embed_dim, block_size
        )

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.ln = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.blocks(x)
        x = self.ln(x)

        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens=10,
        temperature=1.0,
        top_k=None
    ):
        if temperature <= 0:
            raise ValueError("Temperature must be positive")

        was_training = self.training
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]

            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, k)
                threshold = values[:, [-1]]

                logits = logits.masked_fill(
                    logits < threshold, float("-inf")
                )

            probs = torch.softmax(logits, dim=-1)

            next_token = torch.multinomial(
                probs, num_samples=1
            )

            idx = torch.cat((idx, next_token), dim=1)

        if was_training:
            self.train()

        return idx
