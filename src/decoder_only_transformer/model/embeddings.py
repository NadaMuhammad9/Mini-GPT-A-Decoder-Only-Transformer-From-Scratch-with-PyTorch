"""
Embeddings Module

Implements token embeddings and positional embeddings.

Token embeddings represent the semantic identity of each token,
while positional embeddings provide information about token positions
within the input sequence.
"""

import torch
import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size, embed_dim
        )

        self.position_embedding = nn.Embedding(
            block_size, embed_dim
        )

    def forward(self, x):
        B, T = x.shape

        positions = torch.arange(
            T, device=x.device
        )

        token_emb = self.token_embedding(x)
        position_emb = self.position_embedding(positions)

        return token_emb + position_emb
