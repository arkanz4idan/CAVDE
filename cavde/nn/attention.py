from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Manual Multi-Head Self Attention.

    Future Ready:
    - RoPE
    - FlashAttention
    - KV Cache
    - GQA
    """

    def __init__(
        self,
        d_model: int,
        heads: int,
        dropout: float,
    ):

        super().__init__()

        if d_model % heads != 0:
            raise ValueError(
                "d_model harus habis dibagi jumlah head."
            )

        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):

        batch, seq, dim = x.shape

        x = x.view(

            batch,

            seq,

            self.heads,

            self.head_dim,

        )

        return x.transpose(1, 2)

    def merge_heads(self, x):

        batch, heads, seq, dim = x.shape

        x = x.transpose(1, 2)

        return x.reshape(

            batch,

            seq,

            heads * dim,

        )

    def forward(

        self,

        x,

        mask=None,

    ):

        q = self.split_heads(

            self.q_proj(x)

        )

        k = self.split_heads(

            self.k_proj(x)

        )

        v = self.split_heads(

            self.v_proj(x)

        )

        scores = (

            q @ k.transpose(-2, -1)

        ) / math.sqrt(self.head_dim)

        if mask is not None:

            scores = scores.masked_fill(

                mask == 0,

                float("-inf"),

            )

        attention = F.softmax(

            scores,

            dim=-1,

        )

        attention = self.dropout(

            attention

        )

        out = attention @ v

        out = self.merge_heads(out)

        out = self.out_proj(out)

        return out