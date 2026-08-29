from __future__ import annotations

import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import FeedForward


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model,
        heads,
        ff,
        dropout,
    ):

        super().__init__()

        self.attention = MultiHeadAttention(

            d_model,

            heads,

            dropout,

        )

        self.ff = FeedForward(

            d_model,

            ff,

            dropout,

        )

        self.norm1 = nn.LayerNorm(

            d_model

        )

        self.norm2 = nn.LayerNorm(

            d_model

        )

        self.dropout = nn.Dropout(

            dropout

        )

    def forward(

        self,

        x,

        mask=None,

    ):

        x = x + self.dropout(

            self.attention(

                self.norm1(x),

                mask,

            )

        )

        x = x + self.dropout(

            self.ff(

                self.norm2(x)

            )

        )

        return x