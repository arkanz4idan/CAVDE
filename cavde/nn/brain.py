from __future__ import annotations

import torch.nn as nn

from .embedding import TokenEmbedding
from .layers import TransformerBlock
from .positional import PositionalEncoding


from .base import BaseModel

class CAVDEModel(BaseModel):

    def __init__(

        self,

        vocab_size: int,

        d_model: int,

        layers: int,

        heads: int,

        feed_forward: int,

        dropout: float,

        max_length: int,

    ):

        super().__init__()

        self.embedding = TokenEmbedding(

            vocab_size,

            d_model,

        )

        self.position = PositionalEncoding(

            d_model,

            max_length,

        )

        self.blocks = nn.ModuleList(

            [

                TransformerBlock(

                    d_model,

                    heads,

                    feed_forward,

                    dropout,

                )

                for _ in range(layers)

            ]

        )

        self.norm = nn.LayerNorm(

            d_model

        )

        self.output = nn.Linear(

            d_model,

            vocab_size,

        )

    def forward(

        self,

        x,

        mask=None,

    ):

        x = self.embedding(x)

        x = self.position(x)

        for block in self.blocks:

            x = block(

                x,

                mask,

            )

        x = self.norm(x)

        return self.output(x)