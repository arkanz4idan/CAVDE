from __future__ import annotations

import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        dropout: float,
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                d_model,
                hidden_size,
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_size,
                d_model,
            ),

            nn.Dropout(dropout),

        )

    def forward(self, x):

        return self.net(x)