from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    Base class seluruh model CAVDE.
    """

    def __init__(self):

        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        ...