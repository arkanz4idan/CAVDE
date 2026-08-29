from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class BaseTokenizer(ABC):

    @abstractmethod
    def tokenize(self, text: str):
        ...

    @abstractmethod
    def encode(self, text: str):
        ...

    @abstractmethod
    def decode(self, ids):
        ...