from __future__ import annotations

from cavde.tokenizers.base import BaseTokenizer
from cavde.tokenizers.normalizer import TextNormalizer


class SimpleTokenizer(BaseTokenizer):

    def __init__(self, vocabulary):

        self.normalizer = TextNormalizer()

        self.vocab = vocabulary
    
    def __len__(self):

        return len(self.vocab)

    def tokenize(self, text: str):

        text = self.normalizer.normalize(text)

        return text.split()

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ):

        tokens = self.tokenize(text)

        if add_special_tokens:
            tokens = (
                ["<BOS>"]
                + tokens
                + ["<EOS>"]
            )

        unk = self.vocab.token_to_id["<UNK>"]

        return [
            self.vocab.token_to_id.get(
                token,
                unk,
            )
            for token in tokens
        ]

    def decode(self, ids):

        words = []

        for token_id in ids:

            token = self.vocab.id_to_token.get(
                token_id,
                "<UNK>",
            )

            if token in {
                "<PAD>",
                "<BOS>",
                "<EOS>",
                "<SEP>",
            }:
                continue

            words.append(token)

        return " ".join(words)