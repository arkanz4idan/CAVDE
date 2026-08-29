from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from cavde.tokenizers.normalizer import TextNormalizer


SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
    "<SEP>",
]


class Vocabulary:

    def __init__(self):

        self.counter = Counter()

        self.token_to_id = {}

        self.id_to_token = {}

        self.normalizer = TextNormalizer()

    def add_text(self, text: str):

        text = self.normalizer.normalize(text)

        self.counter.update(
            text.split()
        )

    def build(self):

        self.token_to_id.clear()

        self.id_to_token.clear()

        index = 0

        for token in SPECIAL_TOKENS:

            self.token_to_id[token] = index

            self.id_to_token[index] = token

            index += 1

        for token, _ in self.counter.most_common():

            if token in self.token_to_id:
                continue

            self.token_to_id[token] = index

            self.id_to_token[index] = token

            index += 1

    def encode(self, text: str):

        text = self.normalizer.normalize(text)

        unk = self.token_to_id["<UNK>"]

        return [

            self.token_to_id.get(
                token,
                unk,
            )

            for token in text.split()

        ]

    def decode(self, ids):

        return " ".join(

            self.id_to_token.get(
                idx,
                "<UNK>",
            )

            for idx in ids

        )

    def save(self, path: Path):

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.token_to_id,
                file,
                ensure_ascii=False,
                indent=2,
            )

    def load(self, path: Path):

        print("Loading from:", path.resolve())
        print("Exists:", path.exists())

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        print("Type:", type(data))
        print("Length:", len(data))

        self.token_to_id = data

        self.id_to_token = {
            value: key
            for key, value in self.token_to_id.items()
        }

        print("Loaded:", len(self.token_to_id))


    @property
    def vocab_size(self):

        return len(
            self.token_to_id
        )

    def __len__(self):

        return self.vocab_size