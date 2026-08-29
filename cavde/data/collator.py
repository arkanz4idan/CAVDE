from __future__ import annotations

import torch


class ConversationCollator:

    def __init__(self, pad_id=0):
        self.pad_id = pad_id

    def _pad(self, sequences):

        max_length = max(len(seq) for seq in sequences)

        padded = []

        for seq in sequences:

            seq = list(seq)

            seq += [self.pad_id] * (
                max_length - len(seq)
            )

            padded.append(seq)

        return padded

    def __call__(self, batch):

        input_ids = self._pad(
            [x["input_ids"] for x in batch]
        )

        labels = self._pad(
            [x["labels"] for x in batch]
        )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
        }