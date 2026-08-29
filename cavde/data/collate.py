from __future__ import annotations

import torch


class ConversationCollator:

    def __init__(
        self,
        pad_id: int = 0,
    ):

        self.pad_id = pad_id

    def __call__(
        self,
        batch,
    ):

        max_input = max(
            len(item["input_ids"])
            for item in batch
        )

        max_label = max(
            len(item["labels"])
            for item in batch
        )

        input_ids = []
        labels = []
        attention_masks = []

        for item in batch:

            ids = item["input_ids"]

            lbl = item["labels"]

            mask = item["attention_mask"]

            ids = ids + [
                self.pad_id
            ] * (max_input - len(ids))

            lbl = lbl + [
                self.pad_id
            ] * (max_label - len(lbl))

            mask = mask + [
                0
            ] * (max_input - len(mask))

            input_ids.append(ids)

            labels.append(lbl)

            attention_masks.append(mask)

        return {

            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),

            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),

            "attention_mask": torch.tensor(
                attention_masks,
                dtype=torch.long,
            ),
        }