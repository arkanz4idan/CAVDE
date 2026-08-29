from __future__ import annotations

from torch.utils.data import DataLoader

from .collator import ConversationCollator


def create_dataloader(
    dataset,
    batch_size=4,
    shuffle=True,
):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=ConversationCollator(),
    )