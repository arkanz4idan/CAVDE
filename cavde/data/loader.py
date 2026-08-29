from __future__ import annotations

import torch

from torch.utils.data import DataLoader


def collate_fn(batch):

    xs = []

    ys = []

    max_x = max(

        len(x)

        for x, _ in batch

    )

    max_y = max(

        len(y)

        for _, y in batch

    )

    for x, y in batch:

        px = torch.full(

            (max_x,),

            0,

            dtype=torch.long,

        )

        py = torch.full(

            (max_y,),

            0,

            dtype=torch.long,

        )

        px[: len(x)] = x

        py[: len(y)] = y

        xs.append(px)

        ys.append(py)

    return (

        torch.stack(xs),

        torch.stack(ys),

    )


def create_dataloader(

    dataset,

    batch_size=32,

    shuffle=True,

):

    return DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=shuffle,

        collate_fn=collate_fn,

        pin_memory=True,

    )