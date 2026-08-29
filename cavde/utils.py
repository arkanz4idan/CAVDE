from pathlib import Path
import random

import numpy as np
import torch


def seed_everything(seed: int):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


def ensure_directory(path: Path):

    path.mkdir(
        parents=True,
        exist_ok=True,
    )


def count_parameters(model):

    return sum(

        parameter.numel()

        for parameter in model.parameters()

        if parameter.requires_grad

    )


def readable_number(number: int):

    if number >= 1_000_000:

        return f"{number/1_000_000:.2f}M"

    if number >= 1_000:

        return f"{number/1000:.2f}K"

    return str(number)