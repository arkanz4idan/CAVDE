from __future__ import annotations

import torch


def create_scheduler(
    optimizer,
    scheduler_type: str = "none",
    **kwargs,
):
    """
    Create learning rate scheduler.
    """

    scheduler_type = scheduler_type.lower()

    if scheduler_type == "none":
        return None

    if scheduler_type == "step":

        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get(
                "step_size",
                5,
            ),
            gamma=kwargs.get(
                "gamma",
                0.5,
            ),
        )

    if scheduler_type == "cosine":

        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get(
                "t_max",
                10,
            ),
        )

    raise ValueError(
        f"Unknown scheduler: {scheduler_type}"
    )