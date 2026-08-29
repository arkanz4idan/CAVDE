from __future__ import annotations

import torch


class TrainingEngine:

    """
    Core training engine.
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device: str = "cpu",
    ):

        self.model = model

        self.optimizer = optimizer

        self.criterion = criterion

        self.device = torch.device(device)

        self.model.to(self.device)

    def train_batch(
        self,
        input_ids,
        labels,
    ):

        self.model.train()

        input_ids = input_ids.to(self.device)

        labels = labels.to(self.device)

        self.optimizer.zero_grad()

        logits = self.model(input_ids)

        loss = self.criterion(

            logits.view(-1, logits.size(-1)),

            labels.view(-1),

        )

        loss.backward()

        self.optimizer.step()

        return loss.item()