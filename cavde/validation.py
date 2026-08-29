from __future__ import annotations

import torch


class Validator:

    def __init__(
        self,
        model,
        criterion,
    ):
        self.model = model
        self.criterion = criterion

    @torch.no_grad()
    def validate(
        self,
        dataloader,
    ):

        self.model.eval()

        total_loss = 0.0
        batches = 0

        for batch in dataloader:

            logits = self.model(
                batch["input_ids"]
            )

            loss = self.criterion(
                logits.view(
                    -1,
                    logits.size(-1),
                ),
                batch["labels"].view(-1),
            )

            total_loss += loss.item()
            batches += 1

        self.model.train()

        return total_loss / max(
            batches,
            1,
        )