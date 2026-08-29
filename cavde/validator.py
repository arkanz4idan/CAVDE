from __future__ import annotations

import torch


class ValidationEngine:

    def __init__(
        self,
        model,
        criterion,
        device="cpu",
    ):

        self.model = model
        self.criterion = criterion
        self.device = torch.device(device)

    @torch.no_grad()
    def validate(
        self,
        dataloader,
    ):

        self.model.eval()

        total_loss = 0.0

        for batch in dataloader:

            input_ids = batch["input_ids"].to(
                self.device
            )

            labels = batch["labels"].to(
                self.device
            )

            logits = self.model(input_ids)

            loss = self.criterion(

                logits.reshape(
                    -1,
                    logits.size(-1),
                ),

                labels.reshape(-1),

            )

            total_loss += loss.item()

        self.model.train()

        return total_loss / len(dataloader)