from __future__ import annotations

from tqdm import tqdm

from cavde.config import CONFIG


class Trainer:

    def __init__(
        self,
        engine,
        dataloader,
        scheduler=None,
        validator=None,
        checkpoint=None,
        epochs=CONFIG.epochs,
    ):

        self.engine = engine
        self.dataloader = dataloader
        self.scheduler = scheduler
        self.validator = validator
        self.checkpoint = checkpoint
        self.epochs = epochs

    def train(
        self,
        start_epoch: int = 0,
    ):

        best_loss = float("inf")

        for epoch in range(
            start_epoch,
            self.epochs,
        ):

            total_loss = 0.0

            print()

            print("=" * 50)

            print(
                f"Epoch {epoch + 1}/{self.epochs}"
            )

            print("=" * 50)

            progress = tqdm(
                self.dataloader,
                leave=False,
            )

            for batch in progress:

                loss = self.engine.train_batch(
                    batch["input_ids"],
                    batch["labels"],
                )

                total_loss += loss

                progress.set_postfix(
                    loss=f"{loss:.4f}"
                )

            average = total_loss / len(
                self.dataloader
            )

            print(
                f"Train Loss    : {average:.6f}"
            )

            valid_loss = None

            if self.validator is not None:

                valid_loss = self.validator.validate(
                    self.dataloader
                )

                print(
                    f"Valid Loss    : {valid_loss:.6f}"
                )

            if self.scheduler is not None:

                try:

                    self.scheduler.step(
                        valid_loss
                    )

                except TypeError:

                    self.scheduler.step()

            current_lr = (
                self.engine.optimizer.param_groups[0]["lr"]
            )

            print(
                f"Learning Rate : {current_lr:.8f}"
            )

            if self.checkpoint is not None:

                loss_value = (
                    valid_loss
                    if valid_loss is not None
                    else average
                )

                self.checkpoint.save_last(

                    model=self.engine.model,

                    optimizer=self.engine.optimizer,

                    scheduler=self.scheduler,

                    epoch=epoch + 1,

                    loss=loss_value,

                )

                if loss_value < best_loss:

                    best_loss = loss_value

                    self.checkpoint.save_best(

                        model=self.engine.model,

                        optimizer=self.engine.optimizer,

                        scheduler=self.scheduler,

                        epoch=epoch + 1,

                        loss=loss_value,

                    )