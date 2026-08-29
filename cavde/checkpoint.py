from __future__ import annotations

from pathlib import Path

import torch


class Checkpoint:

    def __init__(
        self,
        directory: str | Path = "checkpoints",
    ):

        self.directory = Path(directory)

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(

        self,

        model,

        optimizer,

        scheduler,

        epoch: int,

        loss: float,

        filename: str,

    ):

        path = self.directory / filename

        torch.save(

            {

                "epoch": epoch,

                "loss": loss,

                "model": model.state_dict(),

                "optimizer": optimizer.state_dict(),

                "scheduler": (
                    scheduler.state_dict()
                    if scheduler is not None
                    else None
                ),

            },

            path,

        )

        print(
            f"Checkpoint saved -> {path.name}"
        )

    def save_best(

        self,

        model,

        optimizer,

        scheduler,

        epoch: int,

        loss: float,

    ):

        self.save(

            model=model,

            optimizer=optimizer,

            scheduler=scheduler,

            epoch=epoch,

            loss=loss,

            filename="best.pt",

        )

    def save_last(

        self,

        model,

        optimizer,

        scheduler,

        epoch: int,

        loss: float,

    ):

        self.save(

            model=model,

            optimizer=optimizer,

            scheduler=scheduler,

            epoch=epoch,

            loss=loss,

            filename="last.pt",

        )

    def exists(

        self,

        filename: str,

    ) -> bool:

        return (
            self.directory / filename
        ).exists()

    def load(
        self,
        filename: str,
        model,
        optimizer=None,
        scheduler=None,
    ):

        path = self.directory / filename

        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

        model.load_state_dict(
            checkpoint["model"]
        )

        if (
            optimizer is not None
            and checkpoint["optimizer"] is not None
        ):
            optimizer.load_state_dict(
                checkpoint["optimizer"]
            )

        if (
            scheduler is not None
            and checkpoint["scheduler"] is not None
        ):
            scheduler.load_state_dict(
                checkpoint["scheduler"]
            )

        return checkpoint
    def load_last(

        self,

        model,

        optimizer=None,

        scheduler=None,

    ):

        return self.load(

            filename="last.pt",

            model=model,

            optimizer=optimizer,

            scheduler=scheduler,

        )
    
    def has_last(self):

        return self.exists(
            "last.pt"
        )
    