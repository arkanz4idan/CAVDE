from __future__ import annotations

from cavde.factory import (
    create_model,
    create_tokenizer,
)

from cavde.checkpoint import load_checkpoint

from cavde.validation import Validator

from cavde.data.reader import DatasetReader
from cavde.data.conversation import ConversationDataset
from cavde.data.dataloader import create_dataloader

from cavde.losses import create_loss


def evaluate():

    print()

    print("=" * 50)
    print("          CAVDE EVALUATE")
    print("=" * 50)

    tokenizer = create_tokenizer()

    reader = DatasetReader(
        "datasets/chat.jsonl"
    )

    samples = list(reader)

    dataset = ConversationDataset(
        samples,
        tokenizer,
    )

    loader = create_dataloader(
        dataset,
        batch_size=2,
    )

    model = create_model(
        vocab_size=len(tokenizer),
    )

    checkpoint = load_checkpoint(model)

    if checkpoint is None:

        print()

        print("No trained model found.")

        return

    criterion = create_loss(
        pad_id=0,
    )

    validator = Validator(
        model,
        criterion,
    )

    loss = validator.validate(
        loader,
    )

    print()

    print(
        f"Epoch : {checkpoint['epoch']}"
    )

    print(
        f"Loss  : {loss:.6f}"
    )