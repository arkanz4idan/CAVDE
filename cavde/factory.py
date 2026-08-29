from __future__ import annotations

from pathlib import Path

import torch

from cavde.config import CONFIG
from cavde.losses import create_loss
from cavde.nn.brain import CAVDEModel
from cavde.tokenizers.simple import SimpleTokenizer
from cavde.vocabulary import Vocabulary
from cavde.scheduler import create_scheduler
from torch.utils.data import DataLoader
from cavde.data.collate import ConversationCollator


def create_tokenizer_from_vocab(
    vocab: Vocabulary,
) -> SimpleTokenizer:
    """
    Create tokenizer from an existing vocabulary.
    Used for inference/chat.
    """

    return SimpleTokenizer(vocab)

def create_collator():

    return ConversationCollator(
        pad_id=0,
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

        collate_fn=create_collator(),

    )

def create_tokenizer(dataset_path: str | Path = "datasets/chat.txt") -> SimpleTokenizer:
    """
    Build vocabulary from dataset and create tokenizer.
    """

    vocab = Vocabulary()

    dataset_path = Path(dataset_path)

    if dataset_path.exists():

        with open(dataset_path, "r", encoding="utf-8") as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                if "|" not in line:
                    continue

                question, answer = line.split("|", 1)

                vocab.add_text(question)

                vocab.add_text(answer)

    vocab.build()

    return SimpleTokenizer(vocab)


def create_model(vocab_size: int) -> CAVDEModel:

    return CAVDEModel(
        vocab_size=vocab_size,
        d_model=CONFIG.d_model,
        layers=CONFIG.num_layers,
        heads=CONFIG.num_heads,
        feed_forward=CONFIG.feed_forward,
        dropout=CONFIG.dropout,
        max_length=CONFIG.max_sequence_length,
    )


def create_optimizer(model):

    return torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG.learning_rate,
        weight_decay=CONFIG.weight_decay,
    )


def create_loss_function():

    return create_loss(
        pad_id=0,
    )

def create_scheduler_instance(
    optimizer,
):

    return create_scheduler(
        optimizer,
        scheduler_type=CONFIG.scheduler,
        step_size=CONFIG.step_size,
        gamma=CONFIG.gamma,
        t_max=CONFIG.t_max,
    )
