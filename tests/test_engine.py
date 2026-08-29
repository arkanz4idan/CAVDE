import torch

from cavde.data.reader import DatasetReader
from cavde.data.conversation import ConversationDataset
from cavde.data.dataloader import create_dataloader

from cavde.factory import (
    create_model,
    create_optimizer,
    create_loss_function,
    create_tokenizer,
)

from cavde.engine import TrainingEngine


# Read dataset
reader = DatasetReader("datasets/chat.txt")
samples = list(reader)

# Tokenizer
tokenizer = create_tokenizer()

# Dataset
dataset = ConversationDataset(
    samples,
    tokenizer,
)

# DataLoader
loader = create_dataloader(
    dataset,
    batch_size=2,
)

# Model
model = create_model(
    vocab_size=max(len(tokenizer), 4)
)

# Optimizer
optimizer = create_optimizer(model)

# Loss
criterion = create_loss_function()

# Engine
engine = TrainingEngine(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
)

# First batch
batch = next(iter(loader))

loss = engine.train_batch(
    batch["input_ids"],
    batch["labels"],
)

print()

print("=" * 40)
print(" First Training Step")
print("=" * 40)

print(f"Loss : {loss:.6f}")