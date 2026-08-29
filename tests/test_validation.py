from cavde.data.conversation import ConversationDataset
from cavde.data.dataloader import create_dataloader
from cavde.data.reader import DatasetReader

from cavde.engine import TrainingEngine
from cavde.factory import (
    create_loss_function,
    create_model,
    create_optimizer,
    create_scheduler_instance,
    create_tokenizer,
)

from cavde.validation import Validator


reader = DatasetReader("datasets/chat.txt")

tokenizer = create_tokenizer()

dataset = ConversationDataset(
    reader,
    tokenizer,
)

loader = create_dataloader(
    dataset,
    batch_size=2,
)

model = create_model(
    vocab_size=max(
        len(tokenizer),
        4,
    )
)

optimizer = create_optimizer(
    model
)

criterion = create_loss_function()

scheduler = create_scheduler_instance(
    optimizer
)

engine = TrainingEngine(
    model,
    optimizer,
    criterion,
)

validator = Validator(
    model,
    criterion,
)

loss = validator.validate(
    loader
)

print()

print("=" * 40)
print(" Validation")
print("=" * 40)

print(
    f"Validation Loss : {loss:.6f}"
)