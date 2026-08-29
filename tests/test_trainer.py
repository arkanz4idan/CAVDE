from cavde.data.reader import DatasetReader
from cavde.data.conversation import ConversationDataset
from cavde.data.dataloader import create_dataloader
from cavde.validation import Validator
from cavde.checkpoint import CheckpointManager

from cavde.factory import (
    create_loss_function,
    create_model,
    create_optimizer,
    create_scheduler_instance,
    create_tokenizer,
)

from cavde.engine import TrainingEngine
from cavde.trainer import Trainer


reader = DatasetReader("datasets/chat.txt")

samples = list(reader)

tokenizer = create_tokenizer()

dataset = ConversationDataset(
    samples,
    tokenizer,
)

loader = create_dataloader(
    dataset,
    batch_size=2,
)

model = create_model(
    vocab_size=max(len(tokenizer), 4)
)

optimizer = create_optimizer(model)

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

checkpoint = CheckpointManager()

trainer = Trainer(
    engine,
    loader,
    scheduler=scheduler,
    validator=validator,
    checkpoint=checkpoint,
    epochs=3,
)

trainer.train()