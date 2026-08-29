from cavde.checkpoint import CheckpointManager

from cavde.factory import (
    create_model,
    create_optimizer,
    create_scheduler_instance,
    create_tokenizer,
)

tokenizer = create_tokenizer()

model = create_model(
    vocab_size=max(len(tokenizer), 4)
)

optimizer = create_optimizer(model)

scheduler = create_scheduler_instance(
    optimizer
)

manager = CheckpointManager()

path = manager.save(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=3,
    loss=2.52,
)

print("Saved:", path)

checkpoint = manager.load(
    model,
    optimizer,
    scheduler,
)

print()

print("Epoch :", checkpoint["epoch"])

print("Loss  :", checkpoint["loss"])