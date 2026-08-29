from cavde.data.reader import DatasetReader
from cavde.data.conversation import ConversationDataset
from cavde.data.dataloader import create_dataloader
from cavde.factory import create_tokenizer


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

batch = next(iter(loader))

print(batch)

print()

print(batch["input_ids"].shape)

print(batch["labels"].shape)

print()
print("Decoded Input :")
print(
    tokenizer.decode(
        batch["input_ids"][0].tolist()
    )
)

print()

print("Decoded Label :")
print(
    tokenizer.decode(
        batch["labels"][0].tolist()
    )
)

print()

print("Raw Input IDs :")
print(batch["input_ids"][0].tolist())

print()

print("Raw Label IDs :")
print(batch["labels"][0].tolist())