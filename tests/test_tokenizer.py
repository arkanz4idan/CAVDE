from pathlib import Path

from cavde.vocabulary import Vocabulary
from cavde.tokenizers.simple import SimpleTokenizer


vocab = Vocabulary()

vocab.add_text("Halo Dunia!")

vocab.add_text("Aku adalah CAVDE.")

vocab.build()

tokenizer = SimpleTokenizer(vocab)

ids = tokenizer.encode(
    "Halo dunia!"
)

print("Encoded:", ids)

print(
    "Decoded:",
    tokenizer.decode(ids),
)

vocab.save(
    Path("checkpoints/test_vocab.json")
)

loaded = Vocabulary()

loaded.load(
    Path("checkpoints/test_vocab.json")
)

print(
    "Loaded Size:",
    loaded.vocab_size,
)