from cavde.factory import (
    create_model,
    create_tokenizer,
)

from cavde.generation import GreedyGenerator


tokenizer = create_tokenizer()

model = create_model(
    vocab_size=max(
        len(tokenizer),
        4,
    )
)

generator = GreedyGenerator(
    model,
    tokenizer,
    temperature=1.0,
)

print()

print("=" * 40)
print(" Greedy Search")
print("=" * 40)

tests = [

    (1.0, None, None),

    (1.0, 5, None),

    (1.0, None, 0.90),

    (1.0, 5, 0.90),

]

for temperature, top_k, top_p in tests:

    generator.temperature = temperature
    generator.top_k = top_k
    generator.top_p = top_p

    print()

    print(
        f"T={temperature}  K={top_k}  P={top_p}"
    )

    print(
        generator.generate("halo")
    )