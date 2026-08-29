from __future__ import annotations

import time

from cavde.factory import (
    create_model,
    create_tokenizer,
)

from cavde.generation import GreedyGenerator
from cavde.checkpoint import load_checkpoint


def benchmark():

    tokenizer = create_tokenizer()

    model = create_model(
        vocab_size=max(
            len(tokenizer),
            4,
        )
    )

    checkpoint = load_checkpoint(model)

    generator = GreedyGenerator(
        model,
        tokenizer,
    )

    prompt = "halo"

    start = time.perf_counter()

    response = generator.generate(
        prompt,
    )

    elapsed = (
        time.perf_counter()
        - start
    ) * 1000

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print()

    print("=" * 50)
    print("           CAVDE BENCHMARK")
    print("=" * 50)

    print()

    print("Prompt       :", prompt)
    print("Response     :", response)

    print()

    print(
        f"Inference    : {elapsed:.2f} ms"
    )

    print(
        f"Vocabulary   : {len(tokenizer)}"
    )

    print(
        f"Parameters   : {parameters:,}"
    )

    if checkpoint:

        print(
            f"Epoch         : {checkpoint['epoch']}"
        )

        print(
            f"Loss          : {checkpoint['loss']:.6f}"
        )

    print()