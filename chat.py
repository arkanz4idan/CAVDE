from __future__ import annotations

from pathlib import Path
import sys


from cavde.factory import (
    create_model,
    create_tokenizer_from_vocab,
)

from cavde.generation import GreedyGenerator
from cavde.checkpoint import Checkpoint
from cavde.vocabulary import Vocabulary


def main():

    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 50)
    print("              CAVDE CHAT")
    print("=" * 50)
    print()

    # ==========================
    # Vocabulary
    # ==========================

    print("Loading Vocabulary...")

    vocab = Vocabulary()

    vocab.load(
        Path("checkpoints/vocab.json")
    )

    print(f"Vocabulary Size : {len(vocab)}")
    print("      ✓ Vocabulary Ready")

    print()

    # ==========================
    # Tokenizer
    # ==========================

    print("Creating Tokenizer...")

    tokenizer = create_tokenizer_from_vocab(
        vocab
    )

    print("      ✓ Tokenizer Ready")

    print()

    # ==========================
    # Model
    # ==========================

    print("Creating Model...")

    model = create_model(
        vocab_size=len(vocab)
    )

    print("      ✓ Model Ready")

    print()

    # ==========================
    # Checkpoint
    # ==========================

    checkpoint = Checkpoint()

    state = None

    if checkpoint.has_last():

        print("Loading Checkpoint...")

        try:
            state = checkpoint.load_last(
                model
            )

            print("      ✓ Checkpoint Ready")
        except Exception as e:
            print(f"Warning: Could not load checkpoint ({e}). Using untrained model.")
            state = None

    else:

        print("No checkpoint found.")

    print()

    if state is None:

        print("Model : CAVDE-Base (Untrained)")

    else:

        print("Model : CAVDE-Base (Loaded)")
        print(
            f"Epoch : {state['epoch']}"
        )
        print(
            f"Loss  : {state['loss']:.6f}"
        )

    print()

    # ==========================
    # Generator
    # ==========================

    generator = GreedyGenerator(

        model,

        tokenizer,

        temperature=0.8,

        top_k=5,

        top_p=0.9,

    )

    print("Type 'exit' to quit.")
    print()

    while True:

        prompt = input("You : ").strip()

        if not prompt:
            continue

        if prompt.lower() in (
            "exit",
            "quit",
        ):
            break

        answer = generator.generate(
            prompt,
        )

        print()
        print("CAVDE :", answer)
        print()


if __name__ == "__main__":
    raise SystemExit(main())