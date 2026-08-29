from __future__ import annotations

from pathlib import Path
import sys


from cavde.data.reader import DatasetReader
from cavde.data.statistics import DatasetStatistics
from cavde.vocabulary import Vocabulary
from cavde.data.dataset import ConversationDataset
from cavde.config import CONFIG
from cavde.engine import TrainingEngine
from cavde.trainer import Trainer
from cavde.validator import ValidationEngine
from cavde.checkpoint import Checkpoint

from cavde.factory import (
    create_loss_function,
    create_model,
    create_optimizer,
    create_scheduler_instance,
    create_tokenizer,
    create_dataloader,
)

SUPPORTED_DATASETS = (
    ".json",
    ".jsonl",
    ".txt",
)

IGNORED_NAMES = (
    "vocab",
    "tokenizer",
    "config",
    "checkpoint",
    "model",
    "pyfolder",
)


def scan_datasets(directory: Path) -> list[Path]:

    if not directory.exists():
        return []

    datasets = []

    for file in sorted(directory.rglob("*")):

        if not file.is_file():
            continue

        if file.suffix.lower() not in SUPPORTED_DATASETS:
            continue

        # Check if any ignored name is in the file path
        if any(word in str(file).lower() for word in IGNORED_NAMES):
            continue

        datasets.append(file)

    return datasets


def main() -> int:

    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 55)
    print("             CAVDE TRAINER")
    print("=" * 55)

    print("[1/5] Loading tokenizer...")

    tokenizer = create_tokenizer()

    dataset_files = scan_datasets(Path("datasets"))

    vocab = Vocabulary()
    stats = DatasetStatistics()

    if dataset_files:

        print("[Dataset] Reading datasets...")
        print()

        total_files = 0

        for dataset in dataset_files:

            print(f"Dataset : {dataset}")

            reader = DatasetReader(dataset)

            for sample in reader:

                stats.update(sample)

                vocab.add_text(sample["input"])
                vocab.add_text(sample["output"])

            total_files += 1

        print()
        print(f"Datasets Loaded : {total_files}")

    else:

        print("No datasets found.")

    vocab.build()
    print()

    print("Building Dataset...")

    dataset = ConversationDataset(

        dataset_files,

        tokenizer,

    )

    print("      ✓ Dataset Ready")
    print(f"Samples : {len(dataset)}")

    print()

    print("Creating DataLoader...")

    dataloader = create_dataloader(

        dataset,

        batch_size=CONFIG.batch_size,

        shuffle=True,

    )

    print("      ✓ DataLoader Ready")

    print()

    batch = next(iter(dataloader))

    print("Batch Ready")

    print("Input Shape :", batch["input_ids"].shape)

    print("Label Shape :", batch["labels"].shape)

    print("Mask Shape  :", batch["attention_mask"].shape)

    if dataset_files:

        Path("checkpoints").mkdir(
            exist_ok=True
        )

        vocab.save(
            Path("checkpoints/vocab.json")
        )

        print(
            f"Vocabulary Size : {len(vocab)}"
        )

        stats.report()

    print("      ✓ Tokenizer Ready")

    print("[2/5] Creating model...")

    model = create_model(
        vocab_size=len(vocab)
    )

    print("      ✓ Model Ready")

    print("[3/5] Creating optimizer...")

    optimizer = create_optimizer(
        model
    )

    print("      ✓ Optimizer Ready")

    print()

    print("Creating Scheduler...")

    scheduler = create_scheduler_instance(
        optimizer
    )

    print("      ✓ Scheduler Ready")

    print("[4/5] Creating loss function...")

    criterion = create_loss_function()

    print()

    print("Creating Validator...")

    validator = ValidationEngine(

        model=model,

        criterion=criterion,

    )

    print("      ✓ Validator Ready")

    print()

    print("Creating Engine...")

    engine = TrainingEngine(

        model=model,

        optimizer=optimizer,

        criterion=criterion,

    )

    print("      ✓ Engine Ready")

    print()

    print("Creating Trainer...")

    checkpoint = Checkpoint()

    trainer = Trainer(

        engine=engine,

        dataloader=dataloader,

        scheduler=scheduler,

        validator=validator,

        checkpoint=checkpoint,

        epochs=CONFIG.epochs,

    )

    print("      ✓ Trainer Ready")
    start_epoch = 0

    if checkpoint.has_last():

        print()

        print("Loading checkpoint...")

        try:
            state = checkpoint.load_last(

                model=model,

                optimizer=optimizer,

                scheduler=scheduler,

            )

            start_epoch = state["epoch"]

            print(
                f"Resume from epoch {start_epoch}"
            )
        except Exception as e:
            print(
                f"Warning: Could not load checkpoint ({e}). "
                "Vocabulary size or model shape may have changed. Starting training from scratch."
            )

    print("      ✓ Loss Ready")

    print("[5/5] Bootstrap complete.")

    print()
    print("Framework initialized successfully.")

    print()

    print("Starting Training...")

    trainer.train(
        start_epoch=start_epoch
    )

    _ = tokenizer
    _ = optimizer
    _ = criterion

    return 0


if __name__ == "__main__":
    raise SystemExit(main())