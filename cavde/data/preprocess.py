from __future__ import annotations

import json
from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".jsonl",
}


class DatasetValidationError(Exception):
    pass


def validate_dataset(path: Path) -> None:

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise DatasetValidationError(
            f"Unsupported dataset: {path.suffix}"
        )


def iter_dataset(path: Path):

    validate_dataset(path)

    if path.suffix == ".txt":

        with open(path, "r", encoding="utf-8") as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                if "|" not in line:
                    continue

                question, answer = line.split(
                    "|",
                    1,
                )

                yield {
                    "input": question.strip(),
                    "output": answer.strip(),
                }

    else:

        with open(path, "r", encoding="utf-8") as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                yield json.loads(line)