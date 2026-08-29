from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


class DatasetReader:
    """
    Universal dataset reader.

    Supported formats:
    - .txt    (input|output)
    - .jsonl  ({"input":"...","output":"..."})
    - .json   ([{"input":"...","output":"..."}])
    """

    def __init__(self, path: str | Path):

        self.path = Path(path)

    def __iter__(self) -> Iterator[dict]:

        suffix = self.path.suffix.lower()

        if suffix == ".txt":
            yield from self._read_txt()

        elif suffix == ".jsonl":
            yield from self._read_jsonl()

        elif suffix == ".json":
            yield from self._read_json()

        else:
            raise ValueError(
                f"Unsupported dataset format: {suffix}"
            )

    def _read_txt(self):

        with open(
            self.path,
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(file, start=1):

                line = line.strip()

                if not line:
                    continue

                if "|" not in line:
                    raise ValueError(
                        f"Line {line_number}: missing '|' separator."
                    )

                input_text, output_text = line.split("|", 1)

                yield {
                    "input": input_text.strip(),
                    "output": output_text.strip(),
                }

    def _read_jsonl(self):

        with open(
            self.path,
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(file, start=1):

                line = line.strip()

                if not line:
                    continue
                
                if line.startswith("#"):
                    continue

                try:
                    item = json.loads(line)

                except json.JSONDecodeError as exc:

                    raise ValueError(
                        f"Line {line_number}: invalid JSON."
                    ) from exc

                if "input" not in item or "output" not in item:
                    raise ValueError(
                        f"Line {line_number}: missing input/output."
                    )

                yield {
                    "input": item["input"],
                    "output": item["output"],
                }

    def _read_json(self):

        with open(
            self.path,
            "r",
            encoding="utf-8",
        ) as file:

            try:
                data = json.load(file)

            except json.JSONDecodeError as exc:

                raise ValueError(
                    "Invalid JSON dataset."
                ) from exc

        if not isinstance(data, list):
            raise ValueError(
                "JSON dataset must be a list."
            )

        for index, item in enumerate(data, start=1):

            if not isinstance(item, dict):
                raise ValueError(
                    f"Item {index}: must be an object."
                )

            if "input" not in item or "output" not in item:
                raise ValueError(
                    f"Item {index}: missing input/output."
                )

            yield {
                "input": item["input"],
                "output": item["output"],
            }