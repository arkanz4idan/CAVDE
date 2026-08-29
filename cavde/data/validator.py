from __future__ import annotations

from typing import Iterable


class DatasetValidator:

    """
    Validate dataset samples before training.
    """

    def validate(self, samples: Iterable[dict]):

        errors = []

        valid_samples = []

        for index, sample in enumerate(samples, start=1):

            if "input" not in sample:
                errors.append(
                    f"Line {index}: missing 'input'."
                )
                continue

            if "output" not in sample:
                errors.append(
                    f"Line {index}: missing 'output'."
                )
                continue

            if not sample["input"].strip():
                errors.append(
                    f"Line {index}: empty input."
                )
                continue

            if not sample["output"].strip():
                errors.append(
                    f"Line {index}: empty output."
                )
                continue

            valid_samples.append(sample)

        return valid_samples, errors