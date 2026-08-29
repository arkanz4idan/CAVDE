from __future__ import annotations

from collections import Counter


class DatasetStatistics:

    def __init__(self):

        self.samples = 0

        self.input_words = Counter()

        self.output_words = Counter()

    def update(self, sample):

        self.samples += 1

        self.input_words.update(
            sample["input"].split()
        )

        self.output_words.update(
            sample["output"].split()
        )

    def report(self):

        print()

        print("=" * 45)

        print("Dataset Statistics")

        print("=" * 45)

        print(f"Samples : {self.samples}")

        print(
            f"Unique Input Words : {len(self.input_words)}"
        )

        print(
            f"Unique Output Words : {len(self.output_words)}"
        )

        print()