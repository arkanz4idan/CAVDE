from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from cavde.data.reader import DatasetReader


class ConversationDataset(Dataset):

    def __init__(
        self,
        paths: Path | list[Path],
        tokenizer,
    ):

        self.tokenizer = tokenizer
        self.samples = []

        if isinstance(paths, Path):
            paths = [paths]

        bos = tokenizer.vocab.token_to_id["<BOS>"]
        eos = tokenizer.vocab.token_to_id["<EOS>"]
        sep = tokenizer.vocab.token_to_id["<SEP>"]

        for path in paths:

            reader = DatasetReader(path)

            for sample in reader:

                input_tokens = tokenizer.encode(
                    sample["input"],
                    add_special_tokens=False,
                )

                output_tokens = tokenizer.encode(
                    sample["output"],
                    add_special_tokens=False,
                )

                sequence = (
                    [bos]
                    + input_tokens
                    + [sep]
                    + output_tokens
                    + [eos]
                )

                input_ids = sequence[:-1]

                labels = sequence[1:]

                attention_mask = [1] * len(input_ids)

                self.samples.append(
                    {
                        "input_ids": input_ids,
                        "labels": labels,
                        "attention_mask": attention_mask,
                    }
                )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        return self.samples[index]