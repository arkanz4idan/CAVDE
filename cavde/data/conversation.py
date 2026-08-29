from __future__ import annotations

from torch.utils.data import Dataset


class ConversationDataset(Dataset):

    def __init__(
        self,
        samples,
        tokenizer,
    ):

        self.samples = list(samples)

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        conversation = (
            sample["input"]
            + " "
            + sample["output"]
        )

        tokens = self.tokenizer.encode(
            conversation,
            add_special_tokens=True,
        )

        input_ids = tokens[:-1]

        labels = tokens[1:]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }