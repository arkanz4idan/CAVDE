from __future__ import annotations

import torch


class GreedyGenerator:

    def __init__(
        self,
        model,
        tokenizer,
        max_new_tokens=32,
        temperature=1.0,
        top_k=None,
        top_p=None,
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
    ):

        self.model.eval()

        ids = self.tokenizer.encode(prompt)

        bos = self.tokenizer.vocab.token_to_id["<BOS>"]
        eos = self.tokenizer.vocab.token_to_id["<EOS>"]

        tokens = [bos] + ids

        for _ in range(self.max_new_tokens):

            x = torch.tensor(
                [tokens],
                dtype=torch.long,
            )

            logits = self.model(x)

            logits = logits[0, -1]

            logits = logits / self.temperature

            # ===========================
            # TOP-K
            # ===========================

            if self.top_k is not None:

                values, indices = torch.topk(
                    logits,
                    k=min(
                        self.top_k,
                        logits.size(-1),
                    ),
                )

                logits = torch.full_like(
                    logits,
                    float("-inf"),
                )

                logits[indices] = values

            # ===========================
            # TOP-P (Nucleus)
            # ===========================

            if self.top_p is not None:

                sorted_logits, sorted_indices = torch.sort(
                    logits,
                    descending=True,
                )

                probs = torch.softmax(
                    sorted_logits,
                    dim=-1,
                )

                cumulative = torch.cumsum(
                    probs,
                    dim=-1,
                )

                remove = cumulative > self.top_p

                remove[1:] = remove[:-1].clone()

                remove[0] = False

                sorted_logits[remove] = float("-inf")

                logits = torch.full_like(
                    logits,
                    float("-inf"),
                )

                logits[sorted_indices] = sorted_logits

            # ===========================
            # SAMPLE
            # ===========================

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            ).item()

            if next_token == eos:
                break

            tokens.append(next_token)

        return self.tokenizer.decode(
            tokens[1:]
        )