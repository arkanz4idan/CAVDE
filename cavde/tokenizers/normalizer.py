from __future__ import annotations

import re
import unicodedata


class TextNormalizer:

    def normalize(self, text: str) -> str:

        text = unicodedata.normalize("NFKC", text)

        text = text.lower()

        text = re.sub(r"([.,!?;:()])", r" \1 ", text)

        text = re.sub(r"\s+", " ", text)

        return text.strip()