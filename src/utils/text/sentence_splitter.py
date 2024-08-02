import re
import unicodedata
from typing import List


class SentenceSplitter:
    SENT_SEPARATOR = re.compile(r"([．。?!？！\n])")

    def __init__(self, normalize_text: bool = True) -> None:
        self.normalize_text = normalize_text

    def __call__(self, text: str) -> List[str]:
        parts = self.SENT_SEPARATOR.split(text)
        sents = []
        for i in range(0, len(parts), 2):
            text_part = parts[i].strip()
            delim_part = ""
            if i + 1 < len(parts):
                delim_part = parts[i + 1].strip()
            sent = text_part + delim_part
            if sent:
                if self.normalize_text:
                    sent = unicodedata.normalize("NFKC", sent)
                sents.append(sent)

        return sents
