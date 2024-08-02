from random import randint
from typing import Generator, List


class TextSpanSplitter:
    """A class to split a given text into multiple contiguous text spans with
    random lengths. This class is used to take text spans longer than single
    sentences (and shorter also) to pre-train a LLM as described in BERT paper.
    """

    def __init__(self, min_len: int, max_len: int):
        self.min_len = min_len
        self.max_len = max_len

    def _random_lens(self, total_len: int) -> List[int]:
        if total_len < self.min_len:
            return []
        start = 0
        lens = []
        while start < total_len:
            remaining = total_len - start
            if remaining < self.min_len:
                break
            new_len = min(randint(self.min_len, self.max_len), remaining)
            lens.append(new_len)
            start += new_len

        return lens

    def __call__(self, text: str) -> Generator[str, None, None]:
        span_lens = self._random_lens(len(text))
        start = 0
        for span_len in span_lens:
            end = start + span_len
            yield text[start:end]
            start += span_len
