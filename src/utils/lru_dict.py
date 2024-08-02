from collections import OrderedDict
from typing import Any


class LruDict(OrderedDict):
    """Dict with a limited length, ejecting LRUs as needed."""

    def __init__(self, *args: Any, cache_len: int = 10, **kwargs: Any) -> None:
        assert cache_len > 0
        self.cache_len = cache_len
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, value)
        super().move_to_end(key)
        while len(self) > self.cache_len:
            oldkey = next(iter(self))
            super().__delitem__(oldkey)

    def __getitem__(self, key: Any) -> Any:
        val = super().__getitem__(key)
        super().move_to_end(key)
        return val
