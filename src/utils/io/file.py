from pathlib import Path
from typing import Any, Union

import yaml


def load_yaml(file_path: Union[str, Path]) -> dict[Any, Any]:
    with open(file_path, encoding="utf-8") as f:
        data: dict = yaml.safe_load(f)
        return data
