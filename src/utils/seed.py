import random

import numpy as np


def seed_randomizers(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
