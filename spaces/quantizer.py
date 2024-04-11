from typing import Tuple

import numpy as np


class Quantizer:
    """Quantize a continuous element to index of a given table."""
    def __init__(self, table: np.ndarray):
        self.table = table

    def element_to_idx(self, element: float) -> int:
        return np.argmin(np.abs(self.table - element))

    def idx_to_element(self, idx: int) -> float:
        return self.table[idx].item()


