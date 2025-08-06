import numpy as np
from pathlib import Path

def load_memmap(path: str | Path, dtype: str = "uint16") -> np.ndarray:
    return np.memmap(path, dtype=getattr(np, dtype), mode="r")