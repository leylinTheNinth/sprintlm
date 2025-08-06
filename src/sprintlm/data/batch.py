import numpy as np
import torch

# must load dataset using np.memmap
def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device=None):
    idx = np.random.randint(0, len(dataset)-context_length, batch_size)
    # np array will always be on cpu
    inp = np.zeros((batch_size, context_length), dtype=np.int64)
    tgt = np.zeros((batch_size, context_length), dtype=np.int64)
    for i, j in enumerate(idx):
        inp[i] = dataset[j : j + context_length]
        tgt[i] = dataset[j+1 : j+1 + context_length]
    return torch.from_numpy(inp).to(device=device), torch.from_numpy(tgt).to(device=device)