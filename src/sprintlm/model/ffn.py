import torch
import math
import einops
import torch.nn as nn

from sprintlm.model.linear import Linear


class FeedFowardNetwork(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.w1(x)
        v = self.w3(x)
        silu = u * torch.sigmoid(u)     # SwiGLU = SiLU(u) * v
        return self.w2(silu * v)
    