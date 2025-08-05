import torch
import einops
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        temp = torch.square(x)
        rms = torch.sqrt(einops.reduce(temp, " ... d -> ... 1", "mean") + self.eps)
        ans = (x / rms) * self.weight
        return ans.to(in_dtype)
    
