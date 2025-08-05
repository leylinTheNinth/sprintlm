import torch
import math
import einops
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        w = torch.empty(out_features, in_features, dtype=dtype, device=device)
        sigma = math.sqrt(2.0/(in_features+out_features))
        self.weight = nn.Parameter(nn.init.trunc_normal_(w, 0, sigma, -3*sigma, 3*sigma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      #  y = einops.einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        y = x@self.weight.t() # let's see if this speeds up
        return y
    