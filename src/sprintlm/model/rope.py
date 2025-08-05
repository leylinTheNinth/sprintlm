import torch
import math
import einops
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, *, device=None):
        super().__init__()
        inv_freq = theta ** (
            -torch.arange(0, d_k, 2, device=device).float() / d_k
        )  # (d_k/2,)
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        assert d_k % 2 == 0, "RoPE d_k must be even"
        dup_freq = torch.repeat_interleave(inv_freq, 2)          # (d_k,)
        positions = torch.arange(max_seq_len, device=device).float()  # (seq,)
        angles = torch.outer(positions, dup_freq)                # (seq, d_k)
        self.register_buffer("cos_val", angles.cos(), persistent=False)
        self.register_buffer("sin_val", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_val[token_positions]  # type: ignore # (..., seq, d_k)
        sin = self.sin_val[token_positions] # type: ignore
        t = einops.rearrange(x, "... (d two) -> ... d two", two=2)
        rot = torch.stack((-t[..., 1], t[..., 0]), dim=-1)
        rot = einops.rearrange(rot, "... d two -> ... (d two)", two=2)
        return x * cos + rot * sin