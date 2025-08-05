import torch
import torch.nn as nn
from sprintlm.model.rmsnorm import RMSNorm
from sprintlm.model.attention import MultiHeadSelfAttention
from sprintlm.model.ffn import FeedFowardNetwork


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.ffn = FeedFowardNetwork(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        y = x + self.attn(self.ln1(x), token_positions)
        z = y + self.ffn(self.ln2(y))
        return z
