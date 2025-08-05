import torch
import torch.nn as nn
import einops
from sprintlm.model.linear import Linear
from sprintlm.model.rmsnorm import RMSNorm
from sprintlm.model.rope import RotaryPositionalEmbedding
from sprintlm.nn.functional import scaled_dot_product_attention


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_model = d_model

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if max_seq_len is not None and theta is not None:
            d_k = d_model // num_heads
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=device)
        else:
            self.rope = None
        self.register_buffer(
            "causal_mask",
            None if max_seq_len is None else torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, token_positions=None):
        # projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # (..., seq, d_model) -> (..., h, seq, d_k)
        q_head = einops.rearrange(q, "... seq_len (h d_k) -> ... h seq_len d_k ", h=self.num_heads)
        k_head = einops.rearrange(k, "... seq_len (h d_k) -> ... h seq_len d_k ", h=self.num_heads)
        v_head = einops.rearrange(v, "... seq_len (h d_v) -> ... h seq_len d_v ", h=self.num_heads)

        seq_len = x.size(-2)
        B = q_head.shape[0] if q_head.dim() >= 4 else 1

        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(B, -1)  # (B, seq)

        # head axis -> (B, h, seq)
        token_positions = token_positions.unsqueeze(1).expand(-1, self.num_heads, -1)

        # sanity check
        assert q_head.shape[:3] == token_positions.shape, f"{q_head.shape} vs {token_positions.shape}"

        # RoPE on Q/K if available
        if self.rope is not None:
            q_head = self.rope(q_head, token_positions)
            k_head = self.rope(k_head, token_positions)

        # causal mask (seq, seq) -> broadcast to (B,h,seq,seq)
        if self.causal_mask is not None and self.causal_mask.size(0) >= seq_len: # type: ignore
            mask = self.causal_mask[:seq_len, :seq_len] # type: ignore
        else:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

        multihead = scaled_dot_product_attention(q_head, k_head, v_head, mask)
        head = einops.rearrange(multihead, "... h seq_len d_k -> ... seq_len (h d_k)", h=self.num_heads)
        return self.output_proj(head)