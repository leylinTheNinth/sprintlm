import torch
from sprintlm.model.linear import Linear
from sprintlm.model.ffn import FeedFowardNetwork
from sprintlm.model.rmsnorm import RMSNorm
from sprintlm.model.rope import RotaryPositionalEmbedding
from sprintlm.model.attention import MultiHeadSelfAttention


def test_linear_shapes():
    lin = Linear(7, 3)
    x = torch.randn(2,5,7)
    y = lin(x)
    assert y.shape == (2,5,3)

def test_ffn_shapes():
    ff = FeedFowardNetwork(16, 64)
    x = torch.randn(4,8,16)
    y = ff(x)
    assert y.shape == (4,8,16)

def test_rmsnorm_basic():
    rn = RMSNorm(10)
    x = torch.randn(2,3,10, dtype=torch.float16)
    y = rn(x)
    assert y.shape == (2,3,10)
    assert y.dtype == x.dtype

def test_rope_pairs():
    d_k, S, B = 8, 6, 2
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=d_k, max_seq_len=32)
    q = torch.randn(B, S, d_k)
    pos = torch.arange(S).unsqueeze(0).expand(B, S)
    out = rope(q, pos)
    assert out.shape == (B, S, d_k)

def test_attention_positions_batch():
    attn = MultiHeadSelfAttention(d_model=32, num_heads=4, max_seq_len=16, theta=10000.0)
    x = torch.randn(3, 8, 32)                         # B=3, T=8
    # explicit positions
    pos = torch.arange(8).unsqueeze(0).expand(3, -1)  # (3,8)
    y1 = attn(x, pos)
    # None (should still work for B>1)
    y2 = attn(x, None)
    assert y1.shape == y2.shape == (3, 8, 32)