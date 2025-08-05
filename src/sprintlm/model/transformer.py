import torch
import torch.nn as nn
from einops import repeat
from sprintlm.model.embedding import Embedding
from sprintlm.model.linear import Linear
from sprintlm.model.rmsnorm import RMSNorm
from sprintlm.model.block import TransformerBlock



class Transformer(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, 
                 d_model: int, num_heads: int, d_ff: int, theta: float, device=None, dtype=None):
        super().__init__()
        
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, theta,
                             device=device, dtype=dtype)
            for _ in range(num_layers)                 
        ])
        #last rms norm
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        #final linear projection
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids):
        x = self.token_embeddings(token_ids)
        B, T = token_ids.shape
        token_positions = torch.arange(T, device=token_ids.device)
        token_positions = repeat(token_positions, "t -> b t", b = B)

        for layer in self.layers:
            x = layer(x, token_positions)
        
        logits = self.lm_head(self.ln_final(x))
        return logits