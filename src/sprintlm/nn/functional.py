import torch
import torch.nn as nn
import einops
import math


def softmax(x: torch.Tensor, dim: int):
    max_value = torch.max(x, dim=dim, keepdim=True)[0]
    adjusted_value = x - max_value
    exp_value = torch.exp(adjusted_value)
    sum_value = torch.sum(exp_value, dim=dim, keepdim=True)
    return exp_value/sum_value


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
    d_k = k.shape[-1]
    scaled_dot_product = einops.einsum(q, k, "... n d_k, ... m d_k -> ... n m ")/math.sqrt(d_k)
    if mask is not None:
        scaled_dot_product = scaled_dot_product.masked_fill(~mask, float("-inf"))
    attention = softmax(scaled_dot_product, dim=-1)
    return einops.einsum(attention, v, "... n m, ... m d_v -> ... n d_v")


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor):
    max_value = torch.max(logits, dim=-1, keepdim=True)[0]
    adjusted_logits = logits - max_value
    flat_logits = einops.rearrange(adjusted_logits, "... v -> (...) v")
    index = einops.rearrange(targets, "... -> (...) 1")
    target_logits = flat_logits.gather(dim=-1, index=index.to(torch.int64))
    log_exp_logits = torch.log(torch.exp(flat_logits).sum(dim=-1, keepdim=True))
    loss = -1*target_logits + log_exp_logits
    return loss.mean()


def learning_rate_scheduler(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    # warm-up (t < Tw)
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    # cosine annealing (Tw ≤ t ≤ Tc)  
    elif it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_factor = 0.5 * (1 + math.cos(progress * math.pi))
        return min_learning_rate + cosine_factor * (max_learning_rate - min_learning_rate)
    # post annealing (t > Tc)
    else:
        return min_learning_rate
    

#careful this is not per parameter clipping but on whole gradient vector
def gradient_clip(parameters, max_l2_norm: float) -> None:
    eps = 1e-6  #default
    total_norm = 0.0
    params_with_grad = []    
    for param in parameters:
        if param.grad is not None:
            params_with_grad.append(param)
            param_norm = param.grad.data.norm(dtype=torch.float32)
            total_norm += param_norm.item() ** 2
    
    total_norm = total_norm ** 0.5    
    if total_norm > max_l2_norm:
        clip_coeff = max_l2_norm / (total_norm + eps)        
        for param in params_with_grad:
            param.grad.data.mul_(clip_coeff)

