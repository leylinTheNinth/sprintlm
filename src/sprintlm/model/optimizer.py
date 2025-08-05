from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, weight_decay: float, betas: tuple, eps: float):
        defaults = {"lr": lr, "betas": betas, "lambda": weight_decay, "eps":eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> Optional[float]: # type: ignore
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["lambda"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                m = state.get("m", torch.zeros(p.shape, device=p.device))
                v = state.get("v", torch.zeros(p.shape, device=p.device))
                t = state.get("t", 1)
                grad = p.grad.data
                m = b1*m + (1-b1)*grad
                v = b2*v + (1-b2)*(grad**2)
                alpha = lr*(math.sqrt(1 - b2**t)/(1- b1**t))
                p.data -= alpha*m*(1/torch.sqrt(v+eps))
                p.data -= lr*weight_decay*p.data
                state["m"] = m
                state["v"] = v
                state["t"] = t+1
        return loss
        
