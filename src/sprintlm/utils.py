import os, torch, time
from typing import Any

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    itertion: int,
    out: str | os.PathLike
):
    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": itertion,
        "meta": {
            "saved_at": time.time(),
            "model_class": model.__class__.__name__,
            "param_count": sum(p.numel() for p in model.parameters()),
            "torch_version": torch.__version__,
        }
    }
    torch.save(checkpoint, out)

def load_checkpoint(
    src: str | os.PathLike,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
):
    checkpoint = torch.load(src, map_location="cpu")  # for portability
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # moving optimizer state to model device 
    dev = next(model.parameters()).device
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(dev)
    return checkpoint.get("iteration", 0)