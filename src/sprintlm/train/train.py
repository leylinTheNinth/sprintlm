from __future__ import annotations
import os, time, math, csv, numpy as np, torch
from pathlib import Path
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from sprintlm.model.transformer import Transformer
from sprintlm.model.optimizer import AdamW
from sprintlm.nn.functional import cross_entropy_loss, learning_rate_scheduler, gradient_clip

from sprintlm.data.memmap import load_memmap
from sprintlm.data.batch import get_batch
from sprintlm.utils import save_checkpoint, load_checkpoint
from sprintlm.tokenization.tokenizer import Tokenizer

def pick_device(auto: str) -> str:
    if auto != "auto": return auto
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available():         return "cuda"
    return "cpu"

def tokens_per_step(batch_size: int, context_length: int) -> int:
    return batch_size * context_length

@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)
    device = pick_device(cfg.get("device", "auto"))
    print(f"[info] device = {device}")

    # Dataset paths (Hydra changes CWD)
    train_bin = to_absolute_path(cfg.dataset.train_bin)
    val_bin   = to_absolute_path(cfg.dataset.val_bin)

    train_tokens = load_memmap(train_bin, cfg.dataset.dtype)
    val_tokens   = load_memmap(val_bin,   cfg.dataset.dtype)

    # Tokenizer → vocab_size (GPT-2 base; no extra specials here)
    tok = Tokenizer(vocab=None, merges=None, special_tokens=None)
    vocab_size = tok.vocab_size    # 50257 for GPT-2

    # Model
    model = Transformer(
        vocab_size=vocab_size,
        context_length=cfg.dataset.context_length,
        num_layers=cfg.model.num_layers,
        d_model=cfg.model.d_model,
        num_heads=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
        theta=cfg.model.theta,
        device=device,
        dtype=torch.float32,
    ).to(device)

    # Optimizer
    opt = AdamW(model.parameters(),
                lr=cfg.train.optimizer.lr,
                weight_decay=cfg.train.optimizer.weight_decay,
                betas=tuple(cfg.train.optimizer.betas),
                eps=cfg.train.optimizer.eps)

    # Resume
    step = 0
    if cfg.train.resume_from:
        step = load_checkpoint(to_absolute_path(cfg.train.resume_from), model, opt)
        print(f"[info] resumed from {cfg.train.resume_from} @ step {step}")

    # CSV metrics
    csv_f = open("metrics.csv", "w", newline="")
    csv_w = csv.DictWriter(csv_f,
        fieldnames=["step","train_loss","val_loss","ppl","lr","tok_per_s","epoch_equiv"])
    csv_w.writeheader(); csv_f.flush()

    ckpt_dir = Path(cfg.train.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    t_last = time.time()
    TOK_STEP = tokens_per_step(cfg.train.batch_size, cfg.dataset.context_length)
    seen_tokens = step * TOK_STEP
    N_TOK = len(train_tokens)

    # ===== loop =====
    while step < cfg.train.max_steps:
        model.train()
        x, y = get_batch(train_tokens, cfg.train.batch_size, cfg.dataset.context_length, device=device)
        logits = model(x)                         # [B,T,V]
        loss = cross_entropy_loss(logits, y)      # scalar

        # backward
        for p in model.parameters():
            if p.grad is not None: p.grad = None
        loss.backward()
        if cfg.train.grad_clip is not None:
            gradient_clip(model.parameters(), cfg.train.grad_clip)

        # LR schedule
        lr = learning_rate_scheduler(
            it=step,
            max_learning_rate=cfg.train.optimizer.lr,
            min_learning_rate=cfg.train.optimizer.lr * 0.1,
            warmup_iters=cfg.train.scheduler.warmup_steps,
            cosine_cycle_iters=cfg.train.scheduler.cosine_steps,
        )
        for g in opt.param_groups: g["lr"] = lr
        opt.step()

        step += 1
        seen_tokens += TOK_STEP
        dt = time.time() - t_last; t_last = time.time()
        tokps = TOK_STEP / max(dt, 1e-9)
        epoch_eq = seen_tokens / max(N_TOK, 1)

        if step % cfg.train.log_every == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}  lr {lr:.2e}  tok/s {tokps/1e3:5.1f}k  epoch≈{epoch_eq:.3f}")
            csv_w.writerow({"step": step, "train_loss": loss.item(), "val_loss": "", "ppl": "", "lr": lr,
                            "tok_per_s": tokps, "epoch_equiv": epoch_eq})
            csv_f.flush()

        if step % cfg.train.eval_every == 0:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_tokens, cfg.train.batch_size, cfg.dataset.context_length, device=device)
                vloss = cross_entropy_loss(model(vx), vy).item()
            ppl = math.exp(min(20.0, vloss))      # guard overflow
            print(f"[eval] step {step:5d}  val_loss {vloss:.4f}  ppl {ppl:.2f}")
            csv_w.writerow({"step": step, "train_loss": "", "val_loss": vloss, "ppl": ppl, "lr": lr,
                            "tok_per_s": tokps, "epoch_equiv": epoch_eq})
            csv_f.flush()

            if vloss < best_val:
                best_val = vloss
                out = ckpt_dir / f"best_step{step}_val{vloss:.3f}.pt"
                save_checkpoint(model, opt, step, out)
                print(f"[ckpt] saved {out}")

        if step % cfg.train.save_every == 0:
            out = ckpt_dir / f"step{step}.pt"
            save_checkpoint(model, opt, step, out)
            print(f"[ckpt] saved {out}")

    # final
    out = ckpt_dir / f"final_step{step}.pt"
    save_checkpoint(model, opt, step, out)
    print(f"[done] saved {out}")

if __name__ == "__main__":
    main()