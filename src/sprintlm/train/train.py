from __future__ import annotations
import sys, os, time, math, csv, numpy as np, torch
from pathlib import Path
import hydra
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig  
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


try:
    sys.stdout.reconfigure(line_buffering=True) # type: ignore
except Exception:
    pass

@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] output_dir = {run_dir}", flush=True)
    print(f"[run] metrics -> {run_dir / 'metrics.csv'}", flush=True)
    print(f"[run] ckpts   -> {run_dir / cfg.train.ckpt_dir}", flush=True)

    log_file = open(run_dir / "console.log", "a", buffering=1)  
    def log(msg: str):
        print(msg, flush=True)
        try:
            log_file.write(msg + "\n")
        except Exception:
            pass

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
    csv_path = run_dir / "metrics.csv"
    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.DictWriter(csv_f,
        fieldnames=["step","train_loss","val_loss","ppl","lr","tok_per_s","epoch_equiv"])
    csv_w.writeheader(); csv_f.flush()

    ckpt_dir = (run_dir / cfg.train.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    t_last = time.time()
    TOK_STEP = tokens_per_step(cfg.train.batch_size, cfg.dataset.context_length)
    seen_tokens = step * TOK_STEP
    N_TOK = len(train_tokens)

    # rolling window for smoother tok/s (reduces jitter, less ETA spam)
    window_tokens = 0
    window_time = 0.0
    log_count = 0  # for infrequent CSV flushes

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

        window_tokens += TOK_STEP
        window_time += dt

        if step % cfg.train.log_every == 0:
            # average tok/s over the window
            tokps_avg = window_tokens / max(window_time, 1e-9)
            steps_per_s = tokps_avg / float(TOK_STEP)
            remaining = cfg.train.max_steps - step
            eta_sec = remaining / max(steps_per_s, 1e-9)
            eta_min = int(eta_sec // 60); eta_hr = eta_min // 60; eta_min %= 60

            log(f"step {step:5d}  loss {loss.item():.4f}  lr {lr:.2e}  "
                f"tok/s {tokps_avg/1e3:4.1f}k  epoch≈{epoch_eq:.3f}")
            log(f"ETA → {cfg.train.max_steps} steps: ~{eta_hr}h {eta_min}m  "
                f"at {steps_per_s:.3f} steps/s ({tokps_avg:,.0f} tok/s)")

            csv_w.writerow({
                "step": step, "train_loss": loss.item(), "val_loss": "", "ppl": "",
                "lr": lr, "tok_per_s": tokps_avg, "epoch_equiv": epoch_eq
            })
            log_count += 1
            if log_count % 5 == 0:   
                csv_f.flush()
            # reset window
            window_tokens = 0
            window_time = 0.0

        if step % cfg.train.eval_every == 0:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_tokens, cfg.train.batch_size, cfg.dataset.context_length, device=device)
                vloss = cross_entropy_loss(model(vx), vy).item()
            ppl = math.exp(min(20.0, vloss))      # guard overflow
            log(f"[eval] step {step:5d}  val_loss {vloss:.4f}  ppl {ppl:.2f}")
            csv_w.writerow({"step": step, "train_loss": "", "val_loss": vloss, "ppl": ppl, "lr": lr,
                            "tok_per_s": tokps, "epoch_equiv": epoch_eq})
            csv_f.flush()

            if vloss < best_val:
                best_val = vloss
                out = ckpt_dir / f"best_step{step}_val{vloss:.3f}.pt"
                save_checkpoint(model, opt, step, out)
                log(f"[ckpt] saved {out}")

        if step % cfg.train.save_every == 0:
            out = ckpt_dir / f"step{step}.pt"
            save_checkpoint(model, opt, step, out)
            log(f"[ckpt] saved {out}")

    # final
    out = ckpt_dir / f"final_step{step}.pt"
    save_checkpoint(model, opt, step, out)
    log(f"[done] saved {out}")

if __name__ == "__main__":
    main()