import sys, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

def make_plot(run: Path):
    df = pd.read_csv(run / "metrics.csv")

    plt.figure()
    df[df["train_loss"].notna()].plot(x="step", y="train_loss", legend=False)
    plt.title("Train Loss"); plt.ylabel("loss"); plt.xlabel("step"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(run / "loss_train.png", dpi=150)

    plt.figure()
    df[df["val_loss"].notna()].plot(x="step", y="val_loss", legend=False)
    plt.title("Val Loss"); plt.ylabel("loss"); plt.xlabel("step"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(run / "loss_val.png", dpi=150)

    plt.figure()
    df[df["tok_per_s"].notna()].plot(x="step", y="tok_per_s", legend=False)
    plt.title("Tokens / second"); plt.ylabel("tok/s"); plt.xlabel("step"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(run / "tokps.png", dpi=150)

if __name__ == "__main__":
    base = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs")
    if (base / "metrics.csv").exists():
        make_plot(base)
    else:
        runs = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)
        if not runs:
            raise SystemExit("No run directory found under 'outputs/'")
        make_plot(runs[-1])