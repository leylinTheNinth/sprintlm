# tools/prepare_split.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import tiktoken

def _encode_file(path: str, enc) -> list[int]:
    # Read full text to preserve newlines; allow specials so "<|endoftext|>" is a single token.
    text = Path(path).read_text(encoding="utf-8")
    ids = enc.encode(text, allowed_special="all")
    return ids

@hydra.main(config_path="../configs", config_name="prepare_tinystories", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding(cfg.tokenizer_name)  # "gpt2"
    eot_id = enc.encode("<|endoftext|>", allowed_special="all")[0]

    # Encode files (no extra EOT added)
    train_ids = _encode_file(cfg.train_txt, enc)
    val_ids   = _encode_file(cfg.val_txt,   enc)

    # Sanity: count EOTs present
    n_eot_train = sum(1 for x in train_ids if x == eot_id)
    n_eot_val   = sum(1 for x in val_ids   if x == eot_id)
    print(f"[info] EOT tokens found — train: {n_eot_train:,}, val: {n_eot_val:,}")

    # Allocate + write memmaps
    dtype = np.uint16 if cfg.dtype == "uint16" else np.uint32
    train_mm = np.memmap(out / "train.bin", dtype=dtype, mode="w+", shape=(len(train_ids),))
    val_mm   = np.memmap(out / "val.bin",   dtype=dtype, mode="w+", shape=(len(val_ids),))

    train_mm[:] = np.asarray(train_ids, dtype=dtype)
    val_mm[:]   = np.asarray(val_ids,   dtype=dtype)

    train_mm.flush(); val_mm.flush()
    print(f"Wrote {len(train_ids):,} train tokens, {len(val_ids):,} val tokens to {out.resolve()}")

if __name__ == "__main__":
    main()