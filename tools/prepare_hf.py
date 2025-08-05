from __future__ import annotations
from pathlib import Path
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

from datasets import load_dataset
import tiktoken

def _iter_text(ds, text_field: str, limit: int | None):
    n = len(ds) if limit is None else min(len(ds), limit)
    for i in range(n):
        yield ds[i][text_field]

def _count_tokens(ds, text_field, enc, append_eot: bool, eot_id: int, limit: int | None):
    n_tok = 0
    n_eot = 0
    for s in _iter_text(ds, text_field, limit):
        #  in case the string "<|endoftext|>" appears
        ids = enc.encode(s, allowed_special="all")
        if append_eot and (len(ids) == 0 or ids[-1] != eot_id):
            ids.append(eot_id)
            n_eot += 1
        n_tok += len(ids)
    return n_tok, n_eot

def _write_tokens(ds, text_field, enc, mm: np.memmap, dtype, append_eot: bool, eot_id: int, limit: int | None):
    i = 0
    for s in _iter_text(ds, text_field, limit):
        ids = enc.encode(s, allowed_special="all")
        if append_eot and (len(ids) == 0 or ids[-1] != eot_id):
            ids.append(eot_id)
        arr = np.asarray(ids, dtype=dtype)
        mm[i:i+len(arr)] = arr
        i += len(arr)
    mm.flush()

@hydra.main(config_path="../configs", config_name="prepare_hf", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    ds_train = load_dataset(cfg.dataset_name, split=cfg.train_split)
    ds_val   = load_dataset(cfg.dataset_name, split=cfg.val_split)

    enc = tiktoken.get_encoding(cfg.tokenizer_name)  # "gpt2"
    eot_id = enc.encode("<|endoftext|>", allowed_special="all")[0]

    dtype = np.uint16 if cfg.dtype == "uint16" else np.uint32
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)

    n_train, eot_train = _count_tokens(ds_train, cfg.text_field, enc, cfg.append_eot, eot_id, cfg.max_records_train)
    n_val,   eot_val   = _count_tokens(ds_val,   cfg.text_field, enc, cfg.append_eot, eot_id, cfg.max_records_val)
    print(f"[info] will write {n_train:,} train tokens ({eot_train:,} EOT), {n_val:,} val tokens ({eot_val:,} EOT)")

    train_mm = np.memmap(out / "train.bin", dtype=dtype, mode="w+", shape=(n_train,))
    val_mm   = np.memmap(out / "val.bin",   dtype=dtype, mode="w+", shape=(n_val,))

    _write_tokens(ds_train, cfg.text_field, enc, train_mm, dtype, cfg.append_eot, eot_id, cfg.max_records_train)
    _write_tokens(ds_val,   cfg.text_field, enc, val_mm,   dtype, cfg.append_eot, eot_id, cfg.max_records_val)

    print(f"[done] wrote to {out.resolve()}")
    print(f" train.bin: {n_train:,} tokens  | val.bin: {n_val:,} tokens")

if __name__ == "__main__":
    main()