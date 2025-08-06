# SprintLM — tiny GPT from scratch (TinyStories)

**Docs / Demo:** https://leylinTheNinth.github.io/sprintlm
**Result:** val loss **2.713** → ppl **15.07** after **23.0M tokens** at ~**7.5k tok/s** (B=24, T=192, 5k steps, 1× T4).

## Quick start
```bash
uv sync

# 1) Prepare dataset (HF → memmap, GPT-2 BPE, EOT per story)
uv run python tools/prepare_hf.py \
  dataset_name=roneneldan/TinyStories text_field=text \
  out_dir=data/Prepared/TinyStoriesHF append_eot=true \
  dtype=uint16 tokenizer_name=gpt2

# 2) Train (~50–55 min on a T4)
uv run python -m sprintlm.train.train \
  dataset.train_bin=data/Prepared/TinyStoriesHF/train.bin \
  dataset.val_bin=data/Prepared/TinyStoriesHF/val.bin \
  train.batch_size=24 dataset.context_length=192 \
  train.max_steps=5000 train.log_every=50 train.eval_every=250 train.save_every=250

# 3) Visualize + sample
uv run python tools/plot_metrics.py outputs
uv run python tools/decode_cli.py \
  --ckpt outputs/<RUN>/ckpts/step5000.pt \
  --prompt "Once upon a time, " --context_length 192 \
  --temperature 0.8 --top_k 40
```
### Acknowledgments
SprintLM is extension of the public **Stanford CS336 — Assignment 1: “Transformers from Scratch.”**. 
