#!/usr/bin/env python3
import argparse, torch
from sprintlm.model.transformer import Transformer
from sprintlm.tokenization.tokenizer import Tokenizer
from sprintlm.train.decode import generate

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to .pt checkpoint (full or weights-only)")
    p.add_argument("--prompt", required=True)
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--context_length", type=int, default=192)
    p.add_argument("--device", default="auto")      # auto|mps|cuda|cpu
    # model dims (match your training)
    p.add_argument("--num_layers", type=int, default=12)
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--d_ff", type=int, default=1536)
    p.add_argument("--theta", type=float, default=10000.0)
    args = p.parse_args()

    tok = Tokenizer(None, None, special_tokens=None)

    model = Transformer(
        vocab_size=tok.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
    ).eval()

    try:
        obj = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        state_dict = obj  # already a state_dict
    except TypeError:
        # older PyTorch without weights_only param
        obj = torch.load(args.ckpt, map_location="cpu")
        state_dict = obj if isinstance(obj, dict) and "model_state_dict" not in obj else obj.get("model_state_dict", obj)
    except Exception:
        # PyTorch >=2.6 defaulted to weights_only=True; force legacy loader
        obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        state_dict = obj if isinstance(obj, dict) and "model_state_dict" not in obj else obj.get("model_state_dict", obj)

    model.load_state_dict(state_dict)

    text = generate(
        model, tok, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
        context_length=args.context_length,
    )
    print(text)

if __name__ == "__main__":
    main()