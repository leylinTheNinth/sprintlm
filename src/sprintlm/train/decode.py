import torch
from typing import Optional
from sprintlm.tokenization.tokenizer import Tokenizer

GPT2_EOT_ID = 50256

@torch.no_grad()
def generate(
    model,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,                   # 0 = no top-k (pure softmax sampling / greedy if temp→0)
    stop_at_eot: bool = True,
    device: Optional[str] = None,
    context_length: Optional[int] = None,
) -> str:
        
    if device is None or device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available()
                  else "cpu")
    model = model.to(device).eval()

    ids = tokenizer.encode(prompt)
    if not isinstance(ids, (list, tuple)):
        ids = list(ids)
    if len(ids) == 0:
        ids = [GPT2_EOT_ID]  # avoid empty input
    ids_t = torch.tensor([ids], dtype=torch.long, device=device)

    C = context_length
    if C is None:
        C = getattr(model, "context_length", None) or ids_t.shape[1]

    for _ in range(max_new_tokens):
        x = ids_t[:, -C:]
        logits = model(x)[:, -1, :]  # (1, V)
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-8)

        if top_k and top_k > 0 and top_k < logits.shape[-1]:
            vals, _ = torch.topk(logits, top_k)
            thresh = vals[..., -1, None]
            logits = torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)

        # sample next id
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # shape (1, 1)

        ids_t = torch.cat([ids_t, next_id], dim=1)

        # early stop on EOT
        if stop_at_eot and int(next_id.item()) == GPT2_EOT_ID:
            break

    out_ids = ids_t[0].tolist()
    if stop_at_eot and GPT2_EOT_ID in out_ids:
        out_ids = out_ids[: out_ids.index(GPT2_EOT_ID)]

    return tokenizer.decode(out_ids)