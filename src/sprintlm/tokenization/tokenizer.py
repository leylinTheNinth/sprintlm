from __future__ import annotations
from typing import Iterable, Iterator
import tiktoken

class Tokenizer:
    def __init__(self,
                 vocab: dict[int, bytes] | None = None,
                 merges: list[tuple[bytes, bytes]] | None = None,
                 special_tokens: list[str] | None = None,
                 name: str = "gpt2"):
        self.enc = tiktoken.get_encoding(name)
        self.base_vocab_size = self.enc.n_vocab                     # 50257 for gpt2
        self.special_tokens = list(special_tokens or [])            # e.g., ["<|bos|>", "<|eos|>"]
        # Assign special IDs sequentially above base vocab
        self.special_to_id = {tok: self.base_vocab_size + i for i, tok in enumerate(self.special_tokens)}
        self.id_to_special = {v: k for k, v in self.special_to_id.items()}
        self.vocab_size = self.base_vocab_size + len(self.special_tokens)

        self.bos_id = self.special_to_id.get("<|bos|>", None)
        self.eos_id = self.special_to_id.get("<|eos|>", None)

    @classmethod
    def from_files(cls,
                   vocab_filepath: str | None,
                   merges_filepath: str | None,
                   special_tokens: list[str] | None = None,
                   name: str = "gpt2") -> "Tokenizer":
        return cls(None, None, special_tokens, name=name)

    def encode(self, text: str) -> list[int]:
        # If you need BOS/EOS behavior, add them in your dataset/loader, not here.
        return self.enc.encode(text)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            # chunk can be a line or a big string; we simply yield its token IDs
            ids = self.enc.encode(chunk)
            for i in ids:
                yield i

    def decode(self, ids: list[int]) -> str:
        # Drop special IDs (>= base_vocab_size) for plain text reconstruction
        base_ids = [i for i in ids if i < self.base_vocab_size]
        return self.enc.decode(base_ids)