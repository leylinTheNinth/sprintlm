from sprintlm.tokenization.tokenizer import Tokenizer

def test_roundtrip_basic():
    tok = Tokenizer(special_tokens=["<|bos|>", "<|eos|>"])
    s = "Hello, SprintLM!"
    ids = tok.encode(s)
    assert "SprintLM" in tok.decode(ids)

def test_vocab_sizes():
    tok = Tokenizer(special_tokens=["<|bos|>", "<|eos|>"])
    assert tok.base_vocab_size > 50000
    assert tok.vocab_size == tok.base_vocab_size + 2