import pytest
import torch

from transformer_lab.decoding import decoder
from transformer_lab.tokenizer import Tokenizer
from transformer_lab.transformer_lm import TransformerLM


@pytest.fixture
def tokenizer():
    vocab = {0: b"<|endoftext|>"}
    vocab.update({byte + 1: bytes([byte]) for byte in range(256)})
    return Tokenizer(vocab, [], ["<|endoftext|>"])


@pytest.fixture
def model():
    return TransformerLM(
        vocab_size=257,
        context_length=4,
        d_model=8,
        num_layers=1,
        num_heads=2,
        d_ff=16,
        rope_theta=10000,
    )


@pytest.mark.parametrize("prompt", ["a", "abcdef"])
def test_decoder_generates_beyond_context_window(model, tokenizer, monkeypatch, prompt):
    next_token = tokenizer.encode("x")[0]
    monkeypatch.setattr(torch.distributions.Categorical, "sample", lambda _self: torch.tensor(next_token))
    contexts = []
    model.register_forward_pre_hook(lambda _module, args: contexts.append(list(args[0])))

    output = decoder(model, tokenizer, prompt, temperature=0.5, max_tokens=len(prompt) + 7)

    assert output == prompt + "x" * 7
    assert len(contexts) == 7
    assert max(map(len, contexts)) == 4
    assert tokenizer.decode(contexts[-1]) == "xxxx"


@pytest.mark.parametrize("training", [True, False])
def test_decoder_rejects_empty_prompt(model, tokenizer, training):
    model.train(training)

    with pytest.raises(ValueError, match="prompt must contain at least one token"):
        decoder(model, tokenizer, "", temperature=0.5, max_tokens=8)

    assert model.training is training


def test_decoder_stops_at_end_of_text(model, tokenizer, monkeypatch):
    monkeypatch.setattr(torch.distributions.Categorical, "sample", lambda _self: torch.tensor(0))

    output = decoder(model, tokenizer, "abc", temperature=0.5, max_tokens=10)

    assert output == "abc<|endoftext|>"
