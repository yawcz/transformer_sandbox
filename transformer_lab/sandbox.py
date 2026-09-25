import torch as t

from transformer_lab.adamw import AdamW
from transformer_lab.checkpointing import load_checkpoint
from transformer_lab.decoding import decoder
from transformer_lab.training_together import load_tokenizer
from transformer_lab.transformer_lm import TransformerLM

vocab_size = 10000
context_length = 256
d_model = 512
d_ff = 1344
rope_theta = 10000
num_layers = 4
num_heads = 16
weight_decay = 0.01
betas = (0.9, 0.999)

device = t.device("cuda" if t.cuda.is_available() else "cpu")

model = TransformerLM(
    vocab_size=vocab_size,
    context_length=context_length,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    rope_theta=rope_theta,
    device=device,
)
optimizer = AdamW(model.parameters(), betas=betas, weight_decay=weight_decay)
vocab_filepath = "data/TinyStoriesV2-GPT4-vocab-10000.json"
merges_filepath = "data/TinyStoriesV2-GPT4-merges-10000.txt"
special_tokens = ["<|endoftext|>"]

load_checkpoint("checkpoints/TinyStoriesV2-iter-49999", model, optimizer, device)

tokenizer = load_tokenizer(vocab_filepath, merges_filepath, special_tokens)

print(
    decoder(
        model,
        tokenizer,
        "Once upon a time",
        0.5,
        256,
    )
)
