import random
import string

import matplotlib.pyplot as plt
import seaborn as sns
import torch as t

from transformer_lab.adamw import AdamW
from transformer_lab.checkpointing import load_checkpoint
from transformer_lab.interp_tooling import run_with_cache
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


def visualize(label, attn_pattern, x_label):
    for head in range(num_heads):
        sns.heatmap(attn_pattern[head].cpu(), cmap="Blues", vmin=0, vmax=1, xticklabels=x_label, yticklabels=x_label)
        plt.title(f"{label}.{head}")
        plt.savefig(f"figures/{label}.{head}.png")
        plt.clf()


characters = string.ascii_letters + string.digits  # contains a-z, A-Z, 0-9

prompt = "".join(random.sample(characters, k=20)) * 2
token_str = [tokenizer.decode([x]) for x in tokenizer.encode(prompt)]

output, cache = run_with_cache(model, tokenizer, prompt, device)

for key, value in cache.items():
    visualize(key, value, token_str)
