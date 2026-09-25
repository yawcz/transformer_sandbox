import random

import torch as t

from transformer_lab.adamw import AdamW
from transformer_lab.checkpointing import load_checkpoint
from transformer_lab.experiment_config import MODEL_CONFIG, OPTIMIZER_CONFIG, PROBE_CONFIG
from transformer_lab.interp_tooling import run_with_cache
from transformer_lab.training_together import load_tokenizer
from transformer_lab.transformer_lm import TransformerLM

random.seed(PROBE_CONFIG["random_seed"])
t.manual_seed(PROBE_CONFIG["random_seed"])

device = t.device("cuda" if t.cuda.is_available() else "cpu")

model = TransformerLM(
    vocab_size=MODEL_CONFIG["vocab_size"],
    context_length=MODEL_CONFIG["context_length"],
    d_model=MODEL_CONFIG["d_model"],
    num_layers=MODEL_CONFIG["num_layers"],
    num_heads=MODEL_CONFIG["num_heads"],
    d_ff=MODEL_CONFIG["d_ff"],
    rope_theta=MODEL_CONFIG["rope_theta"],
    device=device,
)
optimizer = AdamW(model.parameters(), betas=OPTIMIZER_CONFIG["betas"], weight_decay=OPTIMIZER_CONFIG["weight_decay"])
vocab_filepath = "data/TinyStoriesV2-GPT4-vocab-10000.json"
merges_filepath = "data/TinyStoriesV2-GPT4-merges-10000.txt"
special_tokens = ["<|endoftext|>"]

load_checkpoint("checkpoints/TinyStoriesV2-iter-49999", model, optimizer, device)

tokenizer = load_tokenizer(vocab_filepath, merges_filepath, special_tokens)

repeat_length = PROBE_CONFIG["repeat_length"]
num_prompts = PROBE_CONFIG["num_prompts"]
prompts = t.nn.functional.pad(
    t.randint(PROBE_CONFIG["sample_lower"], PROBE_CONFIG["sample_upper"], (num_prompts, repeat_length)).repeat((1, 2)),
    (1, 0),
)

output, cache = run_with_cache(model, tokenizer, prompts, device)

print("Induction heads:")

values, indices = t.topk(
    t.cat(
        [
            value.diagonal(offset=-repeat_length + 1, dim1=-2, dim2=-1)[:, :, 2:].mean(dim=-1).mean(dim=0)
            for value in cache.values()
        ]
    ),
    k=3,
)

for score, idx in zip(values, indices):
    layer_idx, head_idx = divmod(idx.item(), MODEL_CONFIG["num_heads"])
    print(f"{layer_idx}.{head_idx} {score.item():.2f}")
