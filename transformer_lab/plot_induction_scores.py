import json
import random
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as t

import wandb
from transformer_lab.cross_entropy import cross_entropy
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
vocab_filepath = "data/TinyStoriesV2-GPT4-vocab-10000.json"
merges_filepath = "data/TinyStoriesV2-GPT4-merges-10000.txt"
special_tokens = ["<|endoftext|>"]
tokenizer = load_tokenizer(vocab_filepath, merges_filepath, special_tokens)

k = PROBE_CONFIG["repeat_length"]
num_prompts = PROBE_CONFIG["num_prompts"]
prompts = t.nn.functional.pad(
    t.randint(PROBE_CONFIG["sample_lower"], PROBE_CONFIG["sample_upper"], (num_prompts, k)).repeat((1, 2)), (1, 0)
)

to_investigate = [
    (layer_idx, head_idx)
    for layer_idx in range(MODEL_CONFIG["num_layers"])
    for head_idx in range(MODEL_CONFIG["num_heads"])
]

checkpoint_filepaths = sorted(glob("checkpoints/TinyStoriesV2-iter-*"), key=lambda x: int(x.split("-")[-1]))
valid_loss_filepath = Path("analysis/TinyStoriesV2-valid-loss.json")


def zero_attention_head(head_idx: int):
    def zero_specific_head(_module, _input, output):
        output_copy = output.clone().detach()
        output_copy[..., head_idx, :, :] = 0
        return output_copy

    return zero_specific_head


scores = np.empty((len(to_investigate), len(checkpoint_filepaths)))
first_copy_losses = np.empty(len(checkpoint_filepaths))
ablated_first_copy_losses = np.empty((len(to_investigate), len(checkpoint_filepaths)))
second_copy_losses = np.empty(len(checkpoint_filepaths))
ablated_second_copy_losses = np.empty((len(to_investigate), len(checkpoint_filepaths)))

for checkpoint_idx, checkpoint_filepath in enumerate(checkpoint_filepaths):
    obj = t.load(checkpoint_filepath, map_location=device, weights_only=True, mmap=True)
    model.load_state_dict(obj["model_state"])

    output, cache = run_with_cache(
        model,
        tokenizer,
        prompts,
        device,
    )

    first_copy_losses[checkpoint_idx] = cross_entropy(output[:, :k], prompts[:, 1 : k + 1]).item()
    second_copy_losses[checkpoint_idx] = cross_entropy(output[:, k:-1], prompts[:, k + 1 :]).item()

    for idx, (layer_idx, head_idx) in enumerate(to_investigate):
        ablated_output, _ = run_with_cache(
            model,
            tokenizer,
            prompts,
            device,
            {f"layers.{layer_idx}.attn.hook_pattern": zero_attention_head(head_idx)},
        )

        attn_pattern = cache[f"layers.{layer_idx}.attn.hook_pattern"][:, head_idx]

        scores[idx][checkpoint_idx] = attn_pattern.diagonal(offset=-k + 1, dim1=-2, dim2=-1)[:, 2:].mean().item()

        ablated_first_copy_losses[idx][checkpoint_idx] = cross_entropy(
            ablated_output[:, :k], prompts[:, 1 : k + 1]
        ).item()
        ablated_second_copy_losses[idx][checkpoint_idx] = cross_entropy(
            ablated_output[:, k:-1], prompts[:, k + 1 :]
        ).item()

labels = [int(filepath.split("-")[-1]) for filepath in checkpoint_filepaths]

# get valid_loss
if valid_loss_filepath.is_file():
    with open(valid_loss_filepath) as valid_loss_file:
        valid_loss = json.load(valid_loss_file)
else:
    # get last checkpoint's wandb_run_id
    wandb_run_id = obj["wandb_run_id"]

    api = wandb.Api()
    wandb_run = api.run(f"/cs336-a1/{wandb_run_id}")
    rows = [*wandb_run.scan_history(keys=["_step", "valid/loss"])]

    valid_loss = [row["valid/loss"] for row in rows]

    assert labels == [row["_step"] for row in rows]

    with open(valid_loss_filepath, "w") as valid_loss_file:
        json.dump(valid_loss, valid_loss_file)

# plot figures
for idx, (layer_idx, head_idx) in enumerate(to_investigate):
    fig, (valid_loss_axis, induction_axis, copy_loss_axis, gap_loss_axis) = plt.subplots(
        nrows=4, sharex=True, figsize=(8, 10)
    )

    valid_loss_axis.plot(labels, valid_loss)
    induction_axis.plot(labels, scores[idx])
    copy_loss_axis.plot(labels, first_copy_losses, label="first (not ablated)")
    copy_loss_axis.plot(labels, ablated_first_copy_losses[idx], label="first (ablated)")
    copy_loss_axis.plot(labels, second_copy_losses, label="second (not ablated)")
    copy_loss_axis.plot(labels, ablated_second_copy_losses[idx], label="second (ablated)")
    copy_loss_axis.legend()
    gap_loss_axis.plot(
        labels,
        [first - second for first, second in zip(first_copy_losses, second_copy_losses)],
        label="not ablated",
    )
    gap_loss_axis.plot(
        labels,
        [first - second for first, second in zip(ablated_first_copy_losses[idx], ablated_second_copy_losses[idx])],
        label="ablated",
    )
    gap_loss_axis.legend()

    valid_loss_axis.set_ylabel("valid/loss")
    induction_axis.set_ylabel("induction score")
    copy_loss_axis.set_ylabel("copy losses")
    gap_loss_axis.set_ylabel("loss gap (first - second)")

    gap_loss_axis.set_xlabel("training iteration")

    for axis in [valid_loss_axis, induction_axis, copy_loss_axis, gap_loss_axis]:
        axis.set_xscale("log")
        axis.axvline(OPTIMIZER_CONFIG["T_w"], label="warmup ends", color="red")

    plt.savefig(f"figures/induction_scores_{layer_idx}.{head_idx}.png")
    plt.close()
