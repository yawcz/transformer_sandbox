# Building a Transformer from scratch

This is a self-directed project of building and training a Transformer from scratch, then using basic interpretability techniques to investigate what it learnt. It is my first sustained attempt at understanding how language models work, and is also my introduction to mechanistic interpretability.

I followed Stanford's CS336 Assignment 1 for the tokenizer, model, and training pipeline, and extended it with a first investigation into induction heads.

The model was trained on TinyStories for about seven hours on my laptop. It produces small stories, and one of its attention heads contributes to predicting repeated token sequences.

Read [WRITEUP.md](WRITEUP.md) for a detailed account of the project.

## Running this project

If, for whatever reason, you want to clone this repo and run some scripts, here is how to use them. Run all commands from the repository root.

### Environment and tests

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). This project requires Python 3.14 because the BPE trainer uses its public max-heap functions. Runtime dependencies and development tools are listed separately in [pyproject.toml](pyproject.toml), with resolved versions recorded in [uv.lock](uv.lock). The commands below install both groups, including the plotting libraries.

```sh
uv sync --locked
uv run --locked pytest
```

The tests use the fixtures included in the repository. You do not need to download the training corpus or obtain my checkpoints to run them. The scripts select CUDA when it is available and otherwise use the CPU. The seven-hour training time was on an NVIDIA GeForce RTX 5060 Max-Q / Mobile.

### Data

The run described in the write-up uses TinyStories. Download both splits:

```sh
mkdir -p data checkpoints
curl -L --fail -o data/TinyStoriesV2-GPT4-train.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
curl -L --fail -o data/TinyStoriesV2-GPT4-valid.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
```

The raw data, trained tokenizer, tokenized data, checkpoints, and local wandb logs are excluded from Git. Cloning the repository gives you the code and figures, but you will need to generate those artifacts yourself.

### Prepare the data and train

The entry point is [training_together.py](transformer_lab/training_together.py). Before the first run:

1. Review [experiment_config.py](transformer_lab/experiment_config.py). The current settings are for 50,000 updates, batch size 52, and a context length of 256.
2. In the training script's main function, change the existing `should_tokenize` switch to `True`. This trains the BPE tokenizer and encodes both splits before training the model. On later runs, turn it off to reuse the generated files.
3. Check the paths in the same function. For a fresh run, leave `checkpoint_filepath` at its existing `None` default. Resuming requires setting it to an existing checkpoint. Change `save_checkpoint_filepath` before starting a separate experiment so its checkpoints do not overwrite the previous run.
4. Choose how to record the run. For online [Weights & Biases](https://docs.wandb.ai/models/track/environment-variables) logging, authenticate first:

```sh
uv run --locked wandb login
uv run --locked -m transformer_lab.training_together
```

The script uses the W&B project `cs336-a1`. For a fresh run with local logging, use this command instead:

```sh
WANDB_MODE=offline uv run --locked -m transformer_lab.training_together
```

Offline mode stores the logs locally. The current resume path expects to rejoin an existing online W&B run.

With the default paths and vocabulary size, preprocessing creates:

```text
data/TinyStoriesV2-GPT4-vocab-10000.json
data/TinyStoriesV2-GPT4-merges-10000.txt
data/TinyStoriesV2-GPT4-train-tokenized-10000.bin
data/TinyStoriesV2-GPT4-valid-tokenized-10000.bin
```

Training saves a checkpoint every 500 updates. The filenames use zero-based iteration indices, so the first is `checkpoints/TinyStoriesV2-iter-499` and the last is `checkpoints/TinyStoriesV2-iter-49999`. Keep the tokenizer files with their corresponding checkpoints.

### Generate text

[sandbox.py](transformer_lab/sandbox.py) loads the final checkpoint and prints a sample:

```sh
uv run --locked -m transformer_lab.sandbox
```

It currently expects `checkpoints/TinyStoriesV2-iter-49999` and the tokenizer files listed above. The prompt is `Once upon a time`, the temperature is 0.5, and the length limit is 256 tokens including the prompt. These settings are in the script. If you trained a different model configuration, update its local model settings and checkpoint path to match.

The prompt must be non-empty. For longer outputs, each prediction uses the latest `context_length` tokens, while the returned text keeps the full prompt and continuation.

### Inspect attention and induction heads

Once the tokenizer and checkpoints are available, these scripts reproduce the different parts of the investigation:

| Script | Description |
| --- | --- |
| [detect_induction_heads.py](transformer_lab/detect_induction_heads.py) | Prints the three highest-scoring heads on a repeated-token probe at the final checkpoint. |
| [visualize_attention.py](transformer_lab/visualize_attention.py) | Saves attention heatmaps for a repeated random character string. |
| [plot_induction_scores.py](transformer_lab/plot_induction_scores.py) | Sweeps saved checkpoints and compares probe losses with and without each head. |

The first two scripts expect the final checkpoint path used above. The plotting script loads all files matching `checkpoints/TinyStoriesV2-iter-*`; keep that set to a single training run. Model and probe settings for detection and plotting live in the shared configuration. The heatmap script still has its own model settings.

Run the scripts from the same environment used for training:

```sh
uv run --locked -m transformer_lab.detect_induction_heads
uv run --locked -m transformer_lab.visualize_attention
uv run --locked -m transformer_lab.plot_induction_scores
```

Both plotting scripts write to `figures/`. The sweep evaluates all 64 heads across the checkpoint sequence.

The included [validation-loss cache](analysis/TinyStoriesV2-valid-loss.json) belongs to my original run. For a new run, replace it with that run's validation losses in numerical checkpoint order. If the cache is absent, the script tries to fetch the history from wandb using the saved run ID and its configured project path.
