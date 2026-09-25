import json
import os
import time
from itertools import islice

import numpy as np
import torch as t
from tqdm import tqdm

import wandb
from transformer_lab.adamw import AdamW
from transformer_lab.checkpointing import load_checkpoint, load_checkpoint_non_state_keys, save_checkpoint
from transformer_lab.cross_entropy import cross_entropy
from transformer_lab.data_loading import data_loader
from transformer_lab.experiment_config import MODEL_CONFIG, OPTIMIZER_CONFIG, TRAIN_CONFIG
from transformer_lab.gradient_clipping import gradient_clipping
from transformer_lab.lr_cosine_schedule import lr_cosine_schedule
from transformer_lab.tokenizer import Tokenizer, gpt2_bytes_to_unicode
from transformer_lab.train_bpe import build_freq_dict_from_file, do_merges
from transformer_lab.transformer_lm import TransformerLM


def load_tokenizer(
    vocab_filepath: str | os.PathLike, merges_filepath: str | os.PathLike, special_tokens: list[str]
) -> Tokenizer:
    return Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)


def train_and_save_tokenizer(
    dataset_filepath: str,
    vocab_size: int,
    vocab_filepath: str,
    merges_filepath: str,
    special_tokens: list[str],
):
    freq = build_freq_dict_from_file(dataset_filepath, special_tokens, os.cpu_count() or 4)
    vocab, merges = do_merges(freq, vocab_size, special_tokens)

    gpt2_byte_encoder = gpt2_bytes_to_unicode()

    with open(vocab_filepath, "w", encoding="utf-8") as f:
        vocab_raw = {
            "".join(gpt2_byte_encoder[x] for x in vocab_item): vocab_index for vocab_index, vocab_item in vocab.items()
        }

        json.dump(vocab_raw, f)

    with open(merges_filepath, "w", encoding="utf-8") as f:
        for merge_token_1, merge_token_2 in merges:
            f.write(
                "".join(gpt2_byte_encoder[x] for x in merge_token_1)
                + " "
                + "".join(gpt2_byte_encoder[x] for x in merge_token_2)
                + "\n"
            )


def encode_file(tokenizer: Tokenizer, filepath: str, tokenized_filepath: str):
    with open(filepath, encoding="utf-8") as input_file:
        tokens = tokenizer.encode_iterable(input_file)

        with open(tokenized_filepath, mode="wb") as output_file:
            while True:
                chunk = np.fromiter(islice(tokens, 1_000_000), dtype=np.uint16)
                if chunk.size == 0:
                    break
                chunk.tofile(output_file)


def init_tokenizer(
    train_filepath: str,
    valid_filepath: str,
    vocab_filepath: str,
    merges_filepath: str,
    vocab_size: int,
    special_tokens: list[str],
    tokenized_train_filepath: str,
    tokenized_valid_filepath: str,
):
    train_and_save_tokenizer(train_filepath, vocab_size, vocab_filepath, merges_filepath, special_tokens)

    tokenizer = load_tokenizer(vocab_filepath, merges_filepath, special_tokens)
    encode_file(tokenizer=tokenizer, filepath=train_filepath, tokenized_filepath=tokenized_train_filepath)
    encode_file(tokenizer=tokenizer, filepath=valid_filepath, tokenized_filepath=tokenized_valid_filepath)


def validate(
    model: TransformerLM,
    valid_data: np.memmap,
    batch_size: int,
    context_length: int,
    num_valid_iters: int,
    device: str = "cpu",
) -> float:
    total_valid_loss: t.Tensor = t.zeros(1, device=t.device(device))

    model.eval()

    with t.inference_mode():
        for _ in range(num_valid_iters):
            current_batch = data_loader(valid_data, batch_size, context_length, device)
            total_valid_loss += cross_entropy(model(current_batch[0]), current_batch[1])

    model.train()

    return (total_valid_loss / num_valid_iters).item()


def train_model(
    tokenized_train_filepath: str,
    tokenized_valid_filepath: str,
    num_iters: int,
    num_valid_iters: int,
    eval_steps: int,
    batch_size: int,
    vocab_size: int,
    context_length: int,
    d_model: int,
    d_ff: int,
    rope_theta: float,
    num_layers: int,
    num_heads: int,
    a_max: float,
    a_min: float,
    T_w: int,
    T_c: int,
    betas: tuple[float, float],
    weight_decay: float,
    max_l2_norm: float,
    wandb_run_id: str | None = None,
    checkpoint_filepath: str | None = None,
    save_checkpoint_filepath: str | None = None,
    device: str = "cpu",
):
    assert T_w <= T_c <= num_iters

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=t.device(device),
    )
    optimizer = AdamW(model.parameters(), betas=betas, weight_decay=weight_decay)
    train_data = np.memmap(tokenized_train_filepath, dtype=np.uint16, mode="r")
    valid_data = np.memmap(tokenized_valid_filepath, dtype=np.uint16, mode="r")

    t0 = time.perf_counter()

    if checkpoint_filepath is not None:
        current_iter = 1 + load_checkpoint(checkpoint_filepath, model, optimizer, t.device(device))
        checkpoint_obj = load_checkpoint_non_state_keys(checkpoint_filepath, t.device(device))
        t0 -= checkpoint_obj["elapsed_seconds"]
    else:
        current_iter = 0

    loader = tqdm(range(current_iter, num_iters))

    for idx in loader:
        lr = lr_cosine_schedule(idx + 1, a_max, a_min, T_w, T_c)
        for group in optimizer.param_groups:
            group["lr"] = lr
        current_batch = data_loader(train_data, batch_size, context_length, device)
        optimizer.zero_grad()
        loss = cross_entropy(model(current_batch[0]), current_batch[1])
        loss.backward()
        pre_clip_grad_norm = gradient_clipping(model.parameters(), max_l2_norm=max_l2_norm)
        optimizer.step()

        current_train_loss = loss.item()
        loader.set_postfix(loss=current_train_loss)
        wandb.log(
            {
                "train/loss": current_train_loss,
                "lr": lr,
                "tokens": (idx + 1) * batch_size * context_length,
                "wallclock": time.perf_counter() - t0,
            },
            step=idx,
        )
        if pre_clip_grad_norm is not None:
            wandb.log({"pre-clip-grad-norm": pre_clip_grad_norm.item()}, step=idx)

        if (idx + 1) % eval_steps == 0:
            valid_loss = validate(
                model=model,
                valid_data=valid_data,
                batch_size=batch_size,
                context_length=context_length,
                num_valid_iters=num_valid_iters,
                device=device,
            )
            wandb.log({"valid/loss": valid_loss}, step=idx)
            if save_checkpoint_filepath is not None:
                save_checkpoint(
                    model,
                    optimizer,
                    idx,
                    f"checkpoints/{save_checkpoint_filepath}-iter-{idx}",
                    wandb_run_id=wandb_run_id,
                    elapsed_seconds=time.perf_counter() - t0,
                    train_loss=loss.item(),
                    valid_loss=valid_loss,
                )
                tqdm.write(f"Checkpoint saved at iteration {idx}. Train loss: {loss.item()}. Valid loss: {valid_loss}.")
            else:
                tqdm.write(f"Iteration {idx}. Train loss: {loss.item()}. Valid loss: {valid_loss}.")

    return model


def main():
    t.manual_seed(43)
    np.random.seed(43)
    dataset_name = "TinyStoriesV2-GPT4"
    train_filepath = "data/TinyStoriesV2-GPT4-train.txt"
    valid_filepath = "data/TinyStoriesV2-GPT4-valid.txt"
    special_tokens = ["<|endoftext|>"]
    save_checkpoint_filepath = "TinyStoriesV2"
    # checkpoint_filepath = "checkpoints/TinyStoriesV2-GPT4-train-tokenized-10000-iter-19999"
    checkpoint_filepath = None
    device = "cuda" if t.cuda.is_available() else "cpu"

    vocab_size = MODEL_CONFIG["vocab_size"]

    vocab_filepath = f"data/{dataset_name}-vocab-{vocab_size}.json"
    merges_filepath = f"data/{dataset_name}-merges-{vocab_size}.txt"
    tokenized_train_filepath = f"data/{dataset_name}-train-tokenized-{vocab_size}.bin"
    tokenized_valid_filepath = f"data/{dataset_name}-valid-tokenized-{vocab_size}.bin"

    should_tokenize = False

    if should_tokenize:
        print("Tokenizing...")
        init_tokenizer(
            train_filepath=train_filepath,
            valid_filepath=valid_filepath,
            vocab_filepath=vocab_filepath,
            merges_filepath=merges_filepath,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            tokenized_train_filepath=tokenized_train_filepath,
            tokenized_valid_filepath=tokenized_valid_filepath,
        )
        print("Tokenized.")

    config = {
        **MODEL_CONFIG,
        **OPTIMIZER_CONFIG,
        **TRAIN_CONFIG,
        "checkpoint_filepath": checkpoint_filepath,
        "save_checkpoint_filepath": save_checkpoint_filepath,
        "device": device,
    }

    if checkpoint_filepath is not None:
        checkpoint_obj = load_checkpoint_non_state_keys(checkpoint_filepath, device=t.device(device))

        run = wandb.init(
            project="cs336-a1",
            name="tinystories",
            config=config,
            id=checkpoint_obj["wandb_run_id"],
            resume="must",
        )
    else:
        run = wandb.init(
            project="cs336-a1",
            name="tinystories",
            config=config,
        )

    wandb.define_metric("tokens")
    wandb.define_metric("wallclock")
    wandb.define_metric("train/loss", step_metric="tokens")
    wandb.define_metric("valid/*", step_metric="tokens")  # globs work

    print(f"wandb run {run.id} initialised.")

    try:
        train_model(
            tokenized_train_filepath=tokenized_train_filepath,
            tokenized_valid_filepath=tokenized_valid_filepath,
            wandb_run_id=run.id,
            **config,
        )
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
