import os
import typing

import torch as t


def save_checkpoint(
    model: t.nn.Module,
    optimizer: t.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    **kwargs,
) -> None:
    obj = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
        **kwargs,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t.save(obj, out)


def load_checkpoint_non_state_keys(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    device: t.device | None = None,
):
    obj = t.load(src, map_location=device)
    return {k: v for k, v in obj.items() if k not in ["model_state", "optimizer_state", "iteration"]}


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: t.nn.Module,
    optimizer: t.optim.Optimizer,
    device: t.device | None = None,
) -> int:
    obj = t.load(src, map_location=device)
    model.load_state_dict(obj["model_state"])
    optimizer.load_state_dict(obj["optimizer_state"])
    return obj["iteration"]
