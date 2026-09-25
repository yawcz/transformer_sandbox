import numpy as np
import numpy.typing as npt
import torch as t


def data_loader(x: npt.NDArray, batch_size: int, context_length: int, device: str = "cpu") -> tuple[t.Tensor, t.Tensor]:
    starting_indices = np.random.randint(x.size - context_length, size=batch_size)
    indices = starting_indices[:, None] + np.arange(context_length)[None, :]

    return (
        t.tensor(x[indices], dtype=t.long, device=device),
        t.tensor(x[indices + 1], dtype=t.long, device=device),
    )
