from collections.abc import Iterable

import torch as t


def gradient_clipping(parameters: Iterable[t.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> t.Tensor | None:
    parameters_list = list(parameters)

    if len(parameters_list) == 0:
        return

    grad_l2_norm = t.linalg.vector_norm(
        t.stack([t.linalg.vector_norm(p.grad) for p in parameters_list if p.grad is not None])
    )

    if grad_l2_norm >= max_l2_norm:
        for p in parameters_list:
            if p.grad is not None:
                p.grad *= max_l2_norm / (grad_l2_norm + eps)

    return grad_l2_norm
