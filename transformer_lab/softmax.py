import torch as t


def softmax(x: t.Tensor, dim: int) -> t.Tensor:
    tmp = t.exp(x - t.max(x, dim=dim, keepdim=True).values)
    x_out = tmp / t.sum(tmp, dim=dim, keepdim=True)

    return x_out
