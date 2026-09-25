import math

import torch as t
from einops import einsum


class Linear(t.nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: t.device | None = None, dtype: t.dtype | None = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = t.nn.Parameter(t.empty((out_features, in_features), dtype=dtype, device=device))
        sigma = math.sqrt(2 / (in_features + out_features))
        t.nn.init.trunc_normal_(self.weight, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return einsum(self.weight, x, "out_features in_features, ... in_features -> ... out_features")
