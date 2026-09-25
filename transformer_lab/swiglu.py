import torch as t

from transformer_lab.linear import Linear


class SwiGLU(t.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: t.device | None = None, dtype: t.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def silu(self, x: t.Tensor) -> t.Tensor:
        return x * t.sigmoid(x)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.w2(self.silu(self.w1(x)) * self.w3(x))
