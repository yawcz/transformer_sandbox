import torch as t


class RMSNorm(t.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: t.device | None = None, dtype: t.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = t.nn.Parameter(t.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: t.Tensor) -> t.Tensor:
        in_dtype = x.dtype
        x = x.to(t.float32)
        rms = t.rsqrt(t.mean(x * x, dim=-1, keepdim=True) + self.eps)
        result = x * rms * self.weight

        return result.to(in_dtype)
