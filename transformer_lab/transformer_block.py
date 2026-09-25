import torch as t

from transformer_lab.multihead_self_attention import MultiheadSelfAttention
from transformer_lab.rmsnorm import RMSNorm
from transformer_lab.swiglu import SwiGLU


class TransformerBlock(t.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: t.device | None = None,
        dtype: t.dtype | None = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta, device, dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    def forward(self, x: t.Tensor) -> t.Tensor:
        y = x + self.attn(self.ln1(x))
        z = y + self.ffn(self.ln2(y))
        return z
