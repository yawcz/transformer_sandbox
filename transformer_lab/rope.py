import torch as t
from einops import rearrange


class RoPE(t.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: t.device | None = None):
        assert d_k % 2 == 0
        super().__init__()
        indices = t.arange(max_seq_len, device=device)
        freqs = 1 / t.pow(theta, t.arange(0, d_k, 2, device=device) / d_k)

        angles = t.outer(indices, freqs)

        self.register_buffer("sin_precomp", t.sin(angles), False)
        self.register_buffer("cos_precomp", t.cos(angles), False)

    def forward(self, x: t.Tensor, token_positions: t.LongTensor) -> t.Tensor:
        in_dtype = x.dtype

        x_pairs = rearrange(x, "... (pair two) -> ... pair two", two=2)
        x1, x2 = x_pairs[..., 0], x_pairs[..., 1]
        sin_positions = self.sin_precomp[token_positions]
        cos_positions = self.cos_precomp[token_positions]
        x1, x2 = x1 * cos_positions - x2 * sin_positions, x1 * sin_positions + x2 * cos_positions

        return rearrange(t.stack((x1, x2), dim=-1), "... pair two -> ... (pair two)").to(in_dtype)
