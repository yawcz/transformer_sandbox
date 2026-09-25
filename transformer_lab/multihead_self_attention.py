import torch as t
from einops import rearrange
from jaxtyping import Int

from transformer_lab.hookpoint import HookPoint
from transformer_lab.linear import Linear
from transformer_lab.rope import RoPE
from transformer_lab.scaled_dot_product_attention import scaled_dot_product_attention


class MultiheadSelfAttention(t.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: t.device | None = None,
        dtype: t.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = Linear(self.d_model, self.d_model, device, dtype)
        self.k_proj = Linear(self.d_model, self.d_model, device, dtype)
        self.v_proj = Linear(self.d_model, self.d_model, device, dtype)
        self.output_proj = Linear(self.d_model, self.d_model, device, dtype)
        self.hook_pattern = HookPoint()
        if max_seq_len is None:
            self.rope = None
        else:
            assert theta is not None
            self.rope = RoPE(theta, d_model // num_heads, max_seq_len, device)

    def multi_head(self, q, k, v, token_positions: Int[t.Tensor, "..."] | None = None):
        mask = t.tril(t.ones((q.shape[-2], k.shape[-2]), dtype=t.bool, device=q.device))
        q_split = rearrange(q, "... seq (h q_i) -> ... h seq q_i", h=self.num_heads)
        k_split = rearrange(k, "... seq (h k_i) -> ... h seq k_i", h=self.num_heads)
        v_split = rearrange(v, "... seq (h v_i) -> ... h seq v_i", h=self.num_heads)
        if self.rope is not None:
            q_split = self.rope(q_split, token_positions)
            k_split = self.rope(k_split, token_positions)
        before_stack = scaled_dot_product_attention(q_split, k_split, v_split, mask, self.hook_pattern)
        return rearrange(before_stack, "... h seq ans_i -> ... seq (h ans_i)")

    def forward(self, in_features: t.Tensor, token_positions: Int[t.Tensor, "..."] | None = None) -> t.Tensor:
        if self.rope is not None and token_positions is None:
            token_positions = t.arange(in_features.shape[-2], device=in_features.device)
        return self.output_proj(
            self.multi_head(
                self.q_proj(in_features), self.k_proj(in_features), self.v_proj(in_features), token_positions
            )
        )
