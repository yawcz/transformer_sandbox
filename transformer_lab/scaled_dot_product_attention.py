import math

import torch as t
from einops import einsum

from transformer_lab.softmax import softmax


def scaled_dot_product_attention(
    queries: t.Tensor,
    keys: t.Tensor,
    values: t.Tensor,
    mask: t.Tensor | None = None,
    hook_pattern: t.nn.Module | None = None,
) -> t.Tensor:
    d_k = queries.shape[-1]
    pre_softmax = einsum(queries, keys, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
    if mask is not None:
        pre_softmax.masked_fill_(~mask, -t.inf)

    post_softmax = softmax(pre_softmax, dim=-1)

    if hook_pattern is not None:
        post_softmax = hook_pattern(post_softmax)

    return einsum(post_softmax, values, "... queries keys, ... keys d_v -> ... queries d_v")
