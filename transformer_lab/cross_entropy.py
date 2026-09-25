import torch as t
from einops import rearrange
from jaxtyping import Float, Int


def cross_entropy(inputs: Float[t.Tensor, "... vocab_size"], targets: Int[t.Tensor, "..."]) -> Float[t.Tensor, ""]:
    reshaped_inputs = rearrange(inputs, "... vocab_size -> (...) vocab_size")
    reshaped_targets = rearrange(targets, "... -> (...)")
    subtracted_inputs = reshaped_inputs - t.max(reshaped_inputs, dim=-1, keepdim=True).values
    return t.mean(
        t.logsumexp(subtracted_inputs, dim=-1)
        - subtracted_inputs[t.arange(reshaped_inputs.shape[0], device=inputs.device), reshaped_targets]
    )
